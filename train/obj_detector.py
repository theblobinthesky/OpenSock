import jax, jax.nn as nn, jax.numpy as jnp, jax.lax as lax
import jax.nn.initializers as init
from config import TrainConfig, DataConfig, ModelConfig
from data import get_obj_detection_dataset
from train import train_model
from utils import non_maximum_supression, average_precision
from utils import quality_focal_loss, distribution_focal_loss, general_iou_loss

DIM_NUMS = ('NHWC', 'HWIO', 'NHWC')
MAX_NUMBER_OF_BBOXES = 10_000

class Initializer:
    def __init__(self):
        self.key = jax.random.PRNGKey(0)
        self.kernel_init = init.glorot_uniform()
    
    def __call__(self, shape: tuple[int, ...]):
        k1, k2 = jax.random.split(self.key)
        self.key = k1
        return self.kernel_init(k2, shape)

init_fn = Initializer()

def batch_norm(x: jnp.ndarray) -> jnp.ndarray:
    # TODO: Implement the proper batchnorm!
    mean = jnp.mean(x, axis=(0, 1, 2), keepdims=True)
    var = jnp.var(x, axis=(0, 1, 2), keepdims=True)
    return (x - mean) / jnp.sqrt(var + 1e-5)

def conv(config: ModelConfig, params: dict, x: jnp.ndarray, stride: int=1) -> jnp.ndarray:
    # print(f"{x.shape=}, {params['weights'].shape=}")
    x = lax.conv_general_dilated(x, params['weights'], (stride, stride), 'SAME', dimension_numbers=DIM_NUMS)
    x = x + params['biases']
    x = batch_norm(x)
    x = nn.silu(x)
    
    return x

def init_conv(chs_in: int, chs_out: int, kernel_size: int=1) -> dict:
    return {
        'weights': init_fn((kernel_size, kernel_size, chs_in, chs_out)),
        'biases': jnp.zeros((chs_out,))
    }

def residual(config: ModelConfig, params: dict, x: jnp.array) -> jnp.array:
    residual = x
    x = conv(config, params['conv1'], x)
    x = conv(config, params['conv2'], x)

    if x.shape == residual.shape: 
        x = x + residual
    
    return x

def init_residual(chs: int) -> dict:
    return {
        'conv1': init_conv(chs, chs, kernel_size=3),
        'conv2': init_conv(chs, chs, kernel_size=3)
    }

# Cross Stage Partial Layer
# See https://arxiv.org/pdf/1911.11929
def csp(config: ModelConfig, params: dict, x: jnp.array, n: int=1) -> jnp.array:
    y = [conv(config, params['conv1'], x), conv(config, params['conv2'], x)]
    y.extend(residual(config, params['residuals'][f"residual{i}"], y[-1]) for i in range(n))
    x = conv(config, params['conv3'], jnp.concat(y, axis=-1))
    
    return x

def init_csp(chs_in: int, chs_out: int, n: int=1) -> dict:
    return {
        'conv1': init_conv(chs_in, chs_out // 2),
        'conv2': init_conv(chs_in, chs_out // 2),
        'conv3': init_conv((2 + n) * chs_out // 2, chs_out),
        'residuals': {
            f'residual{i}': init_residual(chs_out // 2)
            for i in range(n)
        }
    }  

def max_pool_2d(x: jnp.ndarray, kernel_size: int, stride: int, padding: int) -> jnp.ndarray:
    x = lax.reduce_window(
        operand=x, 
        init_value=-jnp.inf, 
        computation=lax.max, 
        window_dimensions=(1, kernel_size, kernel_size, 1), 
        window_strides=(1, stride, stride, 1), 
        padding=[(0, 0), (padding, padding), (padding, padding), (0, 0)]
    )

    return x

def spp(config: ModelConfig, params: dict, x: jnp.array, pool_kernel_size: int=5) -> jnp.array:
    x = conv(config, params['conv1'], x)
    pool1 = max_pool_2d(x, kernel_size=pool_kernel_size, stride=1, padding=pool_kernel_size // 2)
    pool2 = max_pool_2d(pool1, kernel_size=pool_kernel_size, stride=1, padding=pool_kernel_size // 2)
    pool3 = max_pool_2d(pool2, kernel_size=pool_kernel_size, stride=1, padding=pool_kernel_size // 2)
    x = jnp.concat([x, pool1, pool2, pool3], axis=-1)
    return conv(config, params['conv2'], x)

def init_spp(chs_in: int, chs_out: int) -> dict:
    return {
        'conv1': init_conv(chs_in, chs_out // 2),
        'conv2': init_conv(chs_in * 2, chs_out)
    }

def dark_net_layer(config: ModelConfig, params: dict, x: jnp.ndarray, n: int=1) -> jnp.ndarray:
    x = conv(config, params['conv'], x, stride=2)
    x = csp(config, params['csp'], x, n=n)
    return x

def dark_net(config: ModelConfig, params: dict, x: jnp.ndarray) -> jnp.ndarray:
    depth: list[int] = config['depth']

    p1 = conv(config, params['p1_conv'], x, stride=2)
    p2 = dark_net_layer(config, params['p2'], p1, n=depth[0]) 
    p3 = dark_net_layer(config, params['p3'], p2, n=depth[1]) 
    p4 = dark_net_layer(config, params['p4'], p3, n=depth[2]) 
    p5 = dark_net_layer(config, params['p5'], p4, n=depth[0]) 
    p5 = spp(config, params['p5']['spp'], p5)
    
    return p3, p4, p5

def init_dark_net(config: ModelConfig) -> dict:
    width: list[int] = config['width']
    depth: list[int] = config['depth']

    return {
        'p1_conv': init_conv(width[0], width[1]),
        'p2': {
            'conv': init_conv(width[1], width[2]),
            'csp': init_csp(width[2], width[2], depth[0])
        },
        'p3': {
            'conv': init_conv(width[2], width[3]),
            'csp': init_csp(width[3], width[3], depth[1])
        },
        'p4': {
            'conv': init_conv(width[3], width[4]),
            'csp': init_csp(width[4], width[4], depth[2])
        },
        'p5': {
            'conv': init_conv(width[4], width[5]),
            'csp': init_csp(width[5], width[5], depth[0]),
            'spp': init_spp(width[5], width[5])
        }
    }

def upsample(x: jnp.ndarray, scale: int=2):
    b, h, w, c = x.shape
    x = jax.image.resize(x, (b, h * scale, w * scale, c), 'nearest')
    
    return x

def dark_fpn(config: ModelConfig, params: dict, x: jnp.ndarray) -> jnp.ndarray:
    depth: list[int] = config['depth']

    p3, p4, p5 = x
  
    # Pyramid: Up
    h1 = csp(config, params['h1'], jnp.concat([p4, upsample(p5)], axis=-1), n=depth[0]) 
    h2 = csp(config, params['h2'], jnp.concat([p3, upsample(h1)], axis=-1), n=depth[0])
  
    # Pyramid: Down
    h4 = csp(config, params['h4'], jnp.concat([p4, conv(config, params['h3'], h2, stride=2)], axis=-1), n=depth[0])
    h6 = csp(config, params['h6'], jnp.concat([p5, conv(config, params['h5'], h4, stride=2)], axis=-1), n=depth[0])
    
    return h2, h4, h6 
    
def init_dark_fpn(config: ModelConfig):
    width: list[int] = config['width']
    depth: list[int] = config['depth']

    return {
        'h1': init_csp(width[4] + width[5], width[4], depth[0]), # TODO: False param is missing
        'h2': init_csp(width[3] + width[4], width[3], depth[0]),
        'h3': init_conv(width[3], width[3], kernel_size=3),
        'h4': init_csp(width[3] + width[4], width[4], depth[0]),
        'h5': init_conv(width[4], width[4], kernel_size=3),
        'h6': init_csp(width[4] + width[5], width[5], depth[0]),
    }

def detection_head_layer(config: ModelConfig, params: dict, s: jnp.ndarray) -> jnp.ndarray:
    s = conv(config, params['conv1'], s)
    s = conv(config, params['conv2'], s)
    
    s = lax.conv_general_dilated(s, params['conv3_weights'], (1, 1), 'SAME', dimension_numbers=DIM_NUMS)
    s = s + params['conv3_biases']
    return s

def detection_head(config: ModelConfig, params: dict, x: list[jnp.ndarray]) -> jnp.ndarray:
    scales = []
    dfl_channels = config['dfl_channels']

    for class_params, box_params, s in zip(params['class_head'], params['box_head'], x):
        class_pred = nn.sigmoid(detection_head_layer(config, class_params, s))

        box_pred = detection_head_layer(config, box_params, s)
        b, h, w, _ = box_pred.shape
        box_pred = box_pred.reshape((b, h, w, 4, dfl_channels))
        box_pred = nn.softmax(box_pred, axis=-1)

        scales.append((class_pred, box_pred))

    return scales

def init_detection_head(config: ModelConfig):
    width: list[int] = config['width']
    filters: list[int] = (width[3], width[4], width[5])
    
    num_classes = config['num_classes']
    dfl_channels = config['dfl_channels']
    
    head_chs_mid = max(filters[0], num_classes)
    box_chs_mid = max(filters[0], dfl_channels * 4)
    
    return {
        'class_head': [
            {
                'conv1': init_conv(x, head_chs_mid, kernel_size=3),
                'conv2': init_conv(head_chs_mid, head_chs_mid, kernel_size=3),
                'conv3_weights': init_fn((1, 1, head_chs_mid, num_classes)), # HWIO
                'conv3_biases': jnp.zeros((num_classes,)),
            } for x in filters
        ],
        'box_head': [
            {
                'conv1': init_conv(x, box_chs_mid, kernel_size=3),
                'conv2': init_conv(box_chs_mid, box_chs_mid, kernel_size=3),
                'conv3_weights': init_fn((1, 1, box_chs_mid, dfl_channels * 4)), # HWIO
                'conv3_biases': jnp.zeros((dfl_channels * 4,))
            } for x in filters
        ]
    }

def yolo(config: ModelConfig, params: dict, x: jnp.ndarray, train: bool=False) -> jnp.ndarray:
    x = dark_net(config, params['backbone'], x)
    x = dark_fpn(config, params['feature_pyramid_network'], x)
    scales = detection_head(config, params['detection_head'], list(x))

    if train:
        return scales
    else:
        bboxes = []
        scores = []

        for (class_target, bbox_target) in scales:
            def expectations(target: jnp.ndarray, horizontal_mode: bool) -> jnp.ndarray:
                scale_min, scale_max = oconfig['bbox_width_range'] if horizontal_mode else oconfig['bbox_height_range']
                expectations = target.mean(-1)
                bbox_scales = scale_min + expectations * (scale_max - scale_min)
                return bbox_scales

            side_exps = [expectations(bbox_target[:, :, :, i], hori_mode) for i, hori_mode in [(0, True), (1, True), (2, False), (3, False)]]
            indices = jnp.nonzero(class_target >= oconfig['score_threshold'])
            # TODO: Atm. nothing makes use of that. Do i really have to pass the counts through everywhere?
            bboxes.append(jnp.concat([side_exp[:, None][indices][:, None] for side_exp in side_exps], axis=1))
            scores.append(class_target[indices])

        bboxes = jnp.concat(bboxes, axis=0)
        scores = jnp.concat(scores, axis=-1)
        bboxes, scores, non_maximum_supression(bboxes, scores, threshold=oconfig['nms_threshold'])
        return scales, bboxes, scores

def init_yolo(config: ModelConfig):
    return {
        'backbone': init_dark_net(config),
        'feature_pyramid_network': init_dark_fpn(config),
        'detection_head': init_detection_head(config)
    }

# TODO For inference: fuse the batch norm into the predeceeding convolution.

if __name__ == "__main__":
    dconfig = DataConfig()
    tconfig = TrainConfig()
    mconfig = ModelConfig()

    if True: 
        oconfig = mconfig.object_detector

        dataset = get_obj_detection_dataset(dconfig)
        train_dataset, valid_dataset, test_dataset = dataset.split_train_validation_test(
            train_percentage=0.8, 
            valid_percentage=0.02
        )
        
        params = init_yolo(mconfig.object_detector)

        width_min, width_max = oconfig['bbox_width_range']
        height_min, height_max = oconfig['bbox_height_range']
        dfl_channels = oconfig['dfl_channels']

        def rescale_and_clip(sizes: jnp.ndarray, size_min: float, size_max: float):
            sizes = jnp.clip((sizes - size_min) / (size_max - size_min), 0.0, 1.0)
            sizes = (dfl_channels - 1) * sizes

            floor = jnp.floor(sizes)
            fract = 1.0 - (sizes - floor)

            return jnp.astype(floor, jnp.uint32), fract

        def get_loss_from_scales(batch: jnp.ndarray, scales: jnp.ndarray) -> jnp.ndarray:
            images, bboxes, masks, classes = batch['images'], batch['bboxes'], batch['masks'], batch['classes']
            num_classes = oconfig['num_classes']

            total_class_loss = jnp.zeros(())
            total_bbox_loss = jnp.zeros(())

            for (class_pred, bbox_pred) in scales:
                b, h, w, _ = class_pred.shape
                left, right, top, bottom = bboxes[..., 0], bboxes[..., 2], bboxes[..., 1], bboxes[..., 3]
                left, right, top, bottom = left * w, right * w, top * h, bottom * h
                cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
                cx, cy = jnp.round(cx), jnp.round(cy)
                cxi, cyi = cx.astype(jnp.uint32), cy.astype(jnp.uint32)

                # Class target
                mask = jnp.zeros((b, h, w, 1), dtype=jnp.bool_)
                mask.at[jnp.arange(b)[:, None], cyi, cxi, 0].set(True)
                class_target = mask
                # Make sure to mask
                # TODO: Add multi class support.

                # BBox target
                bbox_targets = [
                    rescale_and_clip(cx - left, width_min, width_max),
                    rescale_and_clip(right - cx, width_min, width_max),
                    rescale_and_clip(cy - top, height_min, height_max),
                    rescale_and_clip(bottom - cy, height_min, height_max),
                ]

                # total_class_loss += quality_focal_loss(class_target, class_pred, tconfig.focal_loss['exponent'])
                total_bbox_loss += distribution_focal_loss(cxi, cyi, bbox_targets, bbox_pred)
                # total_bbox_loss += general_iou_loss(bbox_targets, bbox_pred, mask)

            # TODO: Weight parameter.
            total_class_loss /= len(scales)
            total_bbox_loss /= len(scales)
            return tconfig.bbox_loss_weight * total_bbox_loss + (1.0 - tconfig.bbox_loss_weight) * total_class_loss

        def loss_fn(params, batch):
            scales = yolo(mconfig.object_detector, params, batch['img'], train=True)
            return get_loss_from_scales(batch, scales)

        def validation_fn(params, batch):
            scales, bboxes, scores = yolo(mconfig.object_detector, params, batch['img'], train=False)

            return {
                'loss': get_loss_from_scales(batch, scales),
                'AP': average_precision(batch['bboxes'], bboxes, scores)
            }

    train_model("obj_detector", tconfig, dconfig, mconfig, train_dataset, valid_dataset, params, loss_fn, validation_fn)
