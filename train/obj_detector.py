import os, jax, jax.nn as nn, jax.numpy as jnp, jax.lax as lax
import jax.nn.initializers as init
from config import TrainConfig, DataConfig, ModelConfig
from data import SimpleYOLOPreprocessor
from train import train_model, load_params

DIM_NUMS = ('NHWC', 'HWIO', 'NHWC')

class Initializer:
    def __init__(self):
        self.key = jax.random.PRNGKey(0)
        self.kernel_init = init.glorot_uniform()
    
    def __call__(self, shape: tuple[int, ...]):
        k1, k2 = jax.random.split(self.key)
        self.key = k1
        return self.kernel_init(k2, shape)

def batch_norm():
    pass
    
def conv(config: ModelConfig, params: dict, x: jnp.ndarray, stride: int=1) -> jnp.ndarray:
    x = lax.conv_general_dilated(x, params['conv_weights'], (stride, stride), 'SAME', dimension_numbers=DIM_NUMS)
    x = batch_norm()
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
    y.extend(residual(config, params[f"residual{i}"], y[-1]) for i in range(n))
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

def spp(config: ModelConfig, params: dict, x: jnp.array) -> jnp.array:
    x = conv(config, params['conv1'], x)
    pool1 = max_pool_2d(x, ...todoparams)
    pool2 = max_pool_2d(pool1, ...todoparams)
    pool3 = max_pool_2d(pool2, ...todoparams)
    x = jnp.concat([x, pool1, pool2, pool3], axis=-1)
    return conv(config, params['conv2'], x)

def init_spp(chs_in: int, chs_out: int) -> dict:
    return {
        'conv1': init_conv(chs_in, chs_out),
        'conv2': init_conv(chs_in, chs_out)
    }

def dark_net_layer(config: ModelConfig, params: dict, x: jnp.ndarray) -> jnp.ndarray:
    x = conv(config, params['conv'], x, stride=2)
    x = csp(config, params['csp'], x)
    return x

def dark_net(config: ModelConfig, params: dict, x: jnp.ndarray) -> jnp.ndarray:
    p1 = conv(config, params['p1_conv'], x, stride=2)
    p2 = dark_net_layer(config, params['p2'], p1) 
    p3 = dark_net_layer(config, params['p3'], p2) 
    p4 = dark_net_layer(config, params['p4'], p3) 
    p5 = dark_net_layer(config, params['p5'], p4) 
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
    p3, p4, p5 = x
  
    # Pyramid: Up
    h1 = csp(config, params['h1'], jnp.concat([p4, upsample(p5)], axis=-1)) 
    h2 = csp(config, params['h2'], jnp.concat([p3, upsample(h1)], axis=-1))
  
    # Pyramid: Down
    h4 = csp(config, params['h4'], jnp.concat([p4, conv(config, params['h3'], h2, stride=2)], axis=-1))
    h6 = csp(config, params['h6'], jnp.concat([p5, conv(config, params['h5'], h4, stride=2)], axis=-1))
    
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
    for class_params, box_params, s in zip(params['class_head'], params['box_head'], x):
        scales.append(detection_head_layer(config, class_params, s))
        scales.append(detection_head_layer(config, box_params, s))

    return jnp.concat(scales, axis=-1)

def init_detection_head(config: ModelConfig, filters: list):
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

def yolo(config: ModelConfig, params: dict, x: jnp.ndarray) -> jnp.ndarray:
    x = dark_net(config, params['backbone'], x)
    x = dark_fpn(config, params['feature_pyramid_network'], x)
    x = detection_head(config, params['detection_head'], list(x))

    return x

def init_yolo(config: ModelConfig):
    return {
        'backbone': init_dark_net(config),
        'feature_pyramid_network': init_dark_fpn(config),
        'detection_head': init_detection_head(config)
    }
    
def init_params(mconfig: ModelConfig, init_fn: Initializer) -> dict[str, jnp.ndarray]:
    
    return params   


# TODO For inference: fuse the batch norm into the predeceeding convolution.
# TODO For inference: The detection head.

if __name__ == "__main__":
    dconfig = DataConfig()
    tconfig = TrainConfig()
    mconfig = ModelConfig()
    
    init_fn = Initializer()
    params = init_yolo(mconfig, init_fn)
    yolo(mconfig, params, jnp.zeros((2, 800, 800, 3)))
    
    ds_preproc = SimpleYOLOPreprocessor(
        videos_dir="path/to/mov_files",
        annotations_dir="path/to/json_files",
        output_dir="path/to/output_dataset"
    )
    
    dataset = ds_preproc.create_dataset(batch_size=16)

    
    exit(0)
    

    if 'VIZ' in os.environ:
        test_dataset = get_test_dataset(dconfig, tconfig)

        import matplotlib.pyplot as plt
        batch = test_dataset.get_next_batch()

        def pad(x):
            h, w, _ = x.shape
            return jnp.concat([x, jnp.zeros((h, w, 1))], axis=-1)

        params = load_params(tconfig.model_path)
        img, uv = batch["img"], batch["uv"]
        x = vit_basic(mconfig, params, img)

        for b in range(tconfig.batch_size):
            _, ax = plt.subplots(1, 3)
            ax[0].imshow(img[b])
            ax[1].imshow(pad(uv[b, :, :, :2]))
            ax[2].imshow(pad(x[b]))
            plt.show()

    else:
        train_dataset = get_train_dataset(dconfig, tconfig)
        valid_dataset = get_valid_dataset(dconfig, tconfig)

        init_fn = Initializer()
        params = init_params(mconfig, init_fn)
        import train
        train._save_model(params, "model.pkl")

        def loss_fn(params, batch):
            x = vit_basic(mconfig, params, batch["img"])
            return jnp.mean(jnp.abs(x - batch["uv"][..., :2]))

        train_model(tconfig, dconfig, train_dataset, valid_dataset, params, loss_fn)