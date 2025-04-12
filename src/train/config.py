
class TrainConfig:
    def __init__(self):
        self.lr: float = 1e-5
        self.num_epochs: int = 10 # 25
        self.batch_size: int = 128
        self.model_path: str = "../data/classifier.jax"
        self.train_dl_num_threads = 12
        self.train_dl_prefetch_size = 12
        self.valid_dl_num_threads = 12
        self.valid_dl_prefetch_size = 12

        self.lr_schedule = {
            'max_lr': 1e-3,
            'const_lr': 1e-4,
            'min_lr': 1e-5,
            'num_warmup_epochs': 1,
            'num_cooldown_epochs': 12,
            'num_decay_epochs': 6,
            'num_annealing_epochs': 5
        }


class DataConfig:
    videos_dir = "../data/sock_videos"
    annotations_dir = "../data/sock_video_master_tracks"

    classifier_output_dir = "../data/temp/classifier_ds"
    classifier_target_image_size = (1080, 1920)
    classifier_image_size = (518, 518)
    classifier_negative_percentage = 0.3
    classifier_imagenet_percentage = 0.4
    classifier_avoid_class = 806 # Imagenet SOCK class
    classifier_max_num_class_samples = 1024 * 4

    classifier_neg_anchors = {
        'min_size': 0.15,
        'max_size': 0.3,
        'steps': 5
    }
    classifier_neg_class_iou_thresh = 0.05


    split_train_percentage = 0.8
    split_valid_percentage = 0.08

    output_dir = "../data/temp/obj_detector_ds"
    max_objs_per_image = 256

def yolo_v8_config(num_classes: int, depth, width):
    return {
        'depth': depth,
        'width': width,
        'num_classes': num_classes,
        'dfl_channels': 16,
        'bbox_width_range': (0.05, 0.2),
        'bbox_height_range': (0.05, 0.2),
        'score_threshold': 0.5,
        'nms_threshold': 0.5,
        'focal_loss': { 'exponent': 2.0, 'pos_class_weight': 0.25 },
        'bbox_loss_weight': 0.5
    }

def yolo_v8_n(num_classes: int):
    return yolo_v8_config(num_classes, [1, 2, 2], [3, 16, 32, 64, 128, 256])

def yolo_v8_s(num_classes: int):
    return yolo_v8_config(num_classes, [1, 2, 2], [3, 32, 64, 128, 256, 512])

def yolo_v8_m(num_classes: int):
    return yolo_v8_config(num_classes, [2, 4, 4], [3, 48, 96, 192, 384, 576])

def yolo_v8_l(num_classes: int):
    return yolo_v8_config(num_classes, [3, 6, 6], [3, 64, 128, 256, 512, 512])

def yolo_v8_x(num_classes: int):
    return yolo_v8_config(num_classes, [3, 6, 6], [3, 80, 160, 320, 640, 640])

class ModelConfig:
    classifier = {
        'focal_loss': { 'exponent': 2.0, 'pos_class_weight': 0.25 }
    }
    object_detector = yolo_v8_x(1)
