
class TrainConfig:
    def __init__(self):
        self.lr: float = 1e-3
        self.num_epochs: int = 10
        self.batch_size: int = 1
        self.model_path: str = "model.pkl"
        self.train_dl_num_threads = 16
        self.train_dl_prefetch_size = 16
        self.valid_dl_num_threads = 16
        self.valid_dl_prefetch_size = 16

class DataConfig:
    img_size = (640, 640)
    videos_dir = "../data/sock_videos"
    annotations_dir = "../data/sock_video_results"
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
    object_detector = yolo_v8_x(1)
