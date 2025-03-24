class TrainConfig:
    pass 
    
class DataConfig:
    pass 


def yolo_v8_config(num_classes: int, depth, width):
    return {
        'depth': depth,
        'width': width,
        'num_classes': num_classes,
        'dfl_channels': 16,
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
