import os
os.environ['TQDM_DISABLE'] = '1'
from config import BaseConfig
from torchvision import models, transforms
from torchvision.models import EfficientNet_V2_L_Weights


if __name__ == "__main__":
    model = models.efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
    model.eval()
    model.to('cuda')

    transform = transforms.Compose([
        transforms.Resize(480, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    config = BaseConfig(
        imagenet_class = 806,
        input_dir = "sock_videos",
        output_dir = "sock_video_results",
        image_size = (1080, 1920),
        output_warped_size = (540, 960),   
    )

    if 'VIZ' in os.environ:
        import visualize
        visualize.visualize_all(config)
    else:
        import autolabel
        autolabel.label_automatically(
            classifier=model,
            classifier_transform=transform,
            config=config
        )
