import autolabel
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

    autolabel.label_automatically(
        imagenet_class=806, # SOCK class
        classifier=model,
        classifier_transform=transform,
        input_dir="sock_videos",
        output_dir="sock_video_results"
    )
