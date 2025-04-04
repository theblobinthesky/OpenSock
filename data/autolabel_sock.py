import os
os.environ['TQDM_DISABLE'] = '1'
from config import BaseConfig
import timm

if __name__ == "__main__":
    model = timm.create_model('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=True)
    model = model.to('cuda')
    model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

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
            classifier_transform=transforms,
            config=config
        )
