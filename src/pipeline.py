import os, logging
# os.environ['LOG'] = '1'
# os.environ['INVALIDATE_DATASET'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from tqdm.contrib.logging import logging_redirect_tqdm
from autolabel.autolabel import label_automatically
# from autolabel.visualize import visualize_all
from autolabel.config import BaseConfig
from train.classifier import train_classifier, linear_classifier
from train.train import load_params
from train.config import TrainConfig, DataConfig
import cv2, jax.numpy as jnp, numpy as np, torch
from PIL import Image
import timm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('pipeline.log'), logging.StreamHandler()]
)

class PretrainedClassifier:
    def __init__(self):
        pass
        # self.model = timm.create_model('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=True)
        # self.model = self.model.to('cuda')
        # self.model.eval()
        
        # data_config = timm.data.resolve_model_data_config(self.model)
        # self.transforms = timm.data.create_transform(**data_config, is_training=False)

    def __call__(self, x: np.ndarray, class_idx: int):
        return 1.0
        # x = cv2.resize(x, (448, 448)) * 255
        # x = Image.fromarray(x)
        # input_tensor = self.transforms(x).unsqueeze(0).float().to('cuda')
        # output = self.model(input_tensor)
        # probabilities = torch.nn.functional.softmax(output[0], dim=0)
        # return probabilities[class_idx]


# class CustomClassifier:
#     def __init__(self, tconfig: TrainConfig, dconfig: DataConfig):
#         self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg').to('cuda')
#         self.dinov2 = torch.compile(self.dinov2)
#         self.jax_params = load_params(tconfig.model_path)
#         self.dconfig = dconfig

#     def __call__(self, x: np.ndarray, _: int):
#         x = cv2.resize(x, self.dconfig.classifier_image_size)
#         x = np.array(x)[None,]
#         with torch.no_grad(), torch.autocast(device_type='cuda'):
#             images = torch.from_numpy(x).to(dtype=torch.float16, device='cuda') / 255.0
#             images = torch.permute(images, (0, 3, 1, 2))
#             features = self.dinov2(images)
#             jax_features = jnp.from_dlpack(torch.utils.dlpack.to_dlpack(features), copy=True)
#             pred = linear_classifier(self.jax_params, jax_features)
#             print(f"{pred[0][0]=}")
#             return pred[0][0]


if __name__ == "__main__":
    with logging_redirect_tqdm():
        config = BaseConfig(
            imagenet_class = 806,
            input_dir = "../data/data_input",
            output_dir = "../data/data_output",
            image_size = (1080, 1920),
            output_warped_size = (540, 960),   
            classifier_confidence_threshold = 0.01
        )

        print("Autolabel using pretrained classifier:")
        classifier = PretrainedClassifier()
        label_automatically(classifier, config)
        # visualize_all(config)

        # print("Training classifier:")
        # train_classifier()

        # config = BaseConfig(
        #     imagenet_class = 806,
        #     input_dir = "../data/sock_videos",
        #     output_dir = "../data/sock_video_enhanced_master_tracks",
        #     image_size = (1080, 1920),
        #     output_warped_size = (540, 960),   
        #     classifier_confidence_threshold = 0.5
        # )

        # print("Autolabel using custom classifier:")
        # tconfig, dconfig = TrainConfig(), DataConfig()
        # classifier = CustomClassifier(tconfig, dconfig)
        # label_automatically(classifier, config)
        # visualize_all(config)
