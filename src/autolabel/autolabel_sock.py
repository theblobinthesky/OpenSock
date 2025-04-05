import os
os.environ['TQDM_DISABLE'] = '1'
from .config import BaseConfig


def autolabel_sock(config: BaseConfig, model = None, transforms = None):
    if 'VIZ' in os.environ:
        from . import visualize
        visualize.visualize_all(config)
    else:
        from . import autolabel
        autolabel.label_automatically(
            classifier=model,
            classifier_transform=transforms,
            config=config
        )
