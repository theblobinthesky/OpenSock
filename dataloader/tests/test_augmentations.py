import pytest
from enum import Enum
import subprocess

VEC_NONE = "NONE"
VEC_SSE41 = "SSE41"
VEC_AVX2 = "AVX2"

class ChannelType(Enum):
    CHANNEL_FIRST = 0
    CHANNEL_LAST = 1

DL_CONFIGS = [
    # (VectorisationType, ChannelFirst)
    (VEC_NONE, ChannelType.CHANNEL_FIRST),
    (VEC_NONE, ChannelType.CHANNEL_LAST),
    (VEC_SSE41, ChannelType.CHANNEL_FIRST),
    (VEC_SSE41, ChannelType.CHANNEL_LAST),
    (VEC_AVX2, ChannelType.CHANNEL_FIRST),
    (VEC_AVX2, ChannelType.CHANNEL_LAST),
]

def compile_for(vec_type: str):
    subprocess.run(f"""uv pip install -e . -Cbuild-dir=build/release -Ccmake.args="--preset Release -DVECTOR_EXTENSIONS={vec_type}""", shell=True, check=True)
    subprocess.run("cmake --preset Release", shell=True, check=True)

@pytest.fixture(params=DL_CONFIGS, ids=lambda p: f"{p[0]}, {p[1]}")
def cfg(request):
    vec_type, channel_type = request.param
    compile_for(vec_type)
    return {"vectorisation_type": vec_type, "channel_type": channel_type}


def test_flip_augmentation(cfg):
    pass

def test_pad_augmentation(cfg):
    pass

def test_random_crop_augmentation(cfg):
    pass

def test_resize_augmentation(cfg):
    pass
