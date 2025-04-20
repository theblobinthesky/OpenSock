from .calibration import apply_calibration
from .colormap import apply_luts
from .config import BaseConfig
import subprocess, json, logging
import numpy as np, cv2


class VideoContext:
    def __init__(self, video_path: str, config: BaseConfig):
        self.video_path = video_path
        self.config = config
        self.calib_config = None
        self.luts = None
        self.homographies = None


def apply_homography(image: np.ndarray, homography: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = cv2.warpPerspective(image, homography, image.shape[:2])
    image = image.transpose((1, 0, 2))
    image = cv2.resize(image, size[::-1], interpolation=cv2.INTER_AREA)
    return image


class Iterator:
    def __init__(self, gen, size):
        self.gen = gen
        self.size = size
    def __iter__(self):
        return self
    def __next__(self):
        return next(self.gen)
    def __len__(self):
        return self.size

def ffprobe_info(path):
    """Run ffprobe to get video stream width, height, nb_frames."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames",
        "-of", "json", path
    ]
    p = subprocess.run(cmd, capture_output=True, check=True)
    info = json.loads(p.stdout)
    s = info["streams"][0]
    # nb_frames may be a string; cast if available
    return int(s["width"]), int(s["height"]), int(s.get("nb_frames", 0))

def _open_video_ffmpeg_generator(path, frames, video_ctx, load_in_8bit_mode):
    # choose pixel format
    pix_fmt = "rgb24" if load_in_8bit_mode else "rgb48le"
    w, h, _ = ffprobe_info(path)

    cmd = [
        "ffmpeg", "-i", path,
        "-f", "rawvideo",
        "-pix_fmt", pix_fmt,
        "-vsync", "0",  # no frame dropping/duplication
        "pipe:1"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    wanted = set(frames) if frames is not None else None
    bytes_per_chan = 1 if load_in_8bit_mode else 2
    channels = 3
    frame_bytes = w * h * channels * bytes_per_chan

    try:
        i = 0
        while True:
            raw = proc.stdout.read(frame_bytes)

            if len(raw) < frame_bytes:
                break

            if (wanted is None) or (i in wanted):
                dtype = np.uint8 if load_in_8bit_mode else np.uint16
                frame = np.frombuffer(raw, dtype=dtype).copy()
                frame = frame.reshape((h, w, channels))

                if not load_in_8bit_mode:
                    frame //= 2 ** 6

                    if video_ctx.calib_config is not None:
                        frame = apply_calibration(frame, video_ctx.calib_config)

                    if video_ctx.luts is not None:
                        frame = apply_luts(frame, video_ctx.luts)
                        frame = frame.astype(np.uint8)

                        if video_ctx.homographies and i in video_ctx.homographies:
                            apply_homography(frame, video_ctx.homographies[i], video_ctx.config.image_size)

                    if wanted:
                        wanted.remove(i)
                yield i, frame
                if wanted is not None and not wanted:
                    break
            i += 1
    except Exception as e:
        logging.error("ffmpeg pipeline error: %s", e)
    finally:
        proc.stdout.close()
        proc.wait()

def open_video_capture(video_ctx, frames: list[int]=None, load_in_8bit_mode: bool=False):
    size = len(frames) if frames is not None else ffprobe_info(video_ctx.video_path)[2]
    gen = _open_video_ffmpeg_generator(video_ctx.video_path, frames, video_ctx, load_in_8bit_mode)
    return Iterator(gen, size)
