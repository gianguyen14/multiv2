from fractions import Fraction

import av
import numpy as np
from PIL import Image


def create_video(path, scene):
    with av.open(str(path), "w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.time_base = Fraction(1, 4)
        for index in range(8):
            pixels = np.zeros((48, 64, 3), dtype=np.uint8)
            if scene == "red_car":
                pixels[:] = (70, 70, 70)
                pixels[24:34, 4 + index * 5:20 + index * 5] = (230, 20, 20)
            elif scene == "blue_object":
                pixels[:] = (20, 35, 80)
                pixels[12:34, 20:44] = (20, 60, 230)
            else:
                pixels[:] = (110, 65, 30)
                pixels[12:34, 24:42] = (240, 240, 235)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 4)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class MeanRGBEncoder:
    embedding_dim = 4

    def __init__(self, revision="v1"):
        self.revision = revision
        self.calls = 0

    def identity(self):
        return {"provider": "test", "model_name": "mean-rgb", "revision": self.revision,
            "embedding_dim": 4, "normalization": "l2", "contract_version": "m15.1-v1"}

    def encode_image(self, images, batch_size=8, normalize=True):
        self.calls += 1
        vectors = []
        for path in images:
            pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
            rgb = pixels.mean(axis=(0, 1))
            vector = np.array([*rgb, 1.0], dtype=np.float32)
            vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.stack(vectors)
