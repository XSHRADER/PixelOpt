"""Generate a deterministic test image for the CI smoke test."""

import sys

import numpy as np
from PIL import Image

rng = np.random.default_rng(0)
y, x = np.mgrid[0:600, 0:800]
base = (np.sin(x / 30.0) * np.cos(y / 40.0) + 1) * 110
channel = np.clip(base + rng.normal(0, 20, (600, 800)), 0, 255)
image = np.dstack([channel, np.roll(channel, 25, 0), np.roll(channel, 25, 1)])
Image.fromarray(image.astype(np.uint8)).save(sys.argv[1])
