import numpy as np
from PIL import Image

from src.adaptive_compressor import AdaptiveImageCompressor

img = np.zeros((1200, 1200, 3), dtype=np.uint8)
img[0:20, :, :] = [220, 180, 90]
img[:, 0:10, :] = [255, 255, 255]
img[100:120, 200:1000, :] = [90, 140, 200]

compressor = AdaptiveImageCompressor()
result = compressor.compress(Image.fromarray(img), 200)
print("resize_factor=%.4f" % result["resize_factor"])
print("quality=%d" % result["quality"])
print("actual_size_kb=%.2f" % result["actual_size_kb"])
print("ssim=%.4f" % result["ssim"])
print("psnr=%.2f" % result["psnr"])
print("mse=%.4f" % result["mse"])
