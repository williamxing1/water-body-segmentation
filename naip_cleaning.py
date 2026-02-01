import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window

img_path = "naip/naip_water_train.tif"
mask_path = "naip/naip_water_masks.tif"
test_path = "naip/naip_water_test.tif"
with rasterio.open(img_path) as f:
    image = f.read().astype(np.float32)
with rasterio.open(mask_path) as f:
    mask = f.read().astype(np.float32)
with rasterio.open(test_path) as f:
    test = f.read().astype(np.float32)

print(image.shape)
print(mask.shape)
print(test.shape)

def tile_tif(in_path, out_dir, size=256, stride=128):
    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(in_path) as src:
        H, W = src.height, src.width
        profile = src.profile

        tile_id = 1

        for row in range(0, H - size + 1, stride):
            for col in range(0, W - size + 1, stride):
                window = Window(row, col, size, size)
                tile = src.read(window=window)

                profile.update(height=size, width=size, transform=src.window_transform(window))

                out_path = os.path.join(out_dir, f"tile_{tile_id}.tif")
                tile_id += 1

                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(tile)
"""
tile_tif(img_path, "naip/naip_train_input")
tile_tif(mask_path, "naip/naip_train_mask")
tile_tif(test_path, "naip/naip_test_input")"""

img_path = "naip/naip_train_input/tile_20.tif"
mask_path = "naip/naip_train_mask/tile_20.tif"

with rasterio.open(img_path) as f:
    image = f.read().astype(np.float32)
with rasterio.open(mask_path) as f:
    mask = f.read().astype(np.float32)

print(image.min(), image.max())
rgb = image[:3].transpose(1, 2, 0)
rgb = rgb / 255.0
plt.figure()

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(rgb)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Water Mask")
plt.imshow(mask.squeeze(0), cmap="gray")
plt.axis("off")

plt.show()