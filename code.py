import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.segmentation import slic, mark_boundaries

# 1. Image Normalization
pil_image = Image.open("table.png").convert("RGB")
pil_image.thumbnail((480, 480))
image = np.array(pil_image) / 255.0

# Apply SLIC
segments = slic(image, n_segments=200, compactness=10,
                sigma=1, start_label=0, convert2lab=True)

h, w = segments.shape
seed = segments[h // 2, w // 2]
n = segments.max() + 1

super_pixels = np.zeros((n, 3))

for i in range(n):
    super_pixels[i] = image[segments == i].mean(axis=0)

print("# of superpixel:", n)
print("The super-piexl of the image center:", np.round(super_pixels[seed], 3))

# Here, RGB features are used,
# But you can also use other features like LAB.
features = super_pixels 

mask = (segments == seed)
prev_mask = mask.copy()

from scipy.ndimage import binary_dilation

border = binary_dilation(mask) & ~mask
neighbors = np.unique(segments[border])

# Visualization (on page 5)
neighbor_mask = np.isin(segments, neighbors)

def show_mask(ax, region, color, title):
    ax.imshow(image)

    overlay = np.zeros((*region.shape, 4))
    overlay[region] = [*color, 0.6]
    ax.imshow(overlay)

    ax.set_title(title)
    ax.axis("off")

threshold = 0.1 # Let's find the optimal threshold for your image.
for i in neighbors:
    difference = np.linalg.norm(features[i] - features[seed])
    if difference < threshold:
        mask[segments == i] = True
# Display all four views in one window.
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

axes[0, 0].imshow(mark_boundaries(image, segments))
axes[0, 0].scatter(w // 2, h // 2, c="cyan", s=30)
axes[0, 0].set_title("SLIC Superpixels + Seed")
axes[0, 0].axis("off")

show_mask(axes[0, 1], prev_mask, (1, 0, 0), "Center Superpixel (Seed)")
show_mask(axes[1, 0], neighbor_mask, (0, 1, 0), "Neighbor Superpixels")
show_mask(axes[1, 1], mask, (0, 0, 1),
          f"Grouped Superpixels (RGB, threshold={threshold})")

plt.tight_layout()
plt.show()
