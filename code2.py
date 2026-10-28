

from ultralytics import SAM
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Image
path = "table.png"
image = np.array(Image.open(path).convert("RGB"))
h, w = image.shape[:2]

# SAM 2.1
model = SAM("sam2.1_t.pt")

# Segmentation based on center point
results = model(path, points=[w//2, h//2], labels=[1])

# Predicted mask
mask = results[0].masks.data[0].cpu().numpy().astype(bool)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(image)
axes[0].scatter(w//2, h//2, c="red", s=40)
axes[0].set_title("Input + Point Prompt")

overlay = np.zeros((*mask.shape, 4))
overlay[mask] = [0, 0, 1, 0.5]  # Blue, alpha=0.5

axes[1].imshow(image)
axes[1].imshow(overlay)
axes[1].set_title("SAM Segmentation")

for ax in axes:
    ax.axis("off")

plt.show()