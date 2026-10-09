import matplotlib.pyplot as plt
import numpy as np

data = np.load("sketchrnn/angel.npz", encoding="latin1", allow_pickle=True)

# print("list:", data.files)

# stroke = data["train"][0]
def draw_strokes(ax, stroke):
    # relative delta (dx, dy) -> absolute coordinates (x, y)
    xy = np.cumsum(stroke[:, :2], axis=0)
    xy[:, 1] = -xy[:, 1] 

    #every 3rd point indicates the end of a stroke (eos)
    eos_indices = np.where(stroke[:, 2] == 1)[0]

    start_idx = 0
    for end_idx in eos_indices:
        #one line
        line = xy[start_idx : end_idx + 1]
        ax.plot(line[:, 0], line[:, 1], "k-", linewidth=1)
        start_idx = end_idx + 1

    if start_idx < len(xy):
        line = xy[start_idx:]
        ax.plot(line[:, 0], line[:, 1], "k-", linewidth=1)

    ax.set_aspect("equal")
    
row, col = 3, 3
fig, axes = plt.subplots(row, col, figsize=(5, 5)) 
axes_flat = axes.flatten()

for i in range (row * col):
    ax = axes_flat[i]
    stroke = data["train"][i]
    draw_strokes(ax, stroke)
    ax.axis("off") # hide axes

plt.tight_layout()
plt.show()