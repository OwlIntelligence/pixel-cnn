import numpy as np


def generate_grid(img_size, value_interval=[-1,1]):
    h = np.linspace(start=value_interval[0], stop=value_interval[1], num=img_size[0], dtype=np.float32)
    w = np.linspace(start=value_interval[0], stop=value_interval[1], num=img_size[1], dtype=np.float32)
    grid = np.zeros((img_size[0], img_size[1], 2), dtype=np.float32)
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            grid[i, j, :] = [h[i], w[j]]
    return grid
