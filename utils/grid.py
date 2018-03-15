import numpy as np


def generate_grid(img_size, value_interval=[-1,1], batch_size=None):
    if type(value_interval[0])==list:
        value_interval_h, value_interval_w = value_interval
    else:
        value_interval_h, value_interval_w = value_interval, value_interval
    h = np.linspace(start=value_interval_h[0], stop=value_interval_h[1], num=img_size[0], dtype=np.float32)
    w = np.linspace(start=value_interval_w[0], stop=value_interval_w[1], num=img_size[1], dtype=np.float32)
    grid = np.zeros((img_size[0], img_size[1], 2), dtype=np.float32)
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            grid[i, j, :] = [h[i], w[j]]
    if batch_size is None:
        return grid
    else:
        return np.broadcast_to(grid, shape=(batch_size,)+grid.shape)

def zoom(grid, resolution):
    h_interval = [grid[:, :, 0].min(), grid[:, :, 0].max()]
    w_interval = [grid[:, :, 1].min(), grid[:, :, 1].max()]
    g = generate_grid(img_size=resolution, value_interval=[h_interval, w_interval])
    return g

def zoom_batch(grids, resolution):
    zg = []
    for g in grids:
        zg.append(zoom(g, resolution))
    return np.array(zg)
