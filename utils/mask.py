import numpy as np

class MaskGenerator(object):

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def gen(self, n):
        self.masks = np.ones((n, self.height, self.width))
        return self.masks

class CenterMaskGenerator(MaskGenerator):

    def __init__(self, height, width):
        super().__init__(height, width)

    def gen(self, n, ratio=0.5):
        self.masks = np.ones((n, self.height, self.width))
        c_height = int(self.height * 0.5)
        c_width = int(self.width * 0.5)
        height_offset = (self.height - c_height) // 2
        width_offset = (self.width - c_width) // 2
        self.masks[:, height_offset:height_offset+c_height, width_offset:width_offset+c_width] = 0
        return self.masks

class RandomRectangleMaskGenerator(MaskGenerator):

    def __init__(self, height, width):
        super().__init__(height, width)

    def gen(self, n, min_ratio=0.25, max_ratio=0.75, margin_ratio=0.):
        self.masks = np.ones((n, self.height, self.width))
        for i in range(self.masks.shape[0]):
            min_height = int(self.height * min_ratio)
            min_width = int(self.width * min_ratio)
            max_height = int(self.height * max_ratio)
            max_width = int(self.width * max_ratio)
            margin_height = int(self.height * margin_ratio)
            margin_width = int(self.width * margin_ratio)
            rng = np.random.RandomState(None)
            c_height = rng.randint(low=min_height, high=max_height)
            c_width = rng.randint(low=min_width, high=max_width)
            height_offset = rng.randint(low=margin_height, high=self.height-margin_height-c_height)
            width_offset = rng.randint(low=margin_width, high=self.width-margin_width-c_width)
            self.masks[i, height_offset:height_offset+c_height, width_offset:width_offset+c_width] = 0
        return self.masks
