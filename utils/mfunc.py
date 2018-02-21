import numpy as np
from utils.mask import *

def mask_inputs(inputs, mgen):
    batch_size, height, width, num_channel = inputs.shape
    masks = mgen.gen(batch_size)
    print(masks.shape)
    print(np.expand_dims(masks, axis=-1).shape)
    inputs = np.concatenate([inputs, np.expand_dims(masks, axis=0)], axis=-1)
    for c in range(num_channel):
        inputs[:, :, :, c] *= masks
    return inputs
