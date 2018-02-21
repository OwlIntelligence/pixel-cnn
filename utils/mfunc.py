import numpy as np
from mask import *

def mask_inputs(inputs, mgen):
    batch_size, height, width, num_channel = inputs.shape
    masks = mgen.gen(batch_size)
    inputs = np.concatenate([inputs, [masks]], axis=-1)
    for c in range(num_channel):
        inputs[:, :, :, c] *= masks
    return inputs
    
