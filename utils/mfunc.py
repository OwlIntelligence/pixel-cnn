import numpy as np
import math
from utils.mask import *

def mask_inputs(inputs, mgen):
    batch_size, height, width, num_channel = inputs.shape
    masks = mgen.gen(batch_size)
    inputs = np.concatenate([inputs, np.expand_dims(masks, axis=-1)], axis=-1)
    for c in range(num_channel):
        inputs[:, :, :, c] *= masks
    return inputs

def find_contour(mask):
    contour = np.zeros_like(mask)
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if mask[y, x] > 0:
                lower_bound = max(y-1, 0)
                upper_bound = min(y+1, h-1)
                left_bound = max(x-1, 0)
                right_bound = min(x+1, w-1)
                nb = mask[lower_bound:upper_bound+1, left_bound:right_bound+1]
                if np.min(nb)  == 0:
                    contour[y, x] = 1
    return contour

def tile_images(imgs, size=(6, 6)):
    imgs = imgs[:size[0]*size[1], :, :, :]
    img_h, img_w = imgs.shape[1], imgs.shape[2]
    all_images = np.zeros((img_h*size[0], img_w*size[1], 3), np.uint8)
    for j in range(size[0]):
        for i in range(size[1]):
            all_images[img_h*j:img_h*(j+1), img_w*i:img_w*(i+1), :] = imgs[j*size[0]+i, :, :, :]
    return all_images


## https://github.com/aizvorski/video-quality/blob/master/psnr.py

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def batch_psnr(imgs1, imgs2, output_mean=True):
    assert imgs1.shape[0]==imgs2.shape[0], "batch size of imgs1 and imgs2 should be the same"
    batch_size = imgs1.shape[0]
    v = [psnr(imgs1[i], imgs2[i]) for i in range(batch_size)]
    if output_mean:
        return np.mean(v)
    return v


def evaluate(original_imgs, completed_imgs):
    print(original_imgs)
    print(completed_imgs)
    return batch_psnr(original_imgs, completed_imgs)
