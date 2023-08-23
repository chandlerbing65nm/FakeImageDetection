import cv2
import numpy as np
from PIL import Image
from random import random, choice
from scipy.ndimage.filters import gaussian_filter
import torchvision.transforms.functional as F
from io import BytesIO
import torch
import torchvision.transforms as transforms

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class ImageAugmentor:
    def __init__(self, opt):
        self.opt = opt
        self.jpeg_dict = {'cv2': self.cv2_jpg, 'pil': self.pil_jpg}
        self.rz_dict = {'bilinear': Image.BILINEAR,
                        'bicubic': Image.BICUBIC,
                        'lanczos': Image.LANCZOS,
                        'nearest': Image.NEAREST}

    def cv2_jpg(self, img, compress_val):
        img_cv2 = img[:,:,::-1]
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
        result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        return decimg[:,:,::-1]

    def pil_jpg(self, img, compress_val):
        out = BytesIO()
        img = Image.fromarray(img)
        img.save(out, format='jpeg', quality=compress_val)
        img = Image.open(out)
        # load from memory before ByteIO closes
        img = np.array(img)
        out.close()
        return img

    def jpeg_from_key(self, img, compress_val, key):
        method = self.jpeg_dict[key]
        return method(img, compress_val)

    def gaussian_blur(self, img, sigma):
        gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
        gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
        gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

    def sample_discrete(self, s):
        if len(s) == 1:
            return s[0]
        return choice(s)

    def sample_continuous(self, s):
        if len(s) == 1:
            return s[0]
        if len(s) == 2:
            rg = s[1] - s[0]
            return random() * rg + s[0]
        raise ValueError("Length of iterable s should be 1 or 2.")

    def data_augment(self, img):
        img = np.array(img)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)

        if random() < self.opt['blur_prob']:
            sig = self.sample_continuous(self.opt['blur_sig'])
            self.gaussian_blur(img, sig)

        if random() < self.opt['jpg_prob']:
            method = self.sample_discrete(self.opt['jpg_method'])
            qual = self.sample_discrete(self.opt['jpg_qual'])
            img = self.jpeg_from_key(img, qual, method)

        return Image.fromarray(img)

    def custom_resize(self, img):
        interp = self.sample_discrete(self.opt['rz_interp'])
        return F.resize(img, self.opt['loadSize'], interpolation=self.rz_dict[interp])