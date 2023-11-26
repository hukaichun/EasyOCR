import os
import re
import math
import torch
import pandas  as pd

from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as transforms

import logging

LOGGER = logging.getLogger(__name__)

def contrast_grey(img):
    high = np.percentile(img, 90)
    low  = np.percentile(img, 10)
    return (high-low)/(high+low), high, low

def adjust_contrast_grey(img, target = 0.4):
    contrast, high, low = contrast_grey(img)
    if contrast < target:
        img = img.astype(int)
        ratio = 200./(high-low)
        img = (img - low + 25)*ratio
        img = np.maximum(np.full(img.shape, 0) ,np.minimum(np.full(img.shape, 255), img)).astype(np.uint8)
    return img

def load_dataset(*data_set_roots:str, 
                  character:str, 
                  # default parameters
                  label_max_length=34,
                  imgH:int=64,
                  imgW:int=600,
                  keep_ratio_with_pad:bool=False,
                  contrast_adjust:float=0,
                  batch_size:int=32,
                  shuffle:bool=True,
                  num_workers:int=6,
                  prefetch_factor:int=512) -> torch.utils.data.DataLoader:
    ocrs = [OCRDataset(root=root, character=character, label_max_length=label_max_length) for root in data_set_roots]
    ocr = ConcatDataset(ocrs)
    aligncollate = AlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=keep_ratio_with_pad, contrast_adjust=contrast_adjust)
    return torch.utils.data.DataLoader(ocr, 
                                       batch_size=batch_size, 
                                       collate_fn=aligncollate, 
                                       shuffle=shuffle, 
                                       num_workers=num_workers,
                                       prefetch_factor=prefetch_factor)
    

class OCRDataset(Dataset):
    def __init__(self, *, root:str, character:str, label_max_length:int, rgb:bool=False, sensitive:bool=True):
        LOGGER.info("dataset: %s", root)
        
        self.root = root
        self.label_max_length = label_max_length
        self.character = character
        self.rgb = rgb
        self.sensitive = sensitive
        try:
            self.df = pd.read_csv(os.path.join(root,'labels.csv'), sep='^([^,]+),', engine='python', usecols=['filename', 'words'], keep_default_na=False)
        except:
            self.df = pd.read_csv(os.path.join(root,'labels.csv'), sep='^([^,]+),', engine='python', usecols=['filename', 'words'], keep_default_na=False, encoding='big5')
        
        self._rename_label()
        self._skip_data_whose_label_is_longer_than(label_max_length)

        self.nSamples = len(self.df)
        # assert self.df.apply(lambda row: len(row["words"])< self.label_max_length, axis=1).all(), f"please check the label length; maximum langth: {self.label_max_length=}"

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        img_fname = self.df.at[index,'filename']
        img_fpath = os.path.join(self.root, img_fname)
        label = self.df.at[index,'words']

        if self.rgb:
            img = Image.open(img_fpath).convert('RGB')  # for color image
        else:
            img = Image.open(img_fpath).convert('L')

        if not self.sensitive:
            label = label.lower()

        return (img, label)

    def _skip_data_whose_label_is_longer_than(self, length):
        check_length = self.df.apply(lambda row: len(row["words"]) > length, axis=1)
        if check_length.any():
            LOGGER.warning("Ignore data whose label is longer than %d: \n%s", length, self.df.loc[check_length])
            self.df = self.df.loc[~check_length]
            self.df.reset_index(inplace=True)

    def _rename_label(self):
        # We only train and evaluate on alphanumerics #(or pre-defined character set in train.py)
        out_of_char = f'[^{self.character}]'
        self.df["words"] = self.df.apply(lambda row: re.sub(out_of_char, '', row["words"]), axis=1)
        check_illegal = self.df["words"] == ""
        if check_illegal.any():
            LOGGER.warning("Illegal words %s", self.df.loc[check_illegal])
            self.df = self.df.loc[~check_illegal]
            self.df.reset_index(inplace=True)
    

class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, contrast_adjust = 0.):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.contrast_adjust = contrast_adjust

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size

                #### augmentation here - change contrast
                if self.contrast_adjust > 0:
                    image = np.array(image.convert("L"))
                    image = adjust_contrast_grey(image, target = self.contrast_adjust)
                    image = Image.fromarray(image, 'L')

                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
