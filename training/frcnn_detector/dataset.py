import os
import json
import typing as tp

import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data import DataLoader
from PIL import Image, ImageOps





def load_dataset(image_root:str,
                 annotation_file:str,
                 with_target=True,
                 batch_size:int=2,
                 shuffle:bool=True,
                 num_workers:int=6,
                 prefetch_factor:int=64,
                 pin_memory:bool=True):
    detection_dataset = DetectionDataset(image_root, annotation_file) 
    collate_fn = DetectionCollate(with_target=with_target)
    return DataLoader(detection_dataset,
                      batch_size=batch_size,
                      collate_fn=collate_fn,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      prefetch_factor=prefetch_factor,
                      pin_memory=pin_memory)


class DetectionCollate:
    def __init__(self, with_target=True):
        self._with_target = with_target

    def __call__(self, batch):
        if self._with_target:
            images, targets = zip(*batch)
            targets_torch = [{k:torch.from_numpy(v) for k,v in target.items()} for target in targets]
        else:
            images = batch

        images_np = [image.transpose((2,0,1)) for image in images]
        images_torch = torch.stack([torch.from_numpy(image).float() for image in images_np])

        if self._with_target:
            return images_torch, targets_torch
        return images_torch


class DetectionDataset(Dataset):
    def __init__(self, image_dir:str, annotation_file:str="val.json"):
        self.image_dir = image_dir
        annotation_file = os.path.join(image_dir, annotation_file)
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_keys = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx):
        image_key = self.image_keys[idx]
        annotation = self.annotations[image_key]
        image_path = os.path.join(self.image_dir, annotation['filename'])
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.exif_transpose(image)
        image = np.array(image)

        num_regions = len(annotation['regions'])
        # masks = np.zeros((num_regions, image.shape[0], image.shape[1]), dtype=np.uint8)
        boxes = np.zeros((num_regions, 4), dtype=np.float32)
        labels = np.zeros(num_regions, dtype=np.int64)

        for i, region in enumerate(annotation['regions']):
            category = region['region_attributes']['category'][0]
            category_code = ord(category) - ord('A') + 21
            labels[i] = category_code

            if 'all_points_x' in region['shape_attributes']:
                all_points_x = region['shape_attributes']['all_points_x']
                all_points_y = region['shape_attributes']['all_points_y']
            elif 'cx' in region['shape_attributes']:
                cx = region['shape_attributes']['cx']
                cy = region['shape_attributes']['cy']
                rx = region['shape_attributes']['rx']
                ry = region['shape_attributes']['ry']
                theta = region['shape_attributes']['theta']

                # Create approximate ellipse with 12 vertices
                t = np.linspace(0, 2*np.pi, 12, endpoint=False)
                all_points_x = cx + rx * np.cos(t) * np.cos(theta) - ry * np.sin(t) * np.sin(theta)
                all_points_y = cy + rx * np.cos(t) * np.sin(theta) + ry * np.sin(t) * np.cos(theta)
            else:
                # Handle the case where neither 'all_points_x' nor 'cx' is present
                # You can add your own logic or raise an error as needed
                print(image_path)
                raise ValueError("Either 'all_points_x' or 'cx' must be present in shape_attributes.")

            # masks[i, :, :] = self.create_mask(all_points_x, all_points_y, image.shape[0], image.shape[1], category_code)
            boxes[i, :] = [np.min(all_points_x), np.min(all_points_y), np.max(all_points_x), np.max(all_points_y)]

        target = {
            # 'masks': masks,
            'boxes': boxes,
            'labels': labels
        }

        return image, target

    def create_mask(self, all_points_x, all_points_y, height, width, category_code):
        mask = np.zeros((height, width), dtype=np.uint8)
        poly = np.array(list(zip(all_points_x, all_points_y)), dtype=np.int32)
        cv2.fillPoly(mask, [poly], category_code)
        return mask


