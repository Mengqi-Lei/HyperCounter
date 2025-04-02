import torch
import torch.utils.data as data
import os
from glob import glob
import random
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from torchvision import transforms

def random_crop(im_h, im_w, crop_h, crop_w):

    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


def parse_xml_points(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()
    points = []
    for obj in root.findall('object'):
        pt = obj.find('point')
        if pt is not None:
            x = pt.find('x').text
            y = pt.find('y').text
            points.append([float(x), float(y)])
    return np.array(points, dtype=np.float32)


class Crowd_Drone(data.Dataset):
    def __init__(self, root_path, crop_size=256,
                 downsample_ratio=8,
                 method='train'):

        self.root_path = root_path

        self.gt_list = sorted(glob(os.path.join(self.root_path, 'GT_', '*.xml')))
        if method not in ['train', 'val', 'test']:
            raise Exception(f"method {method} not implement")
        self.method = method

        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio

        self.RGB_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.407, 0.389, 0.396],
                std=[0.241, 0.246, 0.242]),
        ])

        self.T_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.492, 0.168, 0.430],
                std=[0.317, 0.174, 0.191]),
        ])

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
        xml_path = self.gt_list[idx]
        base_name = os.path.basename(xml_path)  # eg. "1R.xml"
        name_only = os.path.splitext(base_name)[0]  # eg. "1R"

        infrared_path = os.path.join(self.root_path, 'Infrared', name_only + '.jpg')
        if name_only.endswith('R'):
            rgb_name = name_only[:-1] + '.jpg'
        else:
            rgb_name = name_only + '.jpg'
        rgb_path = os.path.join(self.root_path, 'RGB', rgb_name)

        RGB = cv2.imread(rgb_path)
        if RGB is None:
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")
        RGB = RGB[..., ::-1].copy() 

        T = cv2.imread(infrared_path)
        if T is None:
            raise FileNotFoundError(f"Infrared image not found: {infrared_path}")
        T = T[..., ::-1].copy()

        keypoints = parse_xml_points(xml_path)

        if self.method == 'train':
            return self.train_transform(RGB, T, keypoints)
        elif self.method == 'val' or self.method == 'test':
        
            k = np.zeros((T.shape[0], T.shape[1]), dtype=np.float32)
            for (x, y) in keypoints:
                ix, iy = int(x), int(y)
                if 0 <= iy < k.shape[0] and 0 <= ix < k.shape[1]:
                    k[iy, ix] = 1
            target = k

            RGB = self.RGB_transform(RGB)
            T = self.T_transform(T)

            name = os.path.splitext(base_name)[0]

            input_data = [RGB, T]
            return input_data, target, name
        else:
            raise Exception(f"Not implement for method: {self.method}")

    def train_transform(self, RGB, T, keypoints):

        ht, wd, _ = RGB.shape
        st_size = float(min(wd, ht))
        assert st_size >= self.c_size, "Image size too small for crop size"
        if len(keypoints) == 0:
            pass

        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        RGB = RGB[i:i+h, j:j+w, :]
        T = T[i:i+h, j:j+w, :]

        keypoints = keypoints - [j, i]
        idx_mask = (keypoints[:, 0] >= 0) & (keypoints[:, 0] <= w) & \
                   (keypoints[:, 1] >= 0) & (keypoints[:, 1] <= h)
        keypoints = keypoints[idx_mask]

        RGB = self.RGB_transform(RGB)
        T = self.T_transform(T)
        input_data = [RGB, T]

        return input_data, torch.from_numpy(keypoints.copy()).float(), st_size
