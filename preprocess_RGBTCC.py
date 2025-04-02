import os
import shutil
import numpy as np
import json
from glob import glob

# This file is used to preprocess the RGBT-CC dataset, converting the original data format to the format used in the project.

def generate_data(label_path):
    with open(label_path, 'r') as f:
        label_file = json.load(f)
    points = np.asarray(label_file['points'])

    return points


if __name__ == '__main__':
    root_path = 'data/RGBT-CC/RGBT-CC-orig'  # The path to the original dataset root directory
    save_dir = 'data/RGBT-CC/RGBT_CC'    # The path to save the generated data

    for phase in ['train', 'val', 'test']:
        sub_dir = os.path.join(root_path, phase)
        sub_save_dir = os.path.join(save_dir, phase)
        os.makedirs(sub_save_dir, exist_ok=True)

        gt_list = glob(os.path.join(sub_dir, '*json'))
        for gt_path in gt_list:
            name = os.path.basename(gt_path)
            im_save_path = os.path.join(sub_save_dir, name)
            rgb_save_path = im_save_path.replace('GT', 'RGB').replace('json', 'jpg')
            t_save_path = im_save_path.replace('GT', 'T').replace('json', 'jpg')
            gd_save_path = im_save_path.replace('json', 'npy')

            # Copy the original RGB and T images
            original_rgb_path = gt_path.replace('GT', 'RGB').replace('json', 'jpg')
            original_t_path = gt_path.replace('GT', 'T').replace('json', 'jpg')
            shutil.copy(original_rgb_path, rgb_save_path)
            shutil.copy(original_t_path, t_save_path)

            # Save the generated npy file
            points = generate_data(gt_path)
            np.save(gd_save_path, points)
