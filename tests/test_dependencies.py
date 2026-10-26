import pickle
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from datasets.crowd_rgbtcc import Crowd_RGBTCC
from models.HypergraphComputation import HypergraphComputation
from models.pvt_v2_encoders import PyramidVisionTransformerV2


class UnsupportedCheckpoint:
    pass


def make_backbone(pretrained=None):
    return PyramidVisionTransformerV2(
        img_size=64, embed_dims=[8, 16, 32, 64], num_heads=[1, 2, 4, 8],
        depths=[1, 1, 1, 1], pretrained=pretrained
    )


class DependencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_checkpoint_formats(self):
        reference = make_backbone().state_dict()
        formats = [reference, {'state_dict': reference},
                   {'state_dict': OrderedDict(('module.' + k, v) for k, v in reference.items())}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'weights.pth'
            for checkpoint in formats:
                with self.subTest(format=list(checkpoint)[:1]):
                    torch.save(checkpoint, path)
                    actual = make_backbone(str(path)).state_dict()
                    self.assertEqual(actual.keys(), reference.keys())
                    for name in reference:
                        self.assertTrue(torch.equal(reference[name], actual[name]), name)

    def test_checkpoint_rejects_custom_objects(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'weights.pth'
            torch.save(UnsupportedCheckpoint(), path)
            with self.assertRaises(pickle.UnpicklingError):
                make_backbone(str(path))

    def test_dataset_and_image_io(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = np.full((256, 256, 3), 127, dtype=np.uint8)
            for modality in ['RGB', 'T']:
                self.assertTrue(cv2.imwrite(str(root / f'example_{modality}.jpg'), image))
            np.save(root / 'example_GT.npy', np.array([[64., 64.], [128., 128.]], dtype=np.float32))
            images, points, size = Crowd_RGBTCC(folder, method='train')[0]
            self.assertEqual(tuple(images[0].shape), (3, 256, 256))
            self.assertEqual(tuple(points.shape), (2, 2))
            self.assertEqual(size, 256)
            self.assertTrue(torch.isfinite(images[0]).all())
            _, target, _ = Crowd_RGBTCC(folder, method='test')[0]
            self.assertEqual(target.sum(), 2)
            path = root / 'prediction.png'
            Image.fromarray(np.ones((32, 32), dtype=np.float32)).convert('L').save(path)
            with Image.open(path) as prediction:
                self.assertEqual(prediction.size, (32, 32))

    def test_hypergraph_backward(self):
        for method in ['threshold', 'knn']:
            with self.subTest(method=method):
                model = HypergraphComputation(16, construct_method=method)
                inputs = [torch.rand(2, 16, 5, 13, requires_grad=True) for _ in range(3)]
                outputs = model(*inputs)
                sum(output.square().mean() for output in outputs).backward()
                for value in inputs:
                    self.assertEqual(tuple(value.grad.shape), tuple(value.shape))
                    self.assertTrue(torch.isfinite(value.grad).all())


if __name__ == '__main__':
    unittest.main()
