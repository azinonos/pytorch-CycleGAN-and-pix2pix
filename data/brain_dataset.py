import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageChops
import numpy as np


class BrainDataset(BaseDataset):
    """A dataset class for paired image dataset of brain slices.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    It also assumes that the files are of the format NNNN_Nw.ext eg: 0000_25w.png, 0001_8w.jpg
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            diff_map (tensor) - - the difference map resulting from A - B
            hist_diff (tensor) - - the difference of histograms of hist(A) - hist(B)
            time_period (int) - - the time period in weeks between A and B, read from the filename
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # Compute Difference Map between images and histogram difference
        diff_map = ImageChops.difference(A, B)
        hist_a, _ = np.histogram(np.asarray(A), bins=255)
        hist_b, _ = np.histogram(np.asarray(B), bins=255)
        hist_diff = (hist_a - hist_b)[np.newaxis, :]
        hist_diff = hist_diff / np.linalg.norm(hist_diff) # Normalize

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        diff_map_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)
        diff_map = diff_map_transform(diff_map)

        # Extract Time Period from filename
        time_period = int(AB_path.split('_')[-1].split('.')[0][:-1])

        return {'A': A, 'B': B, 'diff_map': diff_map, 'hist_diff': hist_diff, 'time_period': time_period, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
