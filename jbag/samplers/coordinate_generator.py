from abc import ABC, abstractmethod

import numpy as np

from .utils import get_margin, encode_one_hot


class CoordinateGenerator(ABC):

    @abstractmethod
    def __getitem__(self, index):
        ...

    @abstractmethod
    def __len__(self):
        ...

    @abstractmethod
    def regenerate(self):
        ...

    @staticmethod
    def get_non_margin_region(image_shape: tuple | list, patch_size: tuple | list):
        """
        Build a mask that cover the area which can be view as the central point of the patch with the size of
        patch_size. Mask the central area that can be extracted. Left the margin area.

        Args:
            image_shape (sequence):
            patch_size (sequence):

        Returns:

        """
        assert len(image_shape) == len(patch_size)
        margin = get_margin(patch_size)
        mask = np.zeros(image_shape, dtype=bool)
        mask[tuple([slice(j[0], i - j[1]) for i, j in zip(image_shape, margin)])] = 1
        return mask


class WeightedCoordinateGenerator(CoordinateGenerator):

    def __init__(self, num_coordinates, weights):
        """
        Choose coordinates according to the weight.

        Args:
            num_coordinates (int):
            weights (numpy.ndarray or torch.Tensor):
        """
        # number of coordinates that need to be picked.
        self.n = num_coordinates
        self.coordinate_system_shape = weights.shape
        self.a = weights.size

        p = weights / np.sum(weights)
        self.p = p.flatten()

        self.coordinates = self.get_coordinates()

    def __getitem__(self, index):
        return self.coordinates[index]

    def __len__(self):
        return self.n

    def get_coordinates(self):
        coordinates = np.random.choice(self.a, size=self.n, p=self.p)
        coordinates = np.asarray(np.unravel_index(coordinates, self.coordinate_system_shape))
        return coordinates.T

    def regenerate(self):
        self.coordinates = self.get_coordinates()


class BalancedCoordinateGenerator(WeightedCoordinateGenerator):
    def __init__(self, num_coordinates, label_map, patch_size):
        """
        Generate coordinates according to the proportion of categories. In general, the smaller the percentage of
        category, the more likely it is to be taken. Parameter data is the un-one-hot label.

        Args:
            num_coordinates (int):
            label_map (numpy.ndarray or torch.Tensor): The label matrix. this data will be needed to generate the weight map.
            patch_size (sequence):
        """
        self.n = num_coordinates
        self.patch_size = patch_size

        for i in range(len(label_map.shape)):
            assert label_map.shape[i] >= patch_size[i], f'Data size ({label_map.shape[i]}) along dimension {i} must not less than patch size ({patch_size[i]}).'
        super().__init__(num_coordinates=num_coordinates, weights=self.get_weights(label_map))

    def get_weights(self, label_map):
        # one-hot first
        label_onehot = encode_one_hot(label_map)

        # exclude margin
        valid_region = self.get_non_margin_region(label_map.shape, self.patch_size)
        weights = label_onehot * valid_region
        weights = weights.astype(np.float32)

        # normalize on every category axis
        category_sum = np.sum(weights, axis=tuple(range(1, len(label_map.shape) + 1)), keepdims=True)
        weights = weights / (category_sum + 1)
        weights = np.sum(weights, axis=0).astype(np.float32)
        return weights


class GridCoordinateGenerator(CoordinateGenerator):
    def __init__(self, original_shape, patch_shape, valid_shape):
        """

        Args:
            original_shape (sequence): The shape of original image.
            patch_shape (sequence): The shape of one patch.
            valid_shape (sequence): The size of valid patch. Valid shape should be less than patch shape. A valid patch
                needs to be extracted from a patch.
        """
        self.original_shape = np.asarray(original_shape, dtype=np.uint32)
        self.patch_shape = np.asarray(patch_shape, dtype=np.uint32)
        self.valid_shape = np.asarray(valid_shape, dtype=np.uint32)

        self.padding_size = get_margin(patch_shape) - get_margin(valid_shape)
        self.valid_central_coordinates = self.get_coordinates()
        self.padded_coordinates = self.valid_central_coordinates + self.padding_size[:, 0][None, :]

    def get_coordinates(self):
        """
        Get central coordinates according to the original shape and valid shape.

        Returns:

        """
        coordinate_dims = [range(0, i, j) for i, j in zip(self.original_shape, self.valid_shape)]
        coordinates = np.meshgrid(*coordinate_dims, indexing='ij')
        coordinates = list(map(lambda x: x.flatten(), coordinates))
        # get the corner coordinate of each patch
        coordinates = np.stack(coordinates, axis=-1)
        coordinates = np.where(coordinates + self.valid_shape > self.original_shape,
                               self.original_shape - self.valid_shape, coordinates)

        # shift coordinate to patch center
        shift = get_margin(self.valid_shape)[:, 0]
        central_coordinates = coordinates + shift
        return central_coordinates.astype(np.int32)

    def __getitem__(self, index):
        return self.padded_coordinates[index]

    def __len__(self):
        return len(self.padded_coordinates)

    def regenerate(self):
        ...
