from abc import ABC, abstractmethod

import numpy as np

from .utils import get_margin, get_non_margin_region


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


class WeightedCoordinateGenerator(CoordinateGenerator):

    def __init__(self, num_coordinates, weights):
        """
        Choose coordinates according to the weight.

        Args:
            num_coordinates (int):
            weights (numpy.ndarray or torch.Tensor): non-negative sampling weights with at least one positive value.
                When the generated coordinates are used as patch centers, the weights must be zero in the margin
                that cannot hold a full patch (see :func:`jbag.samplers.utils.get_non_margin_region`).
        """
        num_coordinates = int(num_coordinates)
        if num_coordinates < 0:
            raise ValueError(f'The number of coordinates must be non-negative, but got {num_coordinates}.')

        # Accept both numpy arrays and (CPU) torch tensors. Converting first keeps every later operation in numpy.
        if hasattr(weights, 'detach'):
            weights = weights.detach().cpu().numpy()
        weights = np.asarray(weights)
        if weights.ndim == 0:
            raise ValueError('The weights must be an array with at least one dimension.')
        if not np.issubdtype(weights.dtype, np.floating):
            weights = weights.astype(np.float64)
        if not np.all(np.isfinite(weights)):
            raise ValueError('The weights must be finite.')
        if np.any(weights < 0):
            raise ValueError('The weights must be non-negative.')
        total = float(weights.sum())
        if total <= 0:
            raise ValueError('The weights must contain at least one positive value.')

        # number of coordinates that need to be picked.
        self.n = num_coordinates
        self.coordinate_system_shape = weights.shape
        self.a = int(weights.size)

        self.p = (weights / total).flatten()
        self.coordinates = self.get_coordinates()

    def __getitem__(self, index):
        return self.coordinates[index]

    def __len__(self):
        return self.n

    def get_coordinates(self):
        coordinates = np.random.choice(self.a, size=self.n, p=self.p)
        coordinates = np.stack(np.unravel_index(coordinates, self.coordinate_system_shape), axis=-1)
        return coordinates

    def regenerate(self):
        self.coordinates = self.get_coordinates()


class BalancedCoordinateGenerator(WeightedCoordinateGenerator):
    def __init__(self, num_coordinates, label_map, patch_size):
        """
        Generate coordinates according to the proportion of categories. In general, the smaller the percentage of
        category, the more likely it is to be taken. `label_map` is the un-one-hot label map.

        Negative labels (e.g. an ignore label of -1) are excluded from sampling: patch centers are never placed on
        them, while they may still appear inside a patch.

        Args:
            num_coordinates (int):
            label_map (numpy.ndarray or torch.Tensor): the label matrix, needed to generate the weight map. It must
                have exactly the same number of dimensions as `patch_size`, without channel/batch dimensions.
            patch_size (sequence):
        """
        self.patch_size = np.atleast_1d(np.asarray(patch_size, dtype=np.int64))
        if self.patch_size.ndim != 1:
            raise ValueError(
                f'The patch size must be a 1-dimensional sequence, but got shape {self.patch_size.shape}.')

        if hasattr(label_map, 'detach'):
            label_map = label_map.detach().cpu().numpy()
        label_map = np.asarray(label_map)
        if label_map.ndim != self.patch_size.size:
            raise ValueError(
                f'The label map must have exactly {self.patch_size.size} dimensions to match the patch size '
                f'{self.patch_size.tolist()}, but got shape {label_map.shape}. Remove any channel or batch '
                f'dimension (e.g. `label_map[0]`) before passing it to the sampler.')
        if np.any(np.asarray(label_map.shape) < self.patch_size):
            raise ValueError(
                f'The label map shape {label_map.shape} must not be smaller than the patch size '
                f'{self.patch_size.tolist()}.')

        super().__init__(num_coordinates=num_coordinates, weights=self.get_weights(label_map))

    def get_weights(self, weight_label_map):
        """
        Build per-voxel sampling weights so that every category contributes approximately the same probability.

        A voxel of category ``c`` gets weight ``1 / (count(c) + 1)``, where ``count(c)`` is the number of voxels of
        category ``c`` inside the valid region. This is mathematically equivalent to the old one-hot implementation
        but never materialises an ``(num_voxels, num_classes)`` array.

        Args:
            weight_label_map (numpy.ndarray): integer label map without channel dimensions.

        Returns:
            numpy.ndarray: float32 weight map with the same spatial shape as `weight_label_map`.

        """
        label = np.asarray(weight_label_map)
        if np.issubdtype(label.dtype, np.bool_):
            label = label.astype(np.int64)
        elif np.issubdtype(label.dtype, np.integer):
            label = label.astype(np.int64, copy=False)
        elif np.issubdtype(label.dtype, np.floating):
            if not np.all(np.isfinite(label)) or not np.array_equal(label, np.floor(label)):
                raise ValueError('Floating point labels must be finite and integer-valued.')
            label = label.astype(np.int64)
        else:
            raise TypeError(f'Unsupported label dtype {label.dtype!r}; an integer label map is required.')

        valid_region = get_non_margin_region(label.shape, self.patch_size)
        valid = valid_region & (label >= 0)
        if not np.any(valid):
            raise ValueError(
                'No valid (non-negative) labels are available for sampling. Check `label_map` and `patch_size`.')

        labels = label[valid]
        counts = np.bincount(labels)
        # Build float32 weights directly to avoid a full-size float64 temporary for large 3D volumes.
        weights = np.zeros(label.shape, dtype=np.float32)
        weights[valid] = (1.0 / (counts[labels] + 1.0)).astype(np.float32)
        return weights


class GridCoordinateGenerator(CoordinateGenerator):
    def __init__(self, original_shape, patch_shape, valid_shape):
        """

        Args:
            original_shape (sequence): the shape of original improc.
            patch_shape (sequence): the shape of one patch.
            valid_shape (sequence): the size of valid patch. Valid shape should be less than patch shape. A valid patch
                needs to be extracted from a patch.
        """
        self.original_shape = np.atleast_1d(np.asarray(original_shape, dtype=np.int64))
        self.patch_shape = np.atleast_1d(np.asarray(patch_shape, dtype=np.int64))
        self.valid_shape = np.atleast_1d(np.asarray(valid_shape, dtype=np.int64))

        if not (self.original_shape.ndim == self.patch_shape.ndim == self.valid_shape.ndim == 1):
            raise ValueError('The original shape, patch shape and valid shape must be 1-dimensional sequences.')
        if not (self.original_shape.size == self.patch_shape.size == self.valid_shape.size):
            raise ValueError(
                f'The original shape, patch shape and valid shape must have the same length, but got '
                f'{self.original_shape.size}, {self.patch_shape.size} and {self.valid_shape.size}.')
        if np.any(self.original_shape <= 0) or np.any(self.patch_shape <= 0) or np.any(self.valid_shape <= 0):
            raise ValueError('The original shape, patch shape and valid shape must contain positive values.')
        if np.any(self.valid_shape > self.patch_shape):
            raise ValueError(
                f'The valid shape {self.valid_shape.tolist()} must not exceed the patch shape '
                f'{self.patch_shape.tolist()}.')
        if np.any(self.valid_shape > self.original_shape):
            raise ValueError(
                f'The valid shape {self.valid_shape.tolist()} must not exceed the original shape '
                f'{self.original_shape.tolist()}.')

        self.padding_size = get_margin(patch_shape) - get_margin(valid_shape)
        self.valid_central_coordinates = self.get_coordinates()
        self.padded_coordinates = self.valid_central_coordinates + self.padding_size[:, 0][None, :]

    def get_coordinates(self):
        """
        Get central coordinates according to the original shape and valid shape.

        Returns:

        """
        coordinate_dims = [range(0, int(i), int(j)) for i, j in zip(self.original_shape, self.valid_shape)]
        coordinates = np.meshgrid(*coordinate_dims, indexing="ij")
        coordinates = np.stack([x.flatten() for x in coordinates], axis=-1)
        # get the corner coordinate of each patch
        coordinates = np.minimum(coordinates, self.original_shape - self.valid_shape)

        # shift coordinate to patch center
        shift = get_margin(self.valid_shape)[:, 0]
        central_coordinates = coordinates + shift
        return central_coordinates.astype(np.int64)

    def __getitem__(self, index):
        return self.padded_coordinates[index]

    def __len__(self):
        return len(self.padded_coordinates)

    def regenerate(self):
        """The grid is deterministic, so there is nothing to regenerate."""
        return None
