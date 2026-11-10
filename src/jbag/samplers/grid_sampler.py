import numpy as np
import torch
import torch.nn.functional as F

from jbag.samplers.coordinate_generator import GridCoordinateGenerator
from jbag.samplers.utils import central_crop, get_margin


class GridSampler:
    def __init__(self, data: np.ndarray | torch.Tensor, patch_size, valid_size=None):
        """
        Perform grid sample and recovery. The last N dimensions of given data will be split.

        Args:
            data (numpy.ndarray or torch.Tensor): the data needs to be sample.
            patch_size (sequence): the data will be divided into patches with the shape of patch_shape.
            valid_size (sequence or None, optional, default=None): valid patch size, used for restore. If `None`, as
                same as `patch_size`.
        """
        if not isinstance(data, (np.ndarray, torch.Tensor)):
            raise TypeError(f'Data must be a numpy.ndarray or torch.Tensor, but got {type(data)!r}.')

        patch_size = np.atleast_1d(np.asarray(patch_size, dtype=np.int64))
        valid_size = patch_size.copy() if valid_size is None \
            else np.atleast_1d(np.asarray(valid_size, dtype=np.int64))
        if patch_size.ndim != 1 or valid_size.ndim != 1:
            raise ValueError('The patch size and valid size must be 1-dimensional sequences.')
        if patch_size.size != valid_size.size:
            raise ValueError(
                f'The patch size ({patch_size.size} dims) and valid size ({valid_size.size} dims) must have the '
                f'same number of dimensions.')
        if patch_size.size > data.ndim:
            raise ValueError(
                f'The patch size {patch_size.tolist()} has more dimensions than the data shape {tuple(data.shape)}.')
        if np.any(patch_size <= 0) or np.any(valid_size <= 0):
            raise ValueError('The patch size and valid size must contain positive values.')
        if np.any(valid_size > patch_size):
            raise ValueError(
                f'The valid size {valid_size.tolist()} must not exceed the patch size {patch_size.tolist()}.')

        self.patch_size = patch_size
        self.valid_size = valid_size
        self.full_size = tuple(int(s) for s in data.shape)
        # shape of data, but batch and/or channel dimensions are excluded
        self.original_shape = self.full_size[-len(self.patch_size):]
        spatial_shape = np.asarray(self.original_shape, dtype=np.int64)
        if np.any(self.valid_size > spatial_shape):
            raise ValueError(
                f'The valid size {self.valid_size.tolist()} must not exceed the spatial shape of the data '
                f'{self.original_shape}.')

        self.coordinate_generator = GridCoordinateGenerator(self.original_shape, patch_shape=self.patch_size,
                                                            valid_shape=self.valid_size)

        self._need_padding = not np.array_equal(self.patch_size, self.valid_size)
        if self._need_padding:
            padding = self.coordinate_generator.padding_size
            if isinstance(data, torch.Tensor):
                pad_size = tuple(int(v) for v in np.flip(padding, axis=0).ravel())
                self.padded_data = F.pad(data, pad=pad_size, mode='constant', value=0)
            else:
                pad_width = [(0, 0)] * (data.ndim - len(padding)) + \
                            [(int(left), int(right)) for left, right in padding]
                self.padded_data = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)
        else:
            self.padded_data = data
        self.index = 0

    def restore(self, patches, restore_shape=None):
        """
        Every element's batch/channel dimensions in blocks should be put at the beginning.

        Patches are expected in the same order as the order produced by iterating over this sampler. When the sampled
        coordinates overlap (which happens when the spatial size is not a multiple of `valid_size`), later patches
        overwrite earlier ones.

        Args:
            patches (sequence):
            restore_shape (sequence or None, optional, default=None): restore array shape. Batch/channel dimensions
                of restored array can differ from the data which was divided as long as keeping the same spatial
                dimensions. If None, as same as `self.full_size`.
        """
        patches = list(patches)
        if not patches:
            raise ValueError('Patches must not be empty.')

        if restore_shape is None:
            restore_shape = self.full_size
        else:
            try:
                restore_shape = tuple(int(s) for s in restore_shape)
            except TypeError as e:
                raise TypeError(
                    f'The restore shape must be a sequence of integers, but got {type(restore_shape)!r}.') from e

        patches = [self._shrink_shape(patch) for patch in patches]
        return self._rebuild_image(patches, restore_shape)

    def _shrink_shape(self, data: np.ndarray | torch.Tensor):
        """
        Shrink data to assigned shape.

        Args:
            data (numpy.ndarray or torch.Tensor):

        Returns:

        """
        if not self._need_padding:
            return data

        expected_shape = tuple(int(s) for s in self.patch_size)
        if tuple(data.shape[-len(expected_shape):]) != expected_shape:
            raise ValueError(
                f'restore() expects patches with spatial shape {expected_shape}, but got data shape '
                f'{tuple(data.shape)}.')

        center = get_margin(self.valid_size)[:, 0] + self.coordinate_generator.padding_size[:, 0]
        return central_crop(data, center, self.valid_size)

    def _rebuild_image(self, patches, restore_shape):
        first = patches[0]
        is_numpy = isinstance(first, np.ndarray)
        if not is_numpy and not isinstance(first, torch.Tensor):
            raise TypeError(f'Each patch must be a numpy.ndarray or torch.Tensor, but got {type(first)!r}.')

        expected_shape = tuple(int(s) for s in self.valid_size)
        for patch in patches:
            if is_numpy and not isinstance(patch, np.ndarray) or not is_numpy and not isinstance(patch, torch.Tensor):
                raise TypeError('All patches must be of the same type (all numpy.ndarray or all torch.Tensor).')
            if tuple(patch.shape[-len(expected_shape):]) != expected_shape:
                raise ValueError(
                    f'Each patch must have spatial shape {expected_shape}, but got {tuple(patch.shape)}.')

        expected_patches = len(self.coordinate_generator)
        if len(patches) != expected_patches:
            raise ValueError(
                f'Expected {expected_patches} patches for restore_shape {restore_shape}, but got {len(patches)}.')

        if len(restore_shape) < len(self.valid_size):
            raise ValueError(
                f'The restore shape {restore_shape} must have at least {len(self.valid_size)} dimensions.')

        block_margin = get_margin(self.valid_size)
        margin = block_margin[:, 0][None, :]
        coordinates = self.coordinate_generator.valid_central_coordinates - margin
        spatial_restore_shape = np.asarray(restore_shape[-len(self.valid_size):], dtype=np.int64)
        if np.any(coordinates < 0) or np.any(coordinates + self.valid_size > spatial_restore_shape):
            raise ValueError(
                f'The restore shape {restore_shape} is too small for the sampled coordinates.')

        result = np.zeros(shape=restore_shape, dtype=first.dtype) if is_numpy \
            else torch.zeros(size=restore_shape, dtype=first.dtype, device=first.device)

        extra_dimensions = [slice(None)] * (len(restore_shape) - len(self.valid_size))
        for i, block in enumerate(patches):
            slice_sequence = extra_dimensions + \
                             [slice(int(coordinates[i, j]), int(coordinates[i, j] + self.valid_size[j]))
                              for j in range(len(self.valid_size))]

            result[tuple(slice_sequence)] = block
        return result

    def reset(self):
        """Reset the iteration index so that the sampler can be iterated from the beginning again."""
        self.index = 0

    def __len__(self):
        return len(self.coordinate_generator)

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self.index >= len(self):
            raise StopIteration

        patch = central_crop(self.padded_data, self.coordinate_generator[self.index], self.patch_size)
        self.index += 1
        return patch
