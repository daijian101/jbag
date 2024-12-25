from typing import Union

import numpy as np
import torch
from scipy.ndimage import fourier_gaussian
from torch.nn.functional import grid_sample

from jbag.samplers.utils import central_crop
from jbag.transforms._utils import get_scalar
from jbag.transforms import RandomTransform


class SpatialTransform(RandomTransform):
    def __init__(self, keys,
                 apply_probability,
                 patch_size=None,
                 patch_center_dist_from_border: Union[int, list[int], tuple[int]] = 0,
                 random_crop: bool = False,
                 interpolation_modes: Union[str, list[str], tuple[str]] = 'nearest',
                 p_rotation: float = 0.0,
                 rotation_angle_range: Union[float, list[float], tuple[float, float]] = (0, 2 * np.pi),
                 p_scaling: float = 0.0,
                 scaling_range: Union[float, list[float], tuple[float, float]] = (0.7, 1.3),
                 p_synchronize_scaling_across_axes: float = 1,
                 p_elastic_deform: float = 0.0,
                 elastic_deform_scale: Union[float, list[float], tuple[float, float]] = (0, 0.2),
                 elastic_deform_magnitude: Union[float, list[float], tuple[float, float]] = (0, 0.2),
                 ):
        """
        Spatial affine transform. The input data should be in the dimension format of [C, [H, W, D]].
        Args:
            keys (str or sequence):
            apply_probability (float):
            patch_size (sequence or None, optional, default=None):
            random_crop (bool, optional, default=False): If True, crop patch image randomly.
            interpolation_modes (str or sequence(str), optional, default='nearest'): interpolation modes for value of keys.
            Supported options ``'nearest'`` | ``'bilinear'``.
            p_rotation (float, optional, default=0): Probability of applying rotation.
            rotation_angle_range (float or sequence, optional, default=[0, 2 * np.pi]):
            p_scaling (float, optional, default=0): Probability of applying scaling.
            scaling_range (float or sequence, optional, default=[0.7, 1.3]):
            p_synchronize_scaling_across_axes (float, optional, default=1):
            p_elastic_deform (float, optional, default=0): Probability of applying elastic deform.
            elastic_deform_scale (float or sequence, optional, default=[0, 0.2]):
            elastic_deform_magnitude (float or sequence, optional, default=[0, 0.2]):

        """
        super().__init__(keys, apply_probability)
        assert patch_center_dist_from_border is not None

        if patch_size is not None:
            patch_size = tuple(patch_size)
        self.patch_size = patch_size
        if patch_size is not None and not isinstance(patch_center_dist_from_border, (tuple, list)):
            patch_center_dist_from_border = [patch_center_dist_from_border] * len(patch_size)
        self.patch_center_dist_from_border = patch_center_dist_from_border

        self.p_rotation = p_rotation
        self.p_scaling = p_scaling
        self.p_elastic_deform = p_elastic_deform
        self.random_crop = random_crop
        if isinstance(interpolation_modes, str):
            interpolation_modes = [interpolation_modes] * len(self.keys)

        self.interpolation_modes = interpolation_modes

        self.rotation_angle_range = rotation_angle_range
        self.scaling = scaling_range
        self.p_synchronize_scaling_across_axes = p_synchronize_scaling_across_axes
        self.elastic_deform_scale = elastic_deform_scale
        self.elastic_deform_magnitude = elastic_deform_magnitude

    def _call_fun(self, data):
        # the process here comes from nnUnet v2
        do_grid_sampling = False
        original_spatial_shape = data[self.keys[0]].shape[1:]
        spatial_dims = len(original_spatial_shape)
        patch_size = self.patch_size if self.patch_size is not None else original_spatial_shape
        assert len(original_spatial_shape) == len(patch_size)

        do_rotation = np.random.random_sample() < self.p_rotation
        do_scale = np.random.random_sample() < self.p_scaling
        do_deform = np.random.random_sample() < self.p_elastic_deform
        if do_rotation:
            angles = [get_scalar(self.rotation_angle_range) for _ in range(spatial_dims)]
        else:
            angles = [0] * spatial_dims

        if do_scale:
            scales = [get_scalar(self.scaling) for _ in
                      range(spatial_dims)] if np.random.random_sample() < self.p_synchronize_scaling_across_axes \
                else [get_scalar(self.scaling)] * spatial_dims
        else:
            scales = [1] * spatial_dims

        if do_scale or do_rotation:
            do_grid_sampling = True
            if spatial_dims == 3:
                affine = create_affine_matrix_3d(angles, scales)
            elif spatial_dims == 2:
                affine = create_affine_matrix_2d(angles[0], scales)
            else:
                raise RuntimeError(f'Unsupported dimension: {spatial_dims}')
        else:
            affine = None  # this will allow us to detect that we can skip computations

        if do_deform:
            do_grid_sampling = True
            grid_scale = [i / j for i, j in zip(original_spatial_shape, patch_size)]
            deformation_scales = [
                get_scalar(self.elastic_deform_scale) for _ in
                range(spatial_dims)] if np.random.random_sample() < self.p_synchronize_scaling_across_axes \
                else [get_scalar(self.elastic_deform_scale)] * spatial_dims
            # sigmas must be in pixels, as this will be applied to the deformation field
            sigmas = [i * j for i, j in zip(deformation_scales, patch_size)]
            # the magnitude of the deformation field must adhere to the torch's value range for grid_sample, i.e. [-1. 1] and not pixel coordinates. Do not use sigmas here
            # we need to correct magnitude by grid_scale to account for the fact that the grid will be wrt to the image size but the magnitude should be wrt the patch size. oof.
            magnitude = [
                get_scalar(self.elastic_deform_magnitude) / grid_scale[i] for i in range(spatial_dims)]
            # doing it like this for better memory layout for blurring
            elastic_offsets = torch.normal(mean=0, std=1, size=(spatial_dims, *patch_size))

            # all the additional time elastic deform takes is spent here
            for d in range(spatial_dims):
                # np fft is faster than torch
                tmp = np.fft.fftn(elastic_offsets[d].numpy())
                tmp = fourier_gaussian(tmp, sigmas)
                elastic_offsets[d] = torch.from_numpy(np.fft.ifftn(tmp).real)

                mx = torch.max(torch.abs(elastic_offsets[d]))
                elastic_offsets[d] /= (mx / np.clip(magnitude[d], a_min=1e-8, a_max=np.inf))
            elastic_offsets = torch.permute(elastic_offsets, (1, 2, 3, 0))
        else:
            elastic_offsets = None

        if self.random_crop:
            for i in range(len(original_spatial_shape)):
                assert patch_size[i] + self.patch_center_dist_from_border[i] * 2 <= original_spatial_shape[i], \
                    f'patch_size in spatial dimension {i} is {patch_size[i]}, patch center distance from border is {self.patch_center_dist_from_border[i]}, original shape {original_spatial_shape[i]} does not match the size requirement.'
            center_location = []
            for i in range(len(original_spatial_shape)):
                dist_from_boarder = self.patch_center_dist_from_border[i]
                mn = patch_size[i] // 2 + dist_from_boarder
                mx = original_spatial_shape[i] - patch_size[i] // 2 - self.patch_center_dist_from_border[i]
                assert mn <= mx
                center_location.append(np.random.randint(mn, mx))
        else:
            center_location = [i // 2 for i in original_spatial_shape]

        if do_grid_sampling:
            grid = _create_grid(patch_size)
            grid_scale = torch.Tensor([i / j for i, j in zip(original_spatial_shape, patch_size)])
            grid /= grid_scale

            if elastic_offsets is not None:
                grid += elastic_offsets
            if affine is not None:
                grid = torch.matmul(grid, torch.from_numpy(affine).float())

            mn = grid.mean(dim=list(range(len(original_spatial_shape))))
            new_center = torch.Tensor(
                [(j / (i / 2) - 1) for i, j in zip(original_spatial_shape, center_location)])
            grid += - mn + new_center

        for i, key in enumerate(self.keys):
            value = data[key]
            # check if affine transform is needed
            if do_grid_sampling:
                dtype = value.dtype
                if dtype is not torch.float32:
                    value = value.to(torch.float32)
                value = grid_sample(value[None], grid[None], mode=self.interpolation_modes[i], padding_mode='zeros',
                                    align_corners=False).to(dtype)[0]
            else:
                if patch_size != original_spatial_shape:
                    value = central_crop(value, center_location, self.patch_size)
            data[key] = value
            if do_grid_sampling:
                data['spatial'] = True
            else:
                data['spatial'] = False
        return data


def _create_grid(size: list[int]):
    space = [torch.linspace((-s + 1) / s, (s - 1) / s, s) for s in size[::-1]]
    grid = torch.meshgrid(space, indexing='ij')
    grid = torch.stack(grid, -1)
    spatial_dims = list(range(len(size)))
    grid = grid.permute((*spatial_dims[::-1], len(size)))
    return grid


def create_affine_matrix_3d(rotation_angles, scaling_factors):
    # Rotation matrices for each axis
    rotation_x = np.array([[1, 0, 0],
                   [0, np.cos(rotation_angles[0]), -np.sin(rotation_angles[0])],
                   [0, np.sin(rotation_angles[0]), np.cos(rotation_angles[0])]])

    rotation_y = np.array([[np.cos(rotation_angles[1]), 0, np.sin(rotation_angles[1])],
                   [0, 1, 0],
                   [-np.sin(rotation_angles[1]), 0, np.cos(rotation_angles[1])]])

    rotation_z = np.array([[np.cos(rotation_angles[2]), -np.sin(rotation_angles[2]), 0],
                   [np.sin(rotation_angles[2]), np.cos(rotation_angles[2]), 0],
                   [0, 0, 1]])

    # Scaling matrix
    scaling = np.diag(scaling_factors)

    # Combine rotation and scaling
    affine = rotation_z @ rotation_y @ rotation_x @ scaling
    return affine


def create_affine_matrix_2d(rotation_angle, scaling_factors):
    # Rotation matrix
    rotation = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                  [np.sin(rotation_angle), np.cos(rotation_angle)]])

    # Scaling matrix
    scaling = np.diag(scaling_factors)

    # Combine rotation and scaling
    affine = rotation @ scaling
    return affine
