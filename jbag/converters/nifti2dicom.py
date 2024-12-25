import os
import shutil

import nibabel as nib
import numpy as np
from pydicom import Dataset
from pydicom.uid import generate_uid

from jbag import logger
from jbag.converters.dicom import get_dicom_dataset, Modality


def write_slice(ds: Dataset, image_data, slice_index, output_dir):
    output_filename = r'Slice_%04d.dcm' % (slice_index + 1)
    image_slice = image_data[..., slice_index]
    ds.SOPInstanceUID = generate_uid(None)
    ds.PixelData = image_slice.tobytes()
    ds.save_as(os.path.join(output_dir, output_filename), write_like_original=False)


def get_nii2dcm_parameters(nii_data):
    nii_img = nii_data.get_fdata()

    dim_array = nii_data.header['dim'].astype(np.int16)
    num_x, num_y, num_z = int(dim_array[1]), int(dim_array[2]), int(dim_array[3])
    voxel_spacing_x, voxel_spacing_y, voxel_spacing_z = (nii_data.header['pixdim'][1],
                                                         nii_data.header['pixdim'][2],
                                                         nii_data.header['pixdim'][3])

    slice_indices = range(1, num_z + 1)
    last_location_z = (voxel_spacing_z * num_z) - voxel_spacing_z
    slice_locations = np.linspace(0, last_location_z, num=num_z)

    # Windowing
    max_i = np.max(nii_img)
    min_i = np.min(nii_img)
    window_center = round((max_i - min_i) / 2)
    window_width = round(max_i - min_i)

    rescale_intercept = 0
    rescale_slope = 1

    # FOV
    fov_x = num_x * voxel_spacing_x
    fov_y = num_y * voxel_spacing_y
    fov_z = num_z * voxel_spacing_z

    # slice positioning in 3-D space
    # -1 for direction cosines gives consistent orientation between Nifti and DICOM in ITK-Snap
    affine = nii_data.affine
    dir_cos_x = -1 * affine[:3, 0] / voxel_spacing_x
    dir_cos_y = -1 * affine[:3, 1] / voxel_spacing_y

    image_pos_patient_array = []
    for slice_i in range(0, num_z):
        v = affine.dot([0, 0, slice_i - 1, 1])
        image_pos_patient_array.append([v[0], v[1], v[2]])

    # output dictionary
    nii2dcm_parameters = {
        # series parameters
        'DimX': voxel_spacing_x,
        'DimY': voxel_spacing_y,
        'SliceThickness': str(voxel_spacing_z),
        'SpacingBetweenSlices': str(voxel_spacing_z),
        'AcquisitionMatrix': [0, num_x, num_y, 0],
        'Rows': num_x,
        'Columns': num_y,
        'NumberOfSlices': num_z,
        'NumberOfInstances': num_z,
        'PixelSpacing': [voxel_spacing_x, voxel_spacing_y],
        'FOV': [fov_x, fov_y, fov_z],
        'SmallestImagePixelValue': min_i,
        'LargestImagePixelValue': max_i,
        'WindowCenter': str(window_center),
        'WindowWidth': str(window_width),
        'RescaleIntercept': str(rescale_intercept),
        'RescaleSlope': str(rescale_slope),
        'ImageOrientationPatient': [dir_cos_y[0], dir_cos_y[1], dir_cos_y[2], dir_cos_x[0], dir_cos_x[1], dir_cos_x[2]],

        # instance parameters
        'InstanceNumber': slice_indices,
        'SliceLocation': slice_locations,
        'ImagePositionPatient': image_pos_patient_array
    }

    return nii2dcm_parameters


def transfer_dicom_series_tags(nii2dcm_parameters: dict, ds: Dataset):
    ds.Rows = nii2dcm_parameters['Rows']
    ds.Columns = nii2dcm_parameters['Columns']
    ds.PixelSpacing = [round(float(nii2dcm_parameters['DimX']), 2), round(float(nii2dcm_parameters['DimY']), 2)]
    ds.SliceThickness = nii2dcm_parameters['SliceThickness']
    ds.SpacingBetweenSlices = round(float(nii2dcm_parameters['SpacingBetweenSlices']), 2)
    ds.ImageOrientationPatient = nii2dcm_parameters['ImageOrientationPatient']
    ds.AcquisitionMatrix = nii2dcm_parameters['AcquisitionMatrix']
    ds.SmallestImagePixelValue = int(nii2dcm_parameters['SmallestImagePixelValue']) \
        if int(nii2dcm_parameters['SmallestImagePixelValue']) > 0 else 0
    ds.LargestImagePixelValue = int(nii2dcm_parameters['LargestImagePixelValue'])
    ds.WindowCenter = nii2dcm_parameters['WindowCenter']
    ds.WindowWidth = nii2dcm_parameters['WindowWidth']
    ds.RescaleIntercept = nii2dcm_parameters['RescaleIntercept']
    ds.RescaleSlope = nii2dcm_parameters['RescaleSlope']


def transfer_dicom_instance_tags(nii2dcm_parameters: dict, ds: Dataset, instance_index: int):
    ds.InstanceNumber = nii2dcm_parameters['InstanceNumber'][instance_index]
    ds.SliceLocation = nii2dcm_parameters['SliceLocation'][instance_index]
    ds.ImagePositionPatient = [
        str(nii2dcm_parameters['ImagePositionPatient'][instance_index][0]),
        str(nii2dcm_parameters['ImagePositionPatient'][instance_index][1]),
        str(nii2dcm_parameters['ImagePositionPatient'][instance_index][2]),
    ]


def nifti2dicom(input_nifti_file, output_dicom_dir, modality: Modality, force_overwrite=False):
    """
    Convert NIfTI image to DICOM image series. Inspired by https://github.com/tomaroberts/nii2dcm.

    Args:
        input_nifti_file (str or LiteralString):
        output_dicom_dir (str or LiteralString):
        modality (Modality):
        force_overwrite (bool, optional, default=False): If `Ture`, overwrite `output_dicom_dir` if it exists.`

    Returns:

    """

    assert os.path.isfile(input_nifti_file), f'{input_nifti_file} does not exist or is not a file!'

    if os.path.exists(output_dicom_dir):
        if force_overwrite:
            logger.info(f'Overwrite {output_dicom_dir} as it already exists.')
            shutil.rmtree(output_dicom_dir)
        else:
            raise FileExistsError(f'{output_dicom_dir} already exists.')

    nii_data = nib.load(input_nifti_file)

    nii2dcm_properties = get_nii2dcm_parameters(nii_data)
    file_name = os.path.splitext(os.path.basename(input_nifti_file))[0]

    dicom_ds = get_dicom_dataset(file_name, modality=modality)
    transfer_dicom_series_tags(nii2dcm_properties, dicom_ds)

    image = nii_data.get_fdata()
    image = image.astype(np.uint16)
    os.makedirs(output_dicom_dir)
    for instance_index in range(0, nii2dcm_properties['NumberOfInstances']):
        transfer_dicom_instance_tags(nii2dcm_properties, dicom_ds, instance_index)
        write_slice(dicom_ds, image, instance_index, output_dicom_dir)
