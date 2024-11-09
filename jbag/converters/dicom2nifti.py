import os

from jbag.io import ensure_output_file_dir_existence


def dicom2nifti(input_dicom_series, output_nifti_file, pydicom_read_force=False):
    import dicom2nifti as d2n

    assert os.path.exists(input_dicom_series), f'{input_dicom_series} does not exist!'

    if pydicom_read_force:
        d2n.settings.pydicom_read_force = pydicom_read_force

    ensure_output_file_dir_existence(output_nifti_file)
    d2n.convert_directory(input_dicom_series, output_nifti_file)
