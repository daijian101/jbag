import os

from jbag.io import ensure_output_file_dir_existence
import dicom2nifti as d2n


def dicom2nifti(input_dicom_series, output_nifti_file, pydicom_read_force=False):
    if not os.path.exists(input_dicom_series):
        raise ValueError(f"Input DICOM series {input_dicom_series} does not exist.")

    if pydicom_read_force:
        d2n.settings.pydicom_read_force = pydicom_read_force

    ensure_output_file_dir_existence(output_nifti_file)
    d2n.dicom_series_to_nifti(input_dicom_series, output_nifti_file, reorient_nifti=False)
