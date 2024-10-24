import os
from typing import LiteralString

import dicom2nifti as d2n

from jbag.io import ensure_output_file_dir_existence


def nifti2dicom(input_file, output_dir, accession_number=1):
    """
    Require nifti2dicom.
    `sudo apt install nifti2dicom`
    https://github.com/biolab-unige/nifti2dicom
    Args:
        input_file (str or LiteralString):
        output_dir (str or LiteralString):
        accession_number (int, optional, default=1):

    Returns:

    """

    assert os.path.isfile(input_file), f'{input_file} does not exist or is not a file!'

    cmd = f'nifti2dicom -i {input_file} -o {output_dir} -a {accession_number}'
    result = os.popen(cmd)
    return result.readlines()


def dicom2nifti(input_dicom_series, output_nifti_file, pydicom_read_force=False):
    assert os.path.exists(input_dicom_series), f'{input_dicom_series} does not exist!'

    if pydicom_read_force:
        d2n.settings.pydicom_read_force = pydicom_read_force

    ensure_output_file_dir_existence(output_nifti_file)
    d2n.convert_directory(input_dicom_series, output_nifti_file)
