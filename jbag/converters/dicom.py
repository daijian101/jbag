import datetime
from enum import Enum
from random import randint

from pydicom import Dataset
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import generate_uid


class Modality(Enum):
    CT = 1


def get_dicom_dataset(name, modality: Modality) -> FileDataset:
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
    file_meta.ImplementationVersionName = 'DICOM'

    ds = FileDataset(name, {}, file_meta=file_meta, preamble=b'\0' * 128)
    ds.is_implicit_VR = False
    ds.is_little_endian = True
    ds.ImageType = ['DERIVED', 'SECONDARY']

    # add properties
    patient(ds)
    general_study(ds)
    patient_study(ds)
    frame_of_reference(ds)
    general_equipment(ds)
    general_image(ds)
    general_acquisition(ds)
    image_plane(ds)
    image_pixel(ds)
    SOP_common(ds)
    VOI_LUT(ds)
    general_series(ds, file_meta, modality)

    # set datatime
    dt = datetime.datetime.now()
    date_str = dt.strftime('%Y%m%d')
    time_str = dt.strftime('%H%M%S.%f')  # long format with micro seconds

    ds.ContentDate = date_str
    ds.ContentTime = time_str
    ds.StudyDate = date_str
    ds.StudyTime = time_str
    ds.SeriesDate = date_str
    ds.SeriesTime = time_str
    ds.AcquisitionDate = date_str
    ds.AcquisitionTime = time_str
    ds.InstanceCreationDate = date_str
    ds.InstanceCreationTime = time_str

    ds.RescaleIntercept = ''
    ds.RescaleSlope = ''

    return ds


def patient(ds: Dataset):
    ds.PatientName = 'Patient Name'
    ds.PatientID = '1'
    ds.PatientSex = ''
    ds.PatientBirthDate = ''


def general_study(ds: Dataset):
    ds.StudyInstanceUID = generate_uid()
    ds.StudyDescription = ''
    ds.ReferringPhysicianName = ''


def patient_study(ds: Dataset):
    ds.PatientAge = ''
    ds.PatientWeight = ''


def general_series(ds: Dataset, file_meta: FileMetaDataset, modality: Modality):
    ds.Modality = modality.name
    match modality:
        case Modality.CT:
            file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
            ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
            CT_image(ds)

    ds.SeriesInstanceUID = generate_uid(None)
    ds.SeriesNumber = str(randint(1000, 9999))
    ds.ProtocolName = 'DICOM'
    ds.PatientPosition = ''
    ds.AccessionNumber = '123456'


def frame_of_reference(ds: Dataset):
    ds.FrameOfReferenceUID = generate_uid(None)


def general_equipment(ds: Dataset):
    ds.Manufacturer = ''
    ds.InstitutionName = 'INSTITUTION_NAME_UNDEFINED'
    ds.ManufacturerModelName = ''
    ds.SoftwareVersions = ''


def general_image(ds: Dataset):
    ds.InstanceNumber = ''
    ds.PatientOrientation = ''
    ds.ContentDate = ''
    ds.ContentTime = ''
    ds.ImageType = ['SECONDARY', 'DERIVED']
    ds.LossyImageCompression = '00'


def general_acquisition(ds: Dataset):
    ds.AcquisitionNumber = ''
    ds.AcquisitionDate = ''
    ds.AcquisitionTime = ''


def image_plane(ds: Dataset):
    ds.PixelSpacing = ''
    ds.ImageOrientationPatient = ['1', '0', '0', '0', '1', '0']
    ds.ImagePositionPatient = ['0', '0', '0']
    ds.SliceThickness = ''
    ds.SpacingBetweenSlices = ''
    ds.SliceLocation = ''


def image_pixel(ds: Dataset):
    ds.Rows = 0
    ds.Columns = 0

    ds.BitsAllocated = 0
    ds.BitsStored = 0
    ds.HighBit = 0

    ds.PixelRepresentation = 0

    ds.SmallestImagePixelValue = ''
    ds.LargestImagePixelValue = ''

    ds.PixelData = ''


def SOP_common(ds: Dataset):
    ds.SOPClassUID = ''
    ds.SOPInstanceUID = ''

    ds.SpecificCharacterSet = 'ISO_IR 100'
    ds.InstanceCreationDate = ''
    ds.InstanceCreationTime = ''

    ds.InstanceCreatorUID = ''


def VOI_LUT(ds: Dataset):
    ds.WindowCenter = ''
    ds.WindowWidth = ''


def CT_image(ds: Dataset):
    ds.SamplesPerPixel = 1

    ds.PhotometricInterpretation = 'MONOCHROME2'

    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = ds.BitsStored - 1
