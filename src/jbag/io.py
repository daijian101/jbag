import json
import os.path
from base64 import b64encode, b64decode
from collections import OrderedDict

import numpy as np
import pandas as pd
from numpy.lib.format import dtype_to_descr, descr_to_dtype
from openpyxl import load_workbook


from jbag import logger


def read_file_to_list(input_file):
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f'Input file {input_file} not found.')

    with open(input_file, 'r') as input_file:
        return [each.strip('\n') for each in input_file.readlines()]


def save_list_to_file(output_file, data_list):
    """
    Writes each item from a list to a new line in a file.

    Args:
        output_file (str):
        data_list (list):

    Returns:

    """
    ensure_output_file_dir_existence(output_file)
    with open(output_file, 'w') as file:
        for i in range(len(data_list)):
            file.write(str(data_list[i]))
            if i != len(data_list) - 1:
                file.write('\n')


# JSON
def np_object_hook(dct):
    """
    Convert JSON list or scalar to numpy.

    Args:
        dct (mapping):

    Returns:

    """
    if '__ndarray__' in dct:
        shape = dct['shape']
        dtype = descr_to_dtype(dct['dtype'])
        if shape:
            order = 'C' if dct['Corder'] else 'F'
            if dct['base64']:
                np_obj = np.frombuffer(b64decode(dct['__ndarray__']), dtype=dtype)
                np_obj = np_obj.copy(order=order)
            else:
                np_obj = np.asarray(dct['__ndarray__'], dtype=dtype, order=order)
            return np_obj.reshape(shape)

        if dct['base64']:
            np_obj = np.frombuffer(b64decode(dct['__ndarray__']), dtype=dtype)[0]
        else:
            t = getattr(np, dtype.name)
            np_obj = t(dct['__ndarray__'])
        return np_obj

    return dct


def read_json(input_json_file):
    """
    Read and convert `input_json_file`.

    Args:
        input_json_file (str):

    Returns:

    """
    if not os.path.isfile(input_json_file):
        raise FileNotFoundError(f'Input file {input_json_file} not found.')

    with open(input_json_file, 'r') as json_file:
        dct = json.load(json_file, object_hook=np_object_hook)
    return dct


class NumpyJSONEncoder(json.JSONEncoder):
    def __init__(self, primitive=False, base64=True, **kwargs):
        """
        JSON encoder for `numpy.ndarray`.

        Args:
            primitive (bool, optional, default=False): use primitive type if `True`. In primitive schema,
                `numpy.ndarray` is stored as JSON list and `np.generic` is stored as a number.
            base64 (bool, optional, default=True): use base64 to encode.
        """
        self.primitive = primitive
        self.base64 = base64
        super().__init__(**kwargs)

    def default(self, obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            if self.primitive:
                return obj.tolist()
            else:
                if self.base64:
                    data_json = b64encode(obj.data if obj.flags.c_contiguous else obj.tobytes()).decode('ascii')
                else:
                    data_json = obj.tolist()
                dct = OrderedDict(__ndarray__=data_json,
                                  dtype=dtype_to_descr(obj.dtype),
                                  shape=obj.shape,
                                  Corder=obj.flags['C_CONTIGUOUS'],
                                  base64=self.base64)
                return dct
        return super().default(obj)


def save_json(output_file, obj, primitive=False, base64=True):
    """
    Convert obj to JSON object and save as file.

    Args:
        output_file (str):
        obj (mapping):
        primitive (bool, optional, default=False): use primitive type if `True`. In primitive schema, `numpy.ndarray` is
            stored as JSON list and `np.generic` is stored as a number.
        base64 (bool, optional, default=True): use base64 to encode.

    Returns:

    """
    ensure_output_file_dir_existence(output_file)
    with open(output_file, 'w') as file:
        json.dump(obj, file, cls=NumpyJSONEncoder, **{'primitive': primitive, 'base64': base64})


def scp(dst_user, dst_host, dst_path, local_path, dst_port=None, recursive=False, send=False, receive=False):
    """
    Transmit file(s) through scp.

    Args:
        dst_user (str):
        dst_host (str):
        dst_path (str):
        local_path (str):
        dst_port (str or int or None, optional, default=None): if None, usually refer to port 22.
        recursive (bool, default=False): transmit directories recursively.
        send (bool, default=False): send file(s) from local to destination.
        receive (bool, default=False): receive file(s) sent from destination.

    Returns:

    """
    if not (send ^ receive):
        raise ValueError(f'Send and receive must be exclusive.')

    cmd = 'scp'
    dst = f'{dst_user}@{dst_host}:{dst_path}'
    if recursive:
        cmd += ' -r'
    if dst_port:
        cmd += f' -P {dst_port}'
    if send:
        cmd += f' {local_path} {dst}'
    else:
        cmd += f' {dst} {local_path}'
    os.system(cmd)


def save_excel(output_file, data: dict | pd.DataFrame, sheet_name: str = 'Sheet1', append: bool = False,
               overlay_sheet: bool = False,
               column_width: int = None, auto_adjust_width: bool = False, index=False):
    """
    Save data to Excel file.
    Args:
        output_file (str):
        data (dict | pd.DataFrame):
        sheet_name (str, optional, default='Sheet1'):
        append (bool, optional, default=False): if True, append to existing file.
        overlay_sheet (bool, optional, default=False): if True, overwrite existing sheet. Note that this option only works for appending mode.
        column_width (int, optional, default=None): set column width to the given value, if exist.
        auto_adjust_width (bool, optional, default=False): if True, adjust column width according to the content length.
        index (bool, optional, default=False): if True, set index column on the work sheet.

    Returns:

    """
    write_mode = 'a' if append else 'w'

    if write_mode == 'a' and not os.path.exists(output_file):
        logger.warning(f'Try to append data to a non-existing file: {output_file}, change mode to write instead.')
        write_mode = 'w'

    if write_mode == 'a' and overlay_sheet:
        if_sheet_exists = 'overlay'
    else:
        if_sheet_exists = None

    if isinstance(data, dict):
        data = pd.DataFrame(data)

    ensure_output_file_dir_existence(output_file)

    with pd.ExcelWriter(output_file, engine='openpyxl', mode=write_mode, if_sheet_exists=if_sheet_exists) as writer:
        data.to_excel(writer, sheet_name=sheet_name, index=index)

    if column_width is not None or auto_adjust_width:
        book = load_workbook(output_file)
        worksheet = book[sheet_name]

        for column_cells in worksheet.columns:
            if column_width is not None:
                width = column_width
            else:
                max_length = max(len(str(cell.value)) for cell in column_cells)
                width = max_length + 5

            column_letter = column_cells[0].column_letter
            worksheet.column_dimensions[column_letter].width = width

        # Save the workbook
        book.save(output_file)


def ensure_output_dir_existence(output_dir):
    mk_output_dir = not os.path.exists(output_dir)
    if mk_output_dir:
        os.makedirs(output_dir, exist_ok=True)
    return mk_output_dir, output_dir


def ensure_output_file_dir_existence(output_file):
    output_dir = os.path.split(output_file)[0]
    return ensure_output_dir_existence(output_dir)
