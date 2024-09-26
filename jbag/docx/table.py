from typing import Union

from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from docx.table import _Cell


def set_cell(cell: _Cell, text, font='Times New Roman', font_size=10, bold=False, italic=False,
             underline=False):
    cell.text = text
    run = cell.paragraphs[0].runs[0]
    run.font.name = font
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.underline = underline


def set_cell_border(cell: _Cell,
                    borders: Union[str, list[str], tuple[float, ...]],
                    styles: Union[str, list[str], tuple[float, ...]] = 'single',
                    sizes: Union[float, list[float], tuple[float, ...]] = 4,
                    colors: Union[str, list[str], tuple[str, ...]] = 'auto'):
    """

    Args:
        cell:
        borders: Option: top, bottom, left, right
        styles:
        sizes: 4 is equal to 0.5pt in word software

    Returns:

    """
    if isinstance(borders, str):
        borders = [borders]

    valid_borders = ['top', 'bottom', 'left', 'right']
    for border in borders:
        assert border in valid_borders, f'Invalid border: {border}'

    if isinstance(styles, str):
        styles = [styles] * len(borders)
    if not isinstance(sizes, (list, tuple)):
        sizes = [sizes] * len(borders)
    if isinstance(colors, str):
        colors = [colors] * len(borders)

    assert len(borders) == len(styles) == len(sizes) == len(colors)

    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')

    for border, style, size, color in zip(borders, styles, sizes, colors):
        border_element = OxmlElement(f'w:{border}')
        border_element.set(qn('w:val'), style)
        border_element.set(qn('w:sz'), str(size))
        border_element.set(qn('w:color'), color)
        tcBorders.append(border_element)
    tcPr.append(tcBorders)
