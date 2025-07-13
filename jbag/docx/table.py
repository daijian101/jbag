from typing import Union

from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from docx.table import _Cell, Table


def cm_to_dxa(cm):
    """
    Convert centimeters to twentieths of a point (dxa).
    """
    return int(cm * 1440 / 2.54)


def set_cell(cell: _Cell, text, font="Times New Roman", font_size=10, bold=False, italic=False,
             underline=False, cell_margins: list[float] = None):
    """
    Set cell text and properties.
    Args:
        cell (docx.table._Cell):
        text (str):
        font (str, optional, default="Times New Roman"):
        font_size (int, optional, default=10):
        bold (bool, optional, default=False):
        italic (bool, optional, default=False):
        underline (bool, optional, default=False):
        cell_margins (list, optional, default=[]): The cell margins should be in the format of [left, right, top, bottom] in the unit of centimeter.

    Returns:

    """
    cell.text = text
    run = cell.paragraphs[0].runs[0]
    run.font.name = font
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.underline = underline
    if cell_margins:
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()

        tcMar = tcPr.find(qn("w:tcMar"))
        if tcMar is None:
            tcMar = OxmlElement("w:tcMar")
            tcPr.append(tcMar)

        margin_keys = ["left", "right", "top", "bottom"]
        for i, value in enumerate(cell_margins):
            if value is None:
                continue

            tag = "w:{}".format(margin_keys[i])
            margin_element = tcMar.find(qn(tag))
            if margin_element is None:
                margin_element = OxmlElement(tag)
                tcMar.append(margin_element)
            value = cm_to_dxa(value)
            margin_element.set(qn("w:w"), str(value))
            margin_element.set(qn("w:type"), "dxa")


def set_cell_border(cell: _Cell,
                    borders: Union[str, list[str], tuple[float, ...]],
                    styles: Union[str, list[str], tuple[float, ...]] = "single",
                    sizes: Union[float, list[float], tuple[float, ...]] = 4,
                    colors: Union[str, list[str], tuple[str, ...]] = "auto"):
    """

    Args:
        cell:
        borders: Option: top, bottom, left, right.
        styles:
        sizes: 4 is equal to 0.5 pt, 8 for 1 pt.

    Returns:

    """
    if isinstance(borders, str):
        borders = [borders]

    valid_borders = ["top", "bottom", "left", "right"]
    for border in borders:
        if border not in valid_borders:
            raise ValueError(f"Invalid border type: {border}. Supported are : {', '.join(valid_borders)}")

    if isinstance(styles, str):
        styles = [styles] * len(borders)
    if not isinstance(sizes, (list, tuple)):
        sizes = [sizes] * len(borders)
    if isinstance(colors, str):
        colors = [colors] * len(borders)

    assert len(borders) == len(styles) == len(sizes) == len(colors)

    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")

    for border, style, size, color in zip(borders, styles, sizes, colors):
        border_element = OxmlElement(f"w:{border}")
        border_element.set(qn("w:val"), style)
        border_element.set(qn("w:sz"), str(size))
        border_element.set(qn("w:color"), color)
        tcBorders.append(border_element)
    tcPr.append(tcBorders)


def set_three_line_border(table: Table,
                          outer_line_weight: Union[float, int] = 8,
                          inner_line_weight: Union[float, int] = 4):
    """
    Draw three line table borders.
    Args:
        table (docx.table.Table):
        outer_line_weight (float or int, optional, default=8): Line weight for top and bottom borders. Defaults are 1 pt.
        8 indicates 1 pt.
        inner_line_weight (float or int, optional, default=8): Inner line weight. Defaults are 0.5 pt. 4 indicates 0.5 pt.

    Returns:

    """
    first_row = table.rows[0]
    for cell in first_row.cells:
        set_cell_border(cell, borders=["top"], sizes=outer_line_weight)
        set_cell_border(cell, borders=["bottom"], sizes=inner_line_weight)

    last_row = table.rows[-1]
    for cell in last_row.cells:
        set_cell_border(cell, borders=["bottom"], sizes=outer_line_weight)
