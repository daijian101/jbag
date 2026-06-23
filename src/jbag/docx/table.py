from docx.oxml.shared import OxmlElement, qn
from docx.shared import Cm
from docx.shared import Pt
from docx.table import _Cell, Table


def cm_to_dxa(cm):
    """
    Convert centimeters to twentieths of a point (dxa).
    """
    return int(cm * 1440 / 2.54)


def set_cell(cell: _Cell, text, font='Times New Roman', font_size=10, bold=False, italic=False,
             underline=False, cell_margins: list[float] = [0.1, 0.1, 0.0, 0.0]):
    """
    Set cell text and properties.
    Args:
        cell (docx.table._Cell):
        text (str):
        font (str, optional, default='Times New Roman'):
        font_size (int, optional, default=10):
        bold (bool, optional, default=False):
        italic (bool, optional, default=False):
        underline (bool, optional, default=False):
        cell_margins (list, optional, default=[]): the cell margins should be in the format of [left, right, top, bottom] in the unit of centimeter.

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

        tcMar = tcPr.find(qn('w:tcMar'))
        if tcMar is None:
            tcMar = OxmlElement('w:tcMar')
            tcPr.append(tcMar)

        margin_keys = ['left', 'right', 'top', 'bottom']
        for i, value in enumerate(cell_margins):
            if value is None:
                continue

            tag = 'w:{}'.format(margin_keys[i])
            margin_element = tcMar.find(qn(tag))
            if margin_element is None:
                margin_element = OxmlElement(tag)
                tcMar.append(margin_element)
            margin_element.set(qn('w:w'), str(int(Cm(value).twips)))
            margin_element.set(qn('w:type'), 'dxa')


def set_cell_border(cell: _Cell,
                    borders: str | list[str] | tuple[str, ...],
                    styles: str | list[str] | tuple[str, ...] = 'single',
                    sizes_pt: float | list[float] | tuple[float, ...] = 1,
                    colors: str | list[str] | tuple[str, ...] = 'auto'):
    """

    Args:
        cell:
        borders (Option): top, bottom, left, right.
        styles:
        sizes_pt: Default: 1 pt.

    Returns:

    """
    if isinstance(borders, str):
        borders = [borders]

    valid_borders = ['top', 'bottom', 'left', 'right']
    for border in borders:
        if border not in valid_borders:
            raise ValueError(f'Invalid border type: {border}. Supported are : {', '.join(valid_borders)}')

    if isinstance(styles, str):
        styles = [styles] * len(borders)
    if not isinstance(sizes_pt, (list, tuple)):
        sizes_pt = [sizes_pt] * len(borders)
    if isinstance(colors, str):
        colors = [colors] * len(borders)

    assert len(borders) == len(styles) == len(sizes_pt) == len(colors)

    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    sizes_emu = [round(each * 8) for each in sizes_pt]
    for border, style, size, color in zip(borders, styles, sizes_emu, colors):
        border_element = OxmlElement(f'w:{border}')
        border_element.set(qn('w:val'), style)
        border_element.set(qn('w:sz'), str(size))
        border_element.set(qn('w:color'), color)
        tcBorders.append(border_element)
    tcPr.append(tcBorders)


def set_three_line_border(table: Table,
                          outer_line_size_pt: float | int = 1,
                          inner_line_size_pt: float | int = 0.5):
    """
    Draw three line table borders.
    Args:
        table (docx.table.Table):
        outer_line_size_pt (float or int, optional, default=8): line weight for top and bottom borders. Default is 1 pt.
        inner_line_size_pt (float or int, optional, default=8): inner line weight. Default is 0.5 pt.

    Returns:

    """
    first_row = table.rows[0]
    for cell in first_row.cells:
        set_cell_border(cell, borders=['top'], sizes_pt=outer_line_size_pt)
        set_cell_border(cell, borders=['bottom'], sizes_pt=inner_line_size_pt)

    last_row = table.rows[-1]
    for cell in last_row.cells:
        set_cell_border(cell, borders=['bottom'], sizes_pt=outer_line_size_pt)


def set_column_widths(table, widths_cm):
    tbl = table._tbl
    tblPr = tbl.tblPr

    tblLayout = tblPr.find(qn('w:tblLayout'))
    if tblLayout is None:
        tblLayout = OxmlElement('w:tblLayout')
        tblPr.insert(0, tblLayout)
    tblLayout.set(qn('w:type'), 'fixed')

    tblGrid = tbl.find(qn('w:tblGrid'))
    if tblGrid is not None:
        for gridCol, w_cm in zip(tblGrid, widths_cm):
            gridCol.set(qn('w:w'), str(int(Cm(w_cm).twips)))

    for row in table.rows:
        for col_idx, w_cm in enumerate(widths_cm):
            if col_idx < len(row.cells):
                tc = row.cells[col_idx]._tc
                tcPr = tc.get_or_add_tcPr()
                tcW = tcPr.find(qn('w:tcW'))
                if tcW is None:
                    tcW = OxmlElement('w:tcW')
                    tcPr.insert(0, tcW)

                tcW.set(qn('w:w'), str(int(Cm(w_cm).twips)))
                tcW.set(qn('w:type'), 'dxa')
