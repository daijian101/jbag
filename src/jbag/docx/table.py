from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.shared import OxmlElement, qn
from docx.shared import Cm, Pt
from docx.table import _Cell, Table

AFTER_TBLW = ("w:jc", "w:tblCellSpacing", "w:tblInd", "w:tblBorders", "w:shd",
              "w:tblLayout", "w:tblCellMar", "w:tblLook")


def cm_to_dxa(cm):
    """
    Convert centimeters to twentieths of a point (dxa).
    """
    return int(cm * 1440 / 2.54)


def set_cell_margins(cell: _Cell, top: float = 0, left: float = 0.1, bottom: float = 0.1, right: float = 0):
    """
    Set cell margins in cm.
    """
    tcPr = cell._tc.get_or_add_tcPr()
    for e in tcPr.findall(qn("w:tcMar")):
        tcPr.remove(e)
    mar = OxmlElement("w:tcMar")
    for tag, v in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        el = OxmlElement("w:" + tag)
        el.set(qn("w:w"), str(int(cm_to_dxa(v))))
        el.set(qn("w:type"), "dxa")  # dxa = twips
        mar.append(el)
    tcPr.insert_element_before(mar, "w:textDirection", "w:tcFitText", "w:vAlign", "w:hideMark")


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


def set_cell_alignment(cell: _Cell, alignment: str, vertical_center: bool = True):
    for p in cell.paragraphs:
        p.alignment = WD_ALIGN_PARAGRAPH[alignment.upper()]

    if vertical_center:
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def align_table(table: Table,
                first_column_alignment: str = 'left',
                rest_column_alignment: str = 'center',
                vertical_center: bool = True):
    first_column_alignment = WD_ALIGN_PARAGRAPH[first_column_alignment.upper()]
    rest_column_alignment = WD_ALIGN_PARAGRAPH[rest_column_alignment.upper()]
    for row in table.rows:
        for i, cell in enumerate(row.cells):
            alignment = first_column_alignment if i == 0 else rest_column_alignment
            for p in cell.paragraphs:
                p.alignment = alignment
            if vertical_center:
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def set_table_paragraph_spacing(table: Table, before: float | None = None, after: float | None = None,
                                line_spacing: float | str | None = None):
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                if before is not None:
                    p.paragraph_format.space_before = before
                if after is not None:
                    p.paragraph_format.space_after = after
                if line_spacing is not None:
                    if isinstance(line_spacing, str):
                        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING[line_spacing.upper()]
                    else:
                        p.paragraph_format.line_spacing = Pt(line_spacing)


def set_table_cell_margins(table: Table, top: float = 0, left: float = 0.1, bottom: float = 0.1, right: float = 0):
    """
    Set cell margins for the table in cm.

    """
    tblPr = table._tbl.tblPr
    for e in tblPr.findall(qn("w:tblCellMar")):
        tblPr.remove(e)
    mar = OxmlElement("w:tblCellMar")
    for tag, v in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        el = OxmlElement("w:" + tag)
        el.set(qn("w:w"), str(int(cm_to_dxa(v))))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tblPr.insert_element_before(mar, "w:tblLook")


def _set_tblw(table, typ, val):
    tblPr = table._tbl.tblPr
    for e in tblPr.findall(qn("w:tblW")):
        tblPr.remove(e)
    w = OxmlElement("w:tblW")
    w.set(qn("w:type"), typ)
    w.set(qn("w:w"), str(val))
    tblPr.insert_element_before(w, *AFTER_TBLW)


def _set_layout(table, mode="autofit"):
    tblPr = table._tbl.tblPr
    for e in tblPr.findall(qn("w:tblLayout")):
        tblPr.remove(e)
    lay = OxmlElement("w:tblLayout")
    lay.set(qn("w:type"), mode)
    tblPr.insert_element_before(lay, "w:tblCellMar", "w:tblLook")


def _clear_widths(table):
    for row in table.rows:
        for cell in row.cells:
            tcPr = cell._tc.get_or_add_tcPr()
            for e in tcPr.findall(qn("w:tcW")):
                tcPr.remove(e)
    for gc in table._tbl.tblGrid.findall(qn("w:gridCol")):
        gc.set(qn("w:w"), "0")


def autofit(table, mode):
    """
    mode: 'contents' | 'window' | 'fill'
    """
    _set_layout(table, "autofit")
    if mode == "contents":
        _set_tblw(table, "auto", 0)
        _clear_widths(table)
    elif mode == "window":
        _set_tblw(table, "pct", 5000)
    elif mode == "fill":
        _set_tblw(table, "pct", 5000)
        _clear_widths(table)


def set_table_style(table: Table,
                    cell_margins: list[float] = [0, 0.1, 0.1, 0],
                    three_line_border: bool = True,
                    outer_border_line_size: float = 1,
                    inner_border_line_size: float = 0.5,
                    first_column_alignment: str = 'left',
                    rest_column_alignment: str = 'center',
                    vertical_center_alignment: bool = True,
                    paragraph_spacing_before: float = 0,
                    paragraph_spacing_after: float = 0,
                    paragraph_line_spacing: float | str = 'single',
                    fit_mode: str = 'fill'):
    """

    Args:
        table (Table):
        cell_margins (list):
        three_line_border (bool):
        outer_border_line_size (float): In pt.
        inner_border_line_size (float): In pt.
        first_column_alignment (str):
        rest_column_alignment (str):
        vertical_center_alignment (bool):
        paragraph_spacing_before (float):
        paragraph_spacing_after (float):
        paragraph_line_spacing (float or str):
        fit_mode (str):

    Returns:

    """
    assert len(cell_margins) == 4
    set_table_cell_margins(table, cell_margins[0], cell_margins[1], cell_margins[2], cell_margins[3])

    if three_line_border:
        set_three_line_border(table, outer_border_line_size, inner_border_line_size)

    align_table(table, first_column_alignment, rest_column_alignment, vertical_center_alignment)
    set_table_paragraph_spacing(table, paragraph_spacing_before, paragraph_spacing_after, paragraph_line_spacing)
    autofit(table, fit_mode)
