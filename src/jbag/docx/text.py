from docx.shared import Pt
from docx.table import _Cell
from docx.text.paragraph import Paragraph

FONT = 'Times New Roman'
FONT_SIZE = 12


def add_text(paragraph: Paragraph | _Cell, text: str, font=None, size=None, bold=False, italic=False,
             underline=False, subscript=False, superscript=False):
    if font is None:
        font = FONT
    if size is None:
        size = FONT_SIZE
    if isinstance(paragraph, _Cell):
        paragraph = paragraph.paragraphs[-1]
    run = paragraph.add_run()
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.underline = underline
    run.font.subscript = subscript
    run.font.superscript = superscript
    run.text = text


def write(paragraph: Paragraph | _Cell, text_parts: dict[str, dict | None]):
    for k, v in text_parts.items():
        if v is None:
            add_text(paragraph, k)
        else:
            add_text(paragraph, k, **v)
