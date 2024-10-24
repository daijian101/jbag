from docx.shared import Pt
from docx.text.paragraph import Paragraph


def add_text(paragraph: Paragraph, text: str, font: str = 'Times New Roman', size: int = 10, bold=False, italic=False,
             underline=False, subscript=False, superscript=False):
    run = paragraph.add_run()
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.underline = underline
    run.font.subscript = subscript
    run.font.superscript = superscript
    run.text = text
