from typing import Type

from docx import Document
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT


def add_paragraph(doc: Type[Document], text=None, style=None,
                  alignment: Type[WD_PARAGRAPH_ALIGNMENT] = WD_PARAGRAPH_ALIGNMENT.JUSTIFY):
    if text is None:
        text = ''
    p = doc.add_paragraph(text=text, style=style)
    p.alignment = alignment
