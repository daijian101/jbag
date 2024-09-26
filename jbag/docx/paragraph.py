from typing import Type

from docx import Document
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT


def add_paragraph(doc: Type[Document],
                  text=None,
                  alignment: Type[WD_PARAGRAPH_ALIGNMENT] = WD_PARAGRAPH_ALIGNMENT.JUSTIFY,
                  style=None
                  ):
    if text is None:
        text = ''
    p = doc.add_paragraph(text=text, style=style)
    if alignment is not None:
        p.alignment = alignment
    return p
