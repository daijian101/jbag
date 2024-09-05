import os.path
from typing import Type, Union

from docx import Document
from docx.shared import Inches
from docx.text.paragraph import Paragraph


def add_image(image_file,
              doc: Union[Type[Document], None] = None,
              paragraph: Union[Type[Paragraph], None] = None,
              width: float = 5,
              center: bool = True):
    if not os.path.exists(image_file):
        raise FileNotFoundError(image_file)

    assert not all([doc is None, paragraph is None])

    if paragraph is None:
        paragraph = doc.add_paragraph()

    run = paragraph.add_run()
    run.add_picture(image_file, width=Inches(width))
    if center:
        paragraph.alignment = 1
