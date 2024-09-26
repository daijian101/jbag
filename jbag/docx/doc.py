import os.path

from docx import Document


def import_style_from_template(doc: Document, template_doc_file):
    if not os.path.exists(template_doc_file):
        raise FileNotFoundError(template_doc_file)

    template_doc = Document(template_doc_file)

    for style in template_doc.styles:
        doc.styles.add_style(style.style_id, style.type)
