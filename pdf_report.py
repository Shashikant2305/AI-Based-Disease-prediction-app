from io import BytesIO
import textwrap

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from pathlib import Path


def create_pdf_bytes(title, lines):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    _, height = A4
    y = height - 60
    c.setFont("Helvetica-Bold", 18)
    for line in textwrap.wrap(str(title), width=72) or [""]:
        c.drawString(50, y, line)
        y -= 22
    y -= 35
    c.setFont("Helvetica", 10)
    for line in lines:
        for wrapped_line in textwrap.wrap(str(line), width=100) or [""]:
            if y < 60:
                c.showPage()
                y = height - 60
                c.setFont("Helvetica", 10)
            c.drawString(50, y, wrapped_line)
            y -= 16
    c.save()
    return buffer.getvalue()


def create_pdf(output_path, title, lines):
    Path(output_path).write_bytes(create_pdf_bytes(title, lines))
