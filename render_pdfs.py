"""Render the InferTutor capstone Markdown files to PDFs."""

from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF


ROOT = Path(__file__).parent


class CapstonePDF(FPDF):
    def __init__(self, title: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.title_text = title
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(16, 18, 16)
        self.add_font("Body", "", "/Library/Fonts/Arial Unicode.ttf")
        self.add_font("Mono", "", "/System/Library/Fonts/SFNSMono.ttf")
        self.alias_nb_pages()

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Body", "", 8)
        self.set_text_color(110, 110, 110)
        self.cell(0, 6, self.title_text, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-13)
        self.set_font("Body", "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"{self.page_no()} / {{nb}}", align="C")


def clean_inline(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = text.replace("**", "")
    text = text.replace("*", "")
    return text


def add_cover(pdf: CapstonePDF, title: str, subtitle: str):
    pdf.add_page()
    pdf.set_fill_color(20, 31, 45)
    pdf.rect(0, 0, 210, 297, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Body", "", 28)
    pdf.set_xy(18, 72)
    pdf.multi_cell(174, 13, title, align="C")
    pdf.ln(8)
    pdf.set_font("Body", "", 13)
    pdf.set_text_color(212, 220, 230)
    pdf.multi_cell(174, 7, subtitle, align="C")
    pdf.set_y(245)
    pdf.set_font("Body", "", 10)
    pdf.set_text_color(180, 190, 205)
    pdf.cell(0, 8, "Vizuara AI Labs - Inference Engineering Workshop", align="C")


def add_paragraph(pdf: CapstonePDF, text: str):
    if not text.strip():
        return
    pdf.set_font("Body", "", 10.5)
    pdf.set_text_color(35, 35, 35)
    pdf.multi_cell(0, 5.6, clean_inline(text), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1.5)


def add_bullet(pdf: CapstonePDF, text: str):
    pdf.set_font("Body", "", 10.2)
    pdf.set_text_color(35, 35, 35)
    x = pdf.get_x()
    y = pdf.get_y()
    pdf.set_xy(x + 3, y)
    pdf.cell(3, 5.4, "-")
    pdf.set_xy(x + 8, y)
    pdf.multi_cell(0, 5.4, clean_inline(text), new_x="LMARGIN", new_y="NEXT")


def add_code(pdf: CapstonePDF, lines: list[str]):
    if not lines:
        return
    line_h = 4.5
    height = min(85, 6 + line_h * len(lines))
    if pdf.get_y() + height > 276:
        pdf.add_page()
    y = pdf.get_y() + 1
    pdf.set_fill_color(246, 248, 250)
    pdf.set_draw_color(215, 220, 228)
    pdf.rect(pdf.l_margin, y, pdf.w - pdf.l_margin - pdf.r_margin, height, "DF")
    pdf.set_xy(pdf.l_margin + 3, y + 3)
    pdf.set_font("Mono", "", 7.8)
    pdf.set_text_color(30, 40, 50)
    max_lines = int((height - 6) / line_h)
    for line in lines[:max_lines]:
        pdf.cell(0, line_h, line[:108], new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(pdf.l_margin + 3)
    pdf.set_y(y + height + 3)


def split_row(line: str) -> list[str]:
    return [clean_inline(c.strip()) for c in line.strip().strip("|").split("|")]


def add_table(pdf: CapstonePDF, table_lines: list[str]):
    rows = [split_row(line) for line in table_lines if "|" in line and "---" not in line]
    if not rows:
        return
    headers, body = rows[0], rows[1:]
    cols = len(headers)
    usable = pdf.w - pdf.l_margin - pdf.r_margin
    widths = [usable / cols] * cols

    if pdf.get_y() + 12 + 6 * len(body) > 276:
        pdf.add_page()
    pdf.set_font("Body", "", 7.6 if cols > 6 else 8.3)
    pdf.set_fill_color(37, 99, 235)
    pdf.set_text_color(255, 255, 255)
    for i, h in enumerate(headers):
        pdf.cell(widths[i], 7, h[:22], border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_text_color(35, 35, 35)
    fill = False
    for row in body:
        pdf.set_fill_color(245, 247, 250) if fill else pdf.set_fill_color(255, 255, 255)
        for i in range(cols):
            value = row[i] if i < len(row) else ""
            pdf.cell(widths[i], 6, value[:24], border=1, fill=True, align="C")
        pdf.ln()
        fill = not fill
    pdf.ln(3)


def add_heading(pdf: CapstonePDF, level: int, text: str):
    text = clean_inline(text)
    if level == 1:
        pdf.add_page()
        pdf.set_font("Body", "", 22)
        pdf.set_text_color(25, 45, 70)
        pdf.multi_cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(37, 99, 235)
        pdf.set_line_width(0.5)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(4)
    elif level == 2:
        if pdf.get_y() > 252:
            pdf.add_page()
        pdf.ln(3)
        pdf.set_font("Body", "", 15)
        pdf.set_text_color(37, 99, 235)
        pdf.multi_cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
    else:
        pdf.ln(2)
        pdf.set_font("Body", "", 12)
        pdf.set_text_color(45, 45, 45)
        pdf.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")


def render_markdown(md_path: Path, pdf_path: Path, title: str, subtitle: str):
    pdf = CapstonePDF(title)
    add_cover(pdf, title, subtitle)

    lines = md_path.read_text().splitlines()
    paragraph: list[str] = []
    code: list[str] = []
    table: list[str] = []
    in_code = False

    def flush_paragraph():
        nonlocal paragraph
        if paragraph:
            add_paragraph(pdf, " ".join(paragraph))
            paragraph = []

    def flush_table():
        nonlocal table
        if table:
            add_table(pdf, table)
            table = []

    for line in lines:
        if line.startswith("```"):
            if in_code:
                add_code(pdf, code)
                code = []
                in_code = False
            else:
                flush_paragraph()
                flush_table()
                in_code = True
            continue

        if in_code:
            code.append(line)
            continue

        if line.startswith("#"):
            flush_paragraph()
            flush_table()
            level = len(line) - len(line.lstrip("#"))
            add_heading(pdf, level, line[level:].strip())
            continue

        if "|" in line and line.strip().startswith("|"):
            flush_paragraph()
            table.append(line)
            continue

        if line.strip().startswith("- "):
            flush_paragraph()
            flush_table()
            add_bullet(pdf, line.strip()[2:])
            continue

        if not line.strip():
            flush_paragraph()
            flush_table()
            continue

        paragraph.append(line.strip())

    flush_paragraph()
    flush_table()
    pdf.output(str(pdf_path))


def main():
    render_markdown(
        ROOT / "InferTutor_Arena_Capstone.md",
        ROOT / "InferTutor_Arena_Capstone.pdf",
        "InferTutor Arena",
        "Capstone Project for the Inference Engineering Workshop",
    )
    render_markdown(
        ROOT / "Modal_vLLM_Runbook.md",
        ROOT / "Modal_vLLM_Runbook.pdf",
        "Modal and vLLM Runbook",
        "Companion Guide for InferTutor Arena",
    )
    print("Rendered PDFs:")
    print(ROOT / "InferTutor_Arena_Capstone.pdf")
    print(ROOT / "Modal_vLLM_Runbook.pdf")


if __name__ == "__main__":
    main()

