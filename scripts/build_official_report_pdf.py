#!/usr/bin/env python3
"""Build the official M1/M2 report PDF from the reviewed Markdown source."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Frame,
    KeepTogether,
    LongTable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)


NAVY = colors.HexColor("#132A3A")
TEAL = colors.HexColor("#0F766E")
TEAL_DARK = colors.HexColor("#115E59")
MINT = colors.HexColor("#E8F5F2")
BLUE_PALE = colors.HexColor("#EAF1F8")
AMBER = colors.HexColor("#C97A10")
AMBER_PALE = colors.HexColor("#FFF4DE")
INK = colors.HexColor("#24313A")
MUTED = colors.HexColor("#5D6B75")
LINE = colors.HexColor("#CAD4DA")
PAPER = colors.HexColor("#F7F9FA")
WHITE = colors.white


def register_fonts() -> tuple[str, str, str]:
    candidates = [
        (
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
        ),
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
        ),
    ]
    for regular, bold, italic in candidates:
        if all(Path(p).exists() for p in (regular, bold, italic)):
            pdfmetrics.registerFont(TTFont("ReportSans", regular))
            pdfmetrics.registerFont(TTFont("ReportSans-Bold", bold))
            pdfmetrics.registerFont(TTFont("ReportSans-Italic", italic))
            pdfmetrics.registerFontFamily(
                "ReportSans",
                normal="ReportSans",
                bold="ReportSans-Bold",
                italic="ReportSans-Italic",
                boldItalic="ReportSans-Bold",
            )
            return "ReportSans", "ReportSans-Bold", "ReportSans-Italic"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


FONT, FONT_BOLD, FONT_ITALIC = register_fonts()


def inline_markup(value: str) -> str:
    escaped = html.escape(value.strip(), quote=True)
    escaped = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        r'<link href="\2" color="#0F766E"><u>\1</u></link>',
        escaped,
    )
    escaped = re.sub(
        r"&lt;(https?://[^&]+)&gt;",
        r'<link href="\1" color="#0F766E"><u>\1</u></link>',
        escaped,
    )
    escaped = re.sub(r"`([^`]+)`", r'<font name="Courier" color="#334155">\1</font>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", escaped)
    return escaped


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "ReportBody",
            parent=styles["BodyText"],
            fontName=FONT,
            fontSize=9.3,
            leading=13.2,
            textColor=INK,
            spaceAfter=5,
            allowWidows=0,
            allowOrphans=0,
        )
    )
    styles.add(
        ParagraphStyle(
            "ReportH2",
            parent=styles["Heading1"],
            fontName=FONT_BOLD,
            fontSize=17,
            leading=20,
            textColor=NAVY,
            spaceBefore=2,
            spaceAfter=9,
            keepWithNext=False,
        )
    )
    styles.add(
        ParagraphStyle(
            "ReportH3",
            parent=styles["Heading2"],
            fontName=FONT_BOLD,
            fontSize=12.2,
            leading=15,
            textColor=TEAL_DARK,
            spaceBefore=9,
            spaceAfter=4,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            "ReportH4",
            parent=styles["Heading3"],
            fontName=FONT_BOLD,
            fontSize=10.2,
            leading=13,
            textColor=NAVY,
            spaceBefore=7,
            spaceAfter=3,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            "ReportBullet",
            parent=styles["ReportBody"],
            leftIndent=13,
            firstLineIndent=-8,
            bulletIndent=0,
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            "ReportNumber",
            parent=styles["ReportBody"],
            leftIndent=17,
            firstLineIndent=-13,
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            "TableHead",
            fontName=FONT_BOLD,
            fontSize=7.5,
            leading=9.2,
            textColor=WHITE,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            "TableCell",
            fontName=FONT,
            fontSize=7.25,
            leading=9.1,
            textColor=INK,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            "CodeBlock",
            fontName="Courier",
            fontSize=7.4,
            leading=10,
            leftIndent=6,
            rightIndent=6,
            textColor=NAVY,
            backColor=BLUE_PALE,
            borderPadding=7,
            borderColor=colors.HexColor("#B9CDDF"),
            borderWidth=0.5,
            borderRadius=4,
            spaceBefore=5,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            "CoverKicker",
            fontName=FONT_BOLD,
            fontSize=10,
            leading=12,
            textColor=TEAL,
            alignment=TA_CENTER,
            tracking=1.2,
        )
    )
    styles.add(
        ParagraphStyle(
            "CoverTitle",
            fontName=FONT_BOLD,
            fontSize=28,
            leading=32,
            textColor=NAVY,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            "CoverSub",
            fontName=FONT,
            fontSize=13,
            leading=17,
            textColor=MUTED,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            "CardTitle",
            fontName=FONT_BOLD,
            fontSize=11,
            leading=13,
            textColor=NAVY,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            "CardBody",
            fontName=FONT,
            fontSize=8.3,
            leading=11,
            textColor=INK,
            alignment=TA_CENTER,
        )
    )
    return styles


STYLES = make_styles()


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="body",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(PageTemplate(id="main", frames=frame, onPage=self._decorate_page))

    def _decorate_page(self, canvas, doc):
        width, height = A4
        canvas.saveState()
        if doc.page > 1:
            canvas.setStrokeColor(LINE)
            canvas.setLineWidth(0.5)
            canvas.line(doc.leftMargin, height - 16 * mm, width - doc.rightMargin, height - 16 * mm)
            canvas.setFont(FONT_BOLD, 7.2)
            canvas.setFillColor(TEAL_DARK)
            canvas.drawString(doc.leftMargin, height - 12.5 * mm, "GITHUB ISSUE AUTO-LABELER & DUPLICATE FINDER")
            canvas.setFont(FONT, 7.2)
            canvas.setFillColor(MUTED)
            canvas.drawRightString(width - doc.rightMargin, height - 12.5 * mm, "OFFICIAL M1 / M2 REPORT")
            canvas.setStrokeColor(LINE)
            canvas.line(doc.leftMargin, 14 * mm, width - doc.rightMargin, 14 * mm)
            canvas.setFont(FONT, 7.2)
            canvas.drawString(doc.leftMargin, 9.7 * mm, "NLP for Software Engineering • 29 October 2026")
            canvas.drawRightString(width - doc.rightMargin, 9.7 * mm, f"{doc.page}")
        canvas.restoreState()


def cover_story():
    story = [Spacer(1, 19 * mm)]
    story.append(Paragraph("NATURAL LANGUAGE PROCESSING • INDUSTRY TRACK", STYLES["CoverKicker"]))
    story.append(Spacer(1, 11 * mm))
    story.append(Paragraph("GitHub Issue Auto-Labeler<br/>&amp; Duplicate Finder", STYLES["CoverTitle"]))
    story.append(Spacer(1, 7 * mm))
    story.append(Paragraph("Official Milestone 1 and Milestone 2 Report", STYLES["CoverSub"]))
    story.append(Spacer(1, 15 * mm))

    card_data = [
        [
            Paragraph("M1 • COMPLETE", STYLES["CardTitle"]),
            Paragraph("M2 • COMPLETE", STYLES["CardTitle"]),
        ],
        [
            Paragraph("Five-paper comparison<br/>Anchor selection<br/>10-point reproducibility audit", STYLES["CardBody"]),
            Paragraph("Real fastText rerun<br/>3,000 public issues<br/>Full metrics and gap evidence", STYLES["CardBody"]),
        ],
    ]
    cards = Table(card_data, colWidths=[79 * mm, 79 * mm], rowHeights=[13 * mm, 27 * mm])
    cards.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), MINT),
                ("BACKGROUND", (1, 0), (1, -1), BLUE_PALE),
                ("BOX", (0, 0), (0, -1), 0.8, colors.HexColor("#A7D8D0")),
                ("BOX", (1, 0), (1, -1), 0.8, colors.HexColor("#B9CDDF")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(cards)
    story.append(Spacer(1, 13 * mm))

    result = Table(
        [
            [Paragraph("REPRODUCTION RESULT", STYLES["CardTitle"])],
            [
                Paragraph(
                    '<font size="18"><b>0.71839 → 0.71595</b></font><br/>'
                    '<font color="#5D6B75">absolute F1 gap −0.00244 (−0.24 percentage points)</font>',
                    STYLES["CardBody"],
                )
            ],
        ],
        colWidths=[158 * mm],
        rowHeights=[12 * mm, 28 * mm],
    )
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), AMBER_PALE),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#E8C47D")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(result)
    story.append(Spacer(1, 18 * mm))
    story.append(
        Paragraph(
            "Aligned to the student-supplied <i>NLP Research Project Guideline</i>. "
            "M3–M6 are deliberately not claimed complete.",
            STYLES["CoverSub"],
        )
    )
    story.append(Spacer(1, 7 * mm))
    story.append(Paragraph("Prepared 29 October 2026", STYLES["CoverKicker"]))
    story.append(PageBreak())
    return story


def parse_table(lines: list[str], available_width: float):
    rows = []
    for line in lines:
        stripped = line.strip().strip("|")
        rows.append([cell.strip() for cell in stripped.split("|")])
    if len(rows) > 1 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in rows[1]):
        rows.pop(1)
    column_count = max(len(row) for row in rows)
    rows = [row + [""] * (column_count - len(row)) for row in rows]

    scores = []
    for col in range(column_count):
        lengths = [min(95, max(8, len(re.sub(r"[*`]", "", row[col])))) for row in rows[:8]]
        scores.append(max(12, sum(lengths) / max(1, len(lengths))))
    if column_count >= 5:
        scores = [min(score, 35) for score in scores]
    total = sum(scores)
    widths = [available_width * score / total for score in scores]
    min_width = 18 * mm if column_count <= 4 else 12 * mm
    widths = [max(min_width, width) for width in widths]
    scale = available_width / sum(widths)
    widths = [width * scale for width in widths]

    wrapped = []
    for row_index, row in enumerate(rows):
        style = STYLES["TableHead"] if row_index == 0 else STYLES["TableCell"]
        wrapped.append([Paragraph(inline_markup(cell), style) for cell in row])

    table = LongTable(
        wrapped,
        colWidths=widths,
        repeatRows=1,
        hAlign="LEFT",
        splitByRow=1,
    )
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 4.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
    ]
    for row_index in range(1, len(wrapped)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), PAPER))
    table.setStyle(TableStyle(commands))
    return [Spacer(1, 3), table, Spacer(1, 8)]


def markdown_story(markdown_text: str, available_width: float):
    lines = markdown_text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == "## Executive verdict"), 0)
    lines = lines[start:]
    story = []
    paragraph_buffer: list[str] = []
    in_code = False
    code_buffer: list[str] = []

    def flush_paragraph():
        nonlocal paragraph_buffer
        if paragraph_buffer:
            text = " ".join(piece.strip() for piece in paragraph_buffer).strip()
            if text:
                story.append(Paragraph(inline_markup(text), STYLES["ReportBody"]))
            paragraph_buffer = []

    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_paragraph()
            if in_code:
                story.append(Preformatted("\n".join(code_buffer), STYLES["CodeBlock"]))
                code_buffer = []
                in_code = False
            else:
                in_code = True
            index += 1
            continue
        if in_code:
            code_buffer.append(line.rstrip())
            index += 1
            continue
        if stripped.startswith("|"):
            flush_paragraph()
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            story.extend(parse_table(table_lines, available_width))
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[4:]), STYLES["ReportH3"]))
        elif stripped.startswith("#### "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[5:]), STYLES["ReportH4"]))
        elif stripped.startswith("## "):
            flush_paragraph()
            title = stripped[3:]
            if re.match(r"\d+\.", title) and story:
                story.append(CondPageBreak(100 * mm))
            story.append(Paragraph(inline_markup(title), STYLES["ReportH2"]))
            story.append(Spacer(1, 1.5 * mm))
        elif stripped in {"---", "***"}:
            flush_paragraph()
            story.append(Spacer(1, 2 * mm))
        elif re.match(r"^-\s+", stripped):
            flush_paragraph()
            content = re.sub(r"^-\s+", "", stripped)
            story.append(Paragraph(f"•&nbsp;&nbsp;{inline_markup(content)}", STYLES["ReportBullet"]))
        elif re.match(r"^\d+\.\s+", stripped):
            flush_paragraph()
            match = re.match(r"^(\d+)\.\s+(.*)", stripped)
            story.append(
                Paragraph(
                    f'<font color="#0F766E"><b>{match.group(1)}.</b></font>&nbsp;&nbsp;{inline_markup(match.group(2))}',
                    STYLES["ReportNumber"],
                )
            )
        elif not stripped:
            flush_paragraph()
        else:
            paragraph_buffer.append(stripped)
        index += 1

    flush_paragraph()
    if code_buffer:
        story.append(Preformatted("\n".join(code_buffer), STYLES["CodeBlock"]))
    return story


def build_pdf(markdown_path: Path, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = ReportDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=21 * mm,
        bottomMargin=19 * mm,
        title="GitHub Issue Auto-Labeler & Duplicate Finder — Official M1/M2 Report",
        author="NLP Research Project",
        subject="Official Milestone 1 literature survey and Milestone 2 baseline reproduction",
    )
    story = cover_story()
    story.extend(markdown_story(markdown_path.read_text(encoding="utf-8"), doc.width))
    doc.build(story)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_pdf(args.input, args.output)


if __name__ == "__main__":
    main()
