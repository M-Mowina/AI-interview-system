# src/pdf_utils.py
from fpdf import FPDF
import os

def generate_pdf(report_text: str, output_dir: str = "./generated_reports", filename: str = "report.pdf") -> str:
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, filename)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in report_text.split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf.output(pdf_path)
    return pdf_path 