from fpdf import FPDF
import os

def export_to_txt(text, output_dir, filename="result.txt"):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as file:
        file.write(text)
    return filepath

def export_to_pdf(text, output_dir, filename="result.pdf"):
    filepath = os.path.join(output_dir, filename)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    pdf.output(filepath)
    return filepath
