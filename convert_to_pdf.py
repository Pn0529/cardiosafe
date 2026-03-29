import markdown2
import os
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'CardioSafe Mini Project Report', 0, 1, 'C')
        self.ln(10)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def convert_markdown_to_pdf():
    # Read the markdown file
    with open('Mini_Project_Report.md', 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown2.markdown(markdown_content, extras=['tables', 'fenced-code-blocks'])
    
    # Create PDF
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 10)
    
    # Simple text extraction (removing HTML tags for basic PDF)
    import re
    text_content = re.sub('<[^<]+?>', '', html_content)
    
    # Add content to PDF
    lines = text_content.split('\n')
    for line in lines:
        if line.strip():
            # Handle different text formatting
            if line.startswith('# '):
                pdf.set_font('Arial', 'B', 16)
                pdf.multi_cell(0, 10, line[2:].strip())
                pdf.set_font('Arial', '', 10)
            elif line.startswith('## '):
                pdf.set_font('Arial', 'B', 14)
                pdf.multi_cell(0, 8, line[3:].strip())
                pdf.set_font('Arial', '', 10)
            elif line.startswith('### '):
                pdf.set_font('Arial', 'B', 12)
                pdf.multi_cell(0, 6, line[4:].strip())
                pdf.set_font('Arial', '', 10)
            elif '**' in line:
                # Handle bold text
                pdf.set_font('Arial', 'B', 10)
                pdf.multi_cell(0, 5, line.replace('**', ''))
                pdf.set_font('Arial', '', 10)
            else:
                pdf.multi_cell(0, 5, line.strip())
        else:
            pdf.ln(3)
    
    # Save PDF
    pdf.output('CardioSafe_Mini_Project_Report.pdf')
    print("PDF created successfully: CardioSafe_Mini_Project_Report.pdf")

if __name__ == "__main__":
    convert_markdown_to_pdf()
