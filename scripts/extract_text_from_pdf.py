
import pypdfium2 as pdfium
import sys


def extract_text(pdf_path, txt_path):
    """
    Extracts text from a PDF file and saves it to a text file.
    """
    import os

    os.makedirs(os.path.dirname(txt_path), exist_ok=True)

    try:
        doc = pdfium.PdfDocument(pdf_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            for i in range(len(doc)):
                page = doc.get_page(i)
                textpage = page.get_textpage()
                text = textpage.get_text_range()
                f.write(text)
                f.write("\n\n--- Page {} ---\n\n".format(i+1))
        print(f"Successfully extracted text from {pdf_path} to {txt_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_text.py <pdf_path> <txt_path>")
        sys.exit(1)
    pdf_path = sys.argv[1]
    txt_path = sys.argv[2]
    extract_text(pdf_path, txt_path)
