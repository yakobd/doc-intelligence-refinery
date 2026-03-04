from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
import os

def test_docling():
    source = "data/raw/Consumer Price Index March 2025.pdf"
    converter = DocumentConverter()
    
    print(f"--- Running Docling on {os.path.basename(source)} ---")
    result = converter.convert(source)
    
    # Export to Markdown to see the structure
    md_output = result.document.export_to_markdown()
    
    print("\n--- Docling Markdown Output (Snippet) ---")
    print(md_output[:1000]) # First 1000 chars
    
    # Check for tables specifically
    table_count = len(result.document.tables)
    print(f"\n--- Analysis ---")
    print(f"Tables detected: {table_count}")
    
    with open("research/docling_output.md", "w", encoding="utf-8") as f:
        f.write(md_output)
    print("\nFull output saved to research/docling_output.md")

if __name__ == "__main__":
    test_docling()