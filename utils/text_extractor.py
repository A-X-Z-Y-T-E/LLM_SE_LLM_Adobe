"""
Simple standalone script to extract text from PDFs.
Minimal utility for basic PDF text extraction.
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any
import json


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text blocks from PDF using PyMuPDF.
    
    Args:
        pdf_path (str): Path to PDF file
        
    Returns:
        List of text block dictionaries
    """
    blocks = []
    
    try:
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_dict = page.get_text("dict")
            
            for block in text_dict.get("blocks", []):
                if "lines" in block:  # Text block
                    for line in block["lines"]:
                        for span in line.get("spans", []):
                            block_data = {
                                "page_number": page_num + 1,
                                "text": span.get("text", "").strip(),
                                "bbox": span.get("bbox", [0, 0, 0, 0]),
                                "font_size": span.get("size", 12.0),
                                "font_name": span.get("font", ""),
                                "flags": span.get("flags", 0)
                            }
                            
                            if block_data["text"]:  # Only include non-empty text
                                blocks.append(block_data)
        
        doc.close()
        
    except Exception as e:
        print(f"Error extracting from {pdf_path}: {e}")
        
    return blocks


def main():
    """Simple extraction example."""
    pdf_path = input("Enter PDF path: ").strip()
    
    if not Path(pdf_path).exists():
        print("PDF file not found!")
        return
    
    blocks = extract_text_from_pdf(pdf_path)
    
    output_path = Path(pdf_path).stem + "_extracted.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(blocks, f, indent=2, ensure_ascii=False)
    
    print(f"Extracted {len(blocks)} text blocks to {output_path}")


if __name__ == "__main__":
    main()
