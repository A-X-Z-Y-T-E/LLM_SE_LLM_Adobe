"""
Simple standalone script to extract text from PDFs.
Minimal utility for basic PDF text extraction with robust error handling.
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any
import json


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text blocks from PDF using PyMuPDF with robust error handling.
    
    Args:
        pdf_path (str): Path to PDF file
        
    Returns:
        List of text block dictionaries
    """
    blocks = []
    doc = None
    
    try:
        doc = fitz.open(pdf_path)
        
        # Get page count safely
        try:
            page_count = doc.page_count
        except:
            try:
                page_count = len(doc)
            except:
                # Try to count pages manually
                page_count = 0
                while True:
                    try:
                        doc.load_page(page_count)
                        page_count += 1
                    except:
                        break
        
        print(f"Processing {page_count} pages...")
        
        for page_num in range(page_count):
            try:
                page = doc.load_page(page_num)
                
                # Try detailed extraction first
                try:
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
                
                except Exception:
                    # Fallback to simple text blocks
                    text_blocks = page.get_text("blocks")
                    for block in text_blocks:
                        if len(block) >= 5:
                            text = block[4].strip()
                            if text:
                                block_data = {
                                    "page_number": page_num + 1,
                                    "text": text,
                                    "bbox": [block[0], block[1], block[2], block[3]],
                                    "font_size": 12.0,
                                    "font_name": "",
                                    "flags": 0
                                }
                                blocks.append(block_data)
                
            except Exception as page_error:
                print(f"Error processing page {page_num + 1}: {page_error}")
                continue
        
    except Exception as e:
        print(f"Error extracting from {pdf_path}: {e}")
        
        # Last resort: try simple text extraction
        try:
            if doc is None:
                doc = fitz.open(pdf_path)
            
            page = doc.load_page(0)
            simple_text = page.get_text()
            
            if simple_text.strip():
                paragraphs = [p.strip() for p in simple_text.split('\n\n') if p.strip()]
                for idx, para in enumerate(paragraphs):
                    blocks.append({
                        "page_number": 1,
                        "text": para,
                        "bbox": [0, idx * 20, 500, (idx + 1) * 20],
                        "font_size": 12.0,
                        "font_name": "",
                        "flags": 0
                    })
                
                print(f"Used simple extraction: {len(blocks)} paragraphs")
        
        except Exception as final_error:
            print(f"All extraction methods failed: {final_error}")
    
    finally:
        if doc:
            try:
                doc.close()
            except:
                pass
        
    return blocks


def main():
    """Simple extraction example with error handling."""
    pdf_path = input("Enter PDF path: ").strip()
    
    if not Path(pdf_path).exists():
        print("PDF file not found!")
        return
    
    print(f"Extracting text from: {pdf_path}")
    blocks = extract_text_from_pdf(pdf_path)
    
    if blocks:
        output_path = Path(pdf_path).stem + "_extracted.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(blocks, f, indent=2, ensure_ascii=False)
        
        print(f"Extracted {len(blocks)} text blocks to {output_path}")
    else:
        print("No text could be extracted from the PDF")


if __name__ == "__main__":
    main()
