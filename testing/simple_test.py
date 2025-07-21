#!/usr/bin/env python3
import fitz
import json
import os

pdf_path = r"d:\VS_CODE\Adobe\PS\Adobe-India-Hackathon25\Challenge_1a\sample_dataset\pdfs\file01.pdf"

print(f"File exists: {os.path.exists(pdf_path)}")

try:
    doc = fitz.open(pdf_path)
    print(f"PDF opened successfully, pages: {len(doc)}")
    
    # Extract basic text
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        print(f"Page {page_num + 1} text length: {len(text)}")
        if page_num == 0:  # Show first page text
            print("First 500 chars:")
            print(text[:500])
    
    doc.close()
    
except Exception as e:
    print(f"Error: {e}")
