#!/usr/bin/env python3
import sys
sys.path.append('.')
from code1 import HackathonPDFExtractor
import json

# Test our extractor
extractor = HackathonPDFExtractor()
pdf_path = r"d:\VS_CODE\Adobe\PS\Adobe-India-Hackathon25\Challenge_1a\sample_dataset\pdfs\file01.pdf"

print("Testing our extractor...")
try:
    result = extractor.process_pdf(pdf_path)
    print("Our result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # Compare with expected
    expected = {"title": "Application form for grant of LTC advance  ", "outline": []}
    print("\nExpected:")
    print(json.dumps(expected, indent=2))
    
    print(f"\nTitle match: {result['title'].strip() == expected['title'].strip()}")
    print(f"Outline match: {len(result['outline']) == len(expected['outline'])}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
