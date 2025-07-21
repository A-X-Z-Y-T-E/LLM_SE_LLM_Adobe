#!/usr/bin/env python3
"""
Test specific sample file
"""

import sys
import os
import json
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from code1 import HackathonPDFExtractor

def test_file01():
    """Test file01.pdf specifically"""
    
    # File paths
    pdf_path = r"d:\VS_CODE\Adobe\PS\Adobe-India-Hackathon25\Challenge_1a\sample_dataset\pdfs\file01.pdf"
    expected_path = r"d:\VS_CODE\Adobe\PS\Adobe-India-Hackathon25\Challenge_1a\sample_dataset\outputs\file01.json"
    
    print(f"Testing: {pdf_path}")
    print(f"PDF exists: {os.path.exists(pdf_path)}")
    print(f"Expected file exists: {os.path.exists(expected_path)}")
    
    if not os.path.exists(pdf_path):
        print("PDF file not found!")
        return
    
    # Process with our extractor
    extractor = HackathonPDFExtractor()
    try:
        result = extractor.process_pdf(pdf_path)
        
        print("\n=== OUR RESULT ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Load expected result
        if os.path.exists(expected_path):
            with open(expected_path, 'r', encoding='utf-8') as f:
                expected = json.load(f)
            
            print("\n=== EXPECTED RESULT ===")
            print(json.dumps(expected, indent=2, ensure_ascii=False))
            
            # Compare
            print("\n=== COMPARISON ===")
            print(f"Title match: {result['title'].strip() == expected['title'].strip()}")
            print(f"Our title: '{result['title']}'")
            print(f"Expected title: '{expected['title']}'")
            print(f"Our outline count: {len(result['outline'])}")
            print(f"Expected outline count: {len(expected['outline'])}")
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_file01()
