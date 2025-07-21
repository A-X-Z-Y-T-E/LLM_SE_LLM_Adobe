#!/usr/bin/env python3
"""
Test script to process the sample PDF and compare output
"""

import sys
import os
sys.path.append('.')

from code1 import HackathonPDFExtractor
import json

def test_sample_pdf():
    """Test the sample PDF file"""
    extractor = HackathonPDFExtractor()
    
    sample_pdf = r"d:\VS_CODE\Adobe\PS\Adobe-India-Hackathon25\Challenge_1a\sample_dataset\pdfs\file01.pdf"
    
    if os.path.exists(sample_pdf):
        print(f"Processing: {sample_pdf}")
        result = extractor.process_pdf(sample_pdf)
        
        print("Current Output:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Compare with expected
        expected_file = r"d:\VS_CODE\Adobe\PS\Adobe-India-Hackathon25\Challenge_1a\sample_dataset\outputs\file01.json"
        if os.path.exists(expected_file):
            with open(expected_file, 'r', encoding='utf-8') as f:
                expected = json.load(f)
            
            print("\nExpected Output:")
            print(json.dumps(expected, indent=2, ensure_ascii=False))
            
            print(f"\nTitle Match: {result['title'].strip() == expected['title'].strip()}")
            print(f"Outline Length - Current: {len(result['outline'])}, Expected: {len(expected['outline'])}")
    else:
        print(f"Sample PDF not found: {sample_pdf}")

if __name__ == "__main__":
    test_sample_pdf()
