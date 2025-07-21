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
    
    # Save our result
    with open('our_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"Our result saved to our_result.json")
    print(f"Title: '{result['title']}'")
    print(f"Outline count: {len(result['outline'])}")
    
    if result['outline']:
        print("First few outline items:")
        for item in result['outline'][:3]:
            print(f"  {item['level']}: {item['text'][:50]}...")
    
    # Compare with expected
    expected = {"title": "Application form for grant of LTC advance  ", "outline": []}
    print(f"\nComparison:")
    print(f"Expected title: '{expected['title']}'")
    print(f"Expected outline count: {len(expected['outline'])}")
    print(f"Title match: {result['title'].strip() == expected['title'].strip()}")
    print(f"Outline match: {len(result['outline']) == len(expected['outline'])}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
