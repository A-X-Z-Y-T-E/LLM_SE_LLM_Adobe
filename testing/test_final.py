#!/usr/bin/env python3
import sys
sys.path.append('.')
from final_extractor import FinalPDFExtractor
import json
import os

def test_final_version():
    """Test the final extractor on all sample files"""
    
    sample_dir = r"d:\VS_CODE\Adobe\PS\Adobe-India-Hackathon25\Challenge_1a\sample_dataset\pdfs"
    expected_dir = r"d:\VS_CODE\Adobe\PS\Adobe-India-Hackathon25\Challenge_1a\sample_dataset\outputs"
    
    extractor = FinalPDFExtractor()
    
    results = {}
    
    for pdf_file in os.listdir(sample_dir):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(sample_dir, pdf_file)
            expected_path = os.path.join(expected_dir, pdf_file.replace('.pdf', '.json'))
            
            print(f"\n=== Testing {pdf_file} ===")
            
            try:
                # Process with our extractor
                result = extractor.process_pdf(pdf_path)
                
                # Load expected
                with open(expected_path, 'r', encoding='utf-8') as f:
                    expected = json.load(f)
                
                # Compare titles (handle empty titles)
                our_title = result['title'].strip()
                expected_title = expected['title'].strip()
                title_match = our_title == expected_title
                
                outline_match = len(result['outline']) == len(expected['outline'])
                
                print(f"Title match: {title_match}")
                print(f"  Our: '{our_title}'")
                print(f"  Expected: '{expected_title}'")
                print(f"Outline count match: {outline_match}")
                print(f"  Our: {len(result['outline'])}")
                print(f"  Expected: {len(expected['outline'])}")
                
                results[pdf_file] = {
                    'title_match': title_match,
                    'outline_match': outline_match,
                    'our_count': len(result['outline']),
                    'expected_count': len(expected['outline'])
                }
                
            except Exception as e:
                print(f"Error: {e}")
                results[pdf_file] = {'error': str(e)}
    
    print(f"\n=== FINAL SUMMARY ===")
    title_matches = sum(1 for r in results.values() if r.get('title_match', False))
    outline_matches = sum(1 for r in results.values() if r.get('outline_match', False))
    total = len(results)
    
    print(f"Title matches: {title_matches}/{total}")
    print(f"Outline matches: {outline_matches}/{total}")
    print(f"Overall accuracy: {(title_matches + outline_matches)}/{total * 2}")

if __name__ == "__main__":
    test_final_version()
