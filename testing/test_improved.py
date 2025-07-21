#!/usr/bin/env python3
"""
Test the improved PDF extractor
"""

import sys
import os
import json
from pathlib import Path

# Add current directory to path
sys.path.append('.')

try:
    from code1 import HackathonPDFExtractor
    
    def test_sample_files():
        """Test all sample files"""
        sample_dir = Path(r"d:\VS_CODE\Adobe\PS\Adobe-India-Hackathon25\Challenge_1a\sample_dataset\pdfs")
        expected_dir = Path(r"d:\VS_CODE\Adobe\PS\Adobe-India-Hackathon25\Challenge_1a\sample_dataset\outputs")
        test_output_dir = Path(r"d:\VS_CODE\Adobe\test_results")
        
        # Create test output directory
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        extractor = HackathonPDFExtractor()
        
        # Test each sample file
        for pdf_file in sample_dir.glob("*.pdf"):
            print(f"\n=== Testing {pdf_file.name} ===")
            
            # Process with our extractor
            result = extractor.process_pdf(str(pdf_file))
            
            # Save our result
            our_output = test_output_dir / f"{pdf_file.stem}.json"
            with open(our_output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            
            # Load expected result
            expected_file = expected_dir / f"{pdf_file.stem}.json"
            if expected_file.exists():
                with open(expected_file, 'r', encoding='utf-8') as f:
                    expected = json.load(f)
                
                print(f"Our Title: '{result['title']}'")
                print(f"Expected Title: '{expected['title']}'")
                print(f"Our Outline Count: {len(result['outline'])}")
                print(f"Expected Outline Count: {len(expected['outline'])}")
                
                # Show our outline
                if result['outline']:
                    print("Our Outline:")
                    for item in result['outline'][:5]:  # First 5 items
                        print(f"  {item['level']}: {item['text'][:50]}... (page {item['page']})")
                
                # Show expected outline
                if expected['outline']:
                    print("Expected Outline:")
                    for item in expected['outline'][:5]:  # First 5 items
                        print(f"  {item['level']}: {item['text'][:50]}... (page {item['page']})")
                
            else:
                print(f"Expected file not found: {expected_file}")
                print(f"Our result: {result}")
    
    if __name__ == "__main__":
        test_sample_files()
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
