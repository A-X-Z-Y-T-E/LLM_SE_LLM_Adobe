#!/usr/bin/env python3
"""
PDF Labeling Tool - Generate labels for training data
Uses the current best extractor as baseline, with human review capability
"""

import sys
sys.path.append('.')
from code1 import HackathonPDFExtractor
import json
import os
from pathlib import Path

class PDFLabelingTool:
    def __init__(self):
        self.extractor = HackathonPDFExtractor()
        
    def generate_baseline_labels(self, pdf_dir, output_dir, review_mode=False):
        """Generate baseline labels for all PDFs in directory"""
        
        pdf_dir = Path(pdf_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files to label")
        
        for i, pdf_file in enumerate(pdf_files):
            print(f"Processing {i+1}/{len(pdf_files)}: {pdf_file.name}")
            
            try:
                # Generate baseline label
                result = self.extractor.process_pdf(str(pdf_file))
                
                # Save baseline label
                label_file = output_dir / f"{pdf_file.stem}.json"
                with open(label_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                if review_mode:
                    self.review_label(pdf_file, result)
                    
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
    
    def review_label(self, pdf_file, result):
        """Interactive review of generated label"""
        print(f"\n--- Review for {pdf_file.name} ---")
        print(f"Title: '{result['title']}'")
        print(f"Headings found: {len(result['outline'])}")
        
        if result['outline']:
            print("Outline:")
            for item in result['outline'][:5]:  # Show first 5
                print(f"  {item['level']}: {item['text']}")
            if len(result['outline']) > 5:
                print(f"  ... and {len(result['outline']) - 5} more")
        
        # Simple review prompt
        response = input("Accept (y), Skip (s), or Quit (q)? ").lower()
        if response == 'q':
            return False
        return response == 'y'

def main():
    """Main labeling function"""
    labeler = PDFLabelingTool()
    
    # Choose your dataset
    pdf_directory = input("Enter PDF directory path (or press Enter for default): ").strip()
    if not pdf_directory:
        pdf_directory = r"d:\VS_CODE\Adobe\dataset-of-pdf-files\versions\1\Pdf"
    
    output_directory = input("Enter output directory for labels (or press Enter for default): ").strip()
    if not output_directory:
        output_directory = r"d:\VS_CODE\Adobe\training_labels"
     
    print(f"PDF Directory: {pdf_directory}")
    print(f"Output Directory: {output_directory}")
    
    # Option for review mode
    review = input("Enable review mode? (y/n): ").lower() == 'y'
    
    labeler.generate_baseline_labels(pdf_directory, output_directory, review_mode=review)
    print("Labeling complete!")

if __name__ == "__main__":
    main()
