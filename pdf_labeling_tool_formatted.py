#!/usr/bin/env python3
"""
PDF Labeling Tool - Generate labels for training data in specified format
Uses the current best extractor as baseline, with human review capability
Outputs JSON files with block-level annotations including bbox, font info, and labels

Supported Labels:
- TITLE: Document titles (large, prominent text early in document)
- HH1: Main headings (level 1)
- HH2: Sub-headings (level 2) 
- HH3: Sub-sub-headings (level 3)
- H4: Minor headings (level 4)
- BODY: Regular text content
"""

import sys
sys.path.append('.')
from code1 import HackathonPDFExtractor
import json
import os
from pathlib import Path
import fitz  # PyMuPDF
import re

class FormattedPDFLabelingTool:
    def __init__(self):
        self.extractor = HackathonPDFExtractor()
        
    def extract_detailed_blocks(self, pdf_path):
        """Extract all text blocks with detailed formatting information"""
        try:
            print(f"🔍 Opening PDF: {pdf_path}")
            doc = fitz.open(pdf_path)
            all_blocks = []
            pdf_id = Path(pdf_path).stem
            
            print(f"📖 PDF has {len(doc)} pages")
            
            for page_num in range(len(doc)):
                print(f"📄 Processing page {page_num + 1}/{len(doc)}")
                page = doc[page_num]
                
                # Use get_text("dict") for detailed formatting info
                blocks = page.get_text("dict")
                block_counter = 1
                page_blocks = 0
                
                for block in blocks["blocks"]:
                    if "lines" not in block:
                        continue
                        
                    for line in block["lines"]:
                        line_text = ""
                        max_font_size = 0
                        is_bold = False
                        font_name = ""
                        line_bbox = line["bbox"]
                        
                        # Combine spans in the same line
                        for span in line["spans"]:
                            line_text += span["text"]
                            if span["size"] > max_font_size:
                                max_font_size = span["size"]
                                font_name = span["font"]
                                is_bold = span["flags"] & 16  # Bold flag
                        
                        line_text = line_text.strip()
                        if len(line_text) > 2:  # Skip very short text
                            block_id = f"{pdf_id}_p{page_num + 1}_b{block_counter:03d}"
                            
                            block_data = {
                                "pdf_id": pdf_id,
                                "page_number": page_num + 1,
                                "block_id": block_id,
                                "text": line_text,
                                "bbox": [round(line_bbox[0], 1), round(line_bbox[1], 1), 
                                        round(line_bbox[2], 1), round(line_bbox[3], 1)],
                                "font_size": round(max_font_size, 1),
                                "font_name": font_name,
                                "label": "BODY"  # Default label, will be updated
                            }
                            
                            all_blocks.append(block_data)
                            block_counter += 1
                            page_blocks += 1
                
                print(f"   ✓ Extracted {page_blocks} blocks from page {page_num + 1}")
            
            doc.close()
            print(f"✅ Total blocks extracted: {len(all_blocks)}")
            return all_blocks
            
        except Exception as e:
            print(f"❌ Error extracting detailed blocks from {pdf_path}: {e}")
            print(f"🔍 Error type: {type(e).__name__}")
            import traceback
            print(f"📋 Traceback: {traceback.format_exc()}")
            return []
    
    def classify_block_labels(self, blocks):
        """Classify blocks into heading levels and body text"""
        # Use the existing extractor to get heading classifications
        text_blocks = []
        for block in blocks:
            text_blocks.append({
                'text': block['text'],
                'font_size': block['font_size'],
                'is_bold': 'Bold' in block['font_name'] or 'BoldMT' in block['font_name'],
                'font_name': block['font_name'],
                'page': block['page_number'],
                'bbox': block['bbox'],
                'x0': block['bbox'][0],
                'y0': block['bbox'][1],
                'x1': block['bbox'][2],
                'y1': block['bbox'][3]
            })
        
        # Get heading classifications from the extractor
        headings = self.extractor.classify_headings_fast(text_blocks)
        
        # Create a mapping from text to heading level (using HH format to match training data)
        heading_map = {}
        for heading in headings:
            # Convert level to int in case it's returned as string
            try:
                level = int(heading['level']) if isinstance(heading['level'], str) else heading['level']
            except (ValueError, TypeError):
                level = 1  # Default to level 1 if conversion fails
                
            if level <= 3:
                heading_map[heading['text']] = f"HH{level}"
            else:
                heading_map[heading['text']] = "H4"
        
        # First pass: detect title (usually first large text on first page)
        title_detected = False
        
        # Update block labels
        for i, block in enumerate(blocks):
            if block['text'] in heading_map:
                block['label'] = heading_map[block['text']]
            else:
                # Check for title first (early in document, large font, likely bold)
                if not title_detected and self.is_likely_title(block, blocks):
                    block['label'] = "TITLE"
                    title_detected = True
                # Additional heuristics for heading detection
                elif self.is_likely_heading(block):
                    block['label'] = self.determine_heading_level(block, blocks)
                else:
                    block['label'] = "BODY"
        
        return blocks
    
    def is_likely_title(self, block, all_blocks):
        """Check if a block is likely a document title"""
        # Title criteria:
        # 1. Usually appears early in the document (first few blocks)
        # 2. Larger font size than most text
        # 3. Often bold or special font
        # 4. Usually on first page
        # 5. Not too long (titles are typically concise)
        
        font_size = block['font_size']
        page = block['page_number']
        text = block['text']
        font_name = block['font_name']
        
        # Must be on first page
        if page != 1:
            return False
        
        # Skip very short text (likely not a title)
        if len(text.strip()) < 5:
            return False
        
        # Skip very long text (likely not a title)
        if len(text.strip()) > 200:
            return False
        
        # Get font size statistics for comparison
        font_sizes = [b['font_size'] for b in all_blocks if b['page_number'] == 1]
        if not font_sizes:
            return False
        
        avg_font_size = sum(font_sizes) / len(font_sizes)
        max_font_size = max(font_sizes)
        
        # Check if font is significantly larger than average
        is_large_font = font_size > avg_font_size * 1.2
        
        # Check if it's the largest or second largest font on first page
        is_prominent_font = font_size >= sorted(set(font_sizes), reverse=True)[0:2][-1] if len(set(font_sizes)) >= 2 else font_size == max_font_size
        
        # Check for bold or special fonts
        is_bold_or_special = any(keyword in font_name for keyword in ['Bold', 'BoldMT', 'Black', 'Heavy', 'Medium'])
        
        # Check position (titles are often near the top but not necessarily the very first text)
        block_index = all_blocks.index(block)
        is_early_position = block_index < min(10, len(all_blocks) // 4)  # Within first 10 blocks or first quarter
        
        # Title patterns (often ALL CAPS or Title Case)
        import re
        is_caps = text.isupper() and len(text) > 5
        is_title_case = text.istitle()
        
        # Combine criteria
        title_score = 0
        if is_large_font: title_score += 2
        if is_prominent_font: title_score += 2
        if is_bold_or_special: title_score += 1
        if is_early_position: title_score += 1
        if is_caps or is_title_case: title_score += 1
        
        # Require a minimum score to be considered a title
        return title_score >= 3
    
    def is_likely_heading(self, block):
        """Additional heuristics to detect headings"""
        text = block['text']
        font_size = block['font_size']
        font_name = block['font_name']
        
        # Check for bold fonts
        is_bold = any(keyword in font_name for keyword in ['Bold', 'BoldMT', 'Black'])
        
        # Check for typical heading patterns
        import re
        heading_patterns = [
            r'^\d+\.\s+[A-Z]',           # "1. Introduction"
            r'^\d+\.\d+\s+[A-Z]',        # "1.1 Background"
            r'^\d+\.\d+\.\d+\s+[A-Z]',   # "1.1.1 Details"
            r'^[A-Z][A-Z\s]{3,}:?\s*$',  # ALL CAPS headings
            r'^\d+\s+[A-Z][a-z]',        # "1 Introduction"
            r'^[A-Z]\.\s+[A-Z]',         # "A. Section"
            r'^(Chapter|Section|Part|Appendix)\s+\d+',
            r'^[IVX]+\.\s+',             # Roman numerals
        ]
        
        has_pattern = any(re.match(pattern, text) for pattern in heading_patterns)
        
        # Heading indicators
        heading_words = [
            'introduction', 'background', 'summary', 'conclusion', 'methodology',
            'results', 'discussion', 'references', 'abstract', 'overview'
        ]
        
        has_heading_word = any(word in text.lower() for word in heading_words)
        
        # Consider it a heading if it's bold and has patterns, or is significantly larger font
        return (is_bold and (has_pattern or has_heading_word)) or font_size > 16
    
    def determine_heading_level(self, block, all_blocks):
        """Determine the heading level based on font size and context"""
        font_size = block['font_size']
        
        # Get font sizes of all blocks for comparison
        font_sizes = [b['font_size'] for b in all_blocks]
        font_sizes.sort(reverse=True)
        
        # Remove duplicates while preserving order
        unique_sizes = []
        for size in font_sizes:
            if size not in unique_sizes:
                unique_sizes.append(size)
        
        # Assign heading levels based on font size ranking (matching training data format)
        if font_size in unique_sizes[:3]:  # Top 3 font sizes
            rank = unique_sizes.index(font_size)
            if rank == 0:
                return "HH1"
            elif rank == 1:
                return "HH2"
            else:
                return "HH3"
        else:
            return "H4"
    
    def generate_formatted_labels(self, pdf_dir, output_dir, review_mode=False):
        """Generate labels in the specified format for all PDFs in directory"""
        
        pdf_dir = Path(pdf_dir)
        output_dir = Path(output_dir)
        
        # Check if PDF directory exists
        if not pdf_dir.exists():
            print(f"❌ ERROR: PDF directory does not exist: {pdf_dir}")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files to label")
        
        if not pdf_files:
            print(f"❌ No PDF files found in {pdf_dir}")
            print(f"📁 Directory contents: {list(pdf_dir.iterdir())}")
            return
        
        successful_files = 0
        failed_files = 0
        
        for i, pdf_file in enumerate(pdf_files):
            print(f"\n🔄 Processing {i+1}/{len(pdf_files)}: {pdf_file.name}")
            
            try:
                # Check if PDF file is accessible
                if not pdf_file.exists():
                    print(f"❌ PDF file not found: {pdf_file}")
                    failed_files += 1
                    continue
                
                print(f"📄 File size: {pdf_file.stat().st_size} bytes")
                
                # Extract detailed blocks with formatting info
                print("🔍 Extracting text blocks...")
                blocks = self.extract_detailed_blocks(str(pdf_file))
                
                if not blocks:
                    print(f"⚠️  No text blocks found in {pdf_file.name}")
                    failed_files += 1
                    continue
                
                print(f"✓ Extracted {len(blocks)} text blocks")
                
                # Classify blocks into headings and body text
                print("🏷️  Classifying labels...")
                labeled_blocks = self.classify_block_labels(blocks)
                
                # Count labels for verification
                label_counts = {}
                for block in labeled_blocks:
                    label = block['label']
                    label_counts[label] = label_counts.get(label, 0) + 1
                
                print(f"✓ Label distribution: {dict(sorted(label_counts.items()))}")
                
                # Save formatted labels
                label_file = output_dir / f"{pdf_file.stem}.json"
                print(f"💾 Saving to: {label_file}")
                
                with open(label_file, 'w', encoding='utf-8') as f:
                    json.dump(labeled_blocks, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Successfully generated {len(labeled_blocks)} labeled blocks for {pdf_file.name}")
                successful_files += 1
                
                if review_mode:
                    self.review_formatted_labels(pdf_file, labeled_blocks)
                    
            except Exception as e:
                print(f"❌ Error processing {pdf_file.name}: {e}")
                print(f"🔍 Error type: {type(e).__name__}")
                import traceback
                print(f"📋 Full traceback:\n{traceback.format_exc()}")
                failed_files += 1
        
        # Final summary
        print(f"\n📊 PROCESSING SUMMARY:")
        print(f"✅ Successful: {successful_files}")
        print(f"❌ Failed: {failed_files}")
        print(f"📁 Output directory: {output_dir}")
        
        if successful_files > 0:
            json_files = list(output_dir.glob("*.json"))
            print(f"📄 Generated JSON files: {len(json_files)}")
        else:
            print("⚠️  No JSON files were generated!")
    
    def review_formatted_labels(self, pdf_file, labeled_blocks):
        """Interactive review of generated formatted labels"""
        print(f"\n--- Review for {pdf_file.name} ---")
        
        # Count labels
        label_counts = {}
        for block in labeled_blocks:
            label = block['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"Total blocks: {len(labeled_blocks)}")
        print("Label distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count}")
        
        # Show sample headings and titles
        titles = [b for b in labeled_blocks if b['label'] == 'TITLE']
        headings = [b for b in labeled_blocks if b['label'].startswith('H')]
        
        if titles:
            print(f"\nTitles found ({len(titles)}):")
            for title in titles:
                print(f"  {title['label']}: '{title['text']}' (font: {title['font_size']})")
        
        if headings:
            print(f"\nSample headings (showing first 5 of {len(headings)}):")
            for heading in headings[:5]:
                print(f"  {heading['label']}: '{heading['text']}' (font: {heading['font_size']})")
        
        # Simple review prompt
        response = input("\nAccept (y), Skip (s), or Quit (q)? ").lower()
        if response == 'q':
            return False
        return response == 'y'
    
    def show_sample_output(self, labeled_blocks, num_samples=3):
        """Show sample of the formatted output"""
        print(f"\nSample output format (showing first {num_samples} blocks):")
        print(json.dumps(labeled_blocks[:num_samples], indent=2, ensure_ascii=False))

def main():
    """Main labeling function"""
    labeler = FormattedPDFLabelingTool()
    
    # Choose your dataset
    pdf_directory = input("Enter PDF directory path (or press Enter for default): ").strip()
    if not pdf_directory:
        pdf_directory = r"training data\Pdf"
    
    output_directory = input("Enter output directory for formatted labels (or press Enter for default): ").strip()
    if not output_directory:
        output_directory = r"formatted_training_labels"
     
    print(f"PDF Directory: {pdf_directory}")
    print(f"Output Directory: {output_directory}")
    
    # Option for review mode
    review = input("Enable review mode? (y/n): ").lower() == 'y'
    
    # Show sample mode
    show_sample = input("Show sample output for first PDF? (y/n): ").lower() == 'y'
    
    if show_sample and Path(pdf_directory).exists():
        pdf_files = list(Path(pdf_directory).glob("*.pdf"))
        if pdf_files:
            print(f"\nGenerating sample for {pdf_files[0].name}...")
            sample_blocks = labeler.extract_detailed_blocks(str(pdf_files[0]))
            if sample_blocks:
                sample_labeled = labeler.classify_block_labels(sample_blocks)
                labeler.show_sample_output(sample_labeled)
                
                proceed = input("\nProceed with full processing? (y/n): ").lower() == 'y'
                if not proceed:
                    print("Exiting...")
                    return
    
    labeler.generate_formatted_labels(pdf_directory, output_directory, review_mode=review)
    print("Formatted labeling complete!")
    
    # Show final statistics
    if Path(output_directory).exists():
        json_files = list(Path(output_directory).glob("*.json"))
        print(f"\nGenerated {len(json_files)} JSON label files in {output_directory}")

if __name__ == "__main__":
    main()
