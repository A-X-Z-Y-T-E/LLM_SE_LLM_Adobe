#!/usr/bin/env python3
"""
Improved Adobe Hackathon Round 1A: PDF Heading Extractor
Fixed based on sample outputs analysis
"""

import fitz  # PyMuPDF
import json
import os
import re
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedPDFExtractor:
    def __init__(self):
        # Adjusted weights based on sample analysis
        self.weights = {
            'font_size': 0.40,
            'bold': 0.30,
            'position': 0.15,
            'whitespace': 0.10,
            'pattern': 0.05
        }
        
        # More precise heading patterns
        self.heading_patterns = [
            r'^\d+\.\s+[A-Z]',           # "1. Introduction"
            r'^\d+\.\d+\s+[A-Z]',        # "1.1 Background"
            r'^\d+\.\d+\.\d+\s+[A-Z]',   # "1.1.1 Details"
            r'^[A-Z][A-Z\s]{3,}:?\s*$',  # ALL CAPS headings
            r'^\d+\s+[A-Z][a-z]',        # "1 Introduction"
            r'^[A-Z]\.\s+[A-Z]',         # "A. Section"
            r'^\d+\.\s*$',               # Just numbers "1."
            r'^(Chapter|Section|Part|Appendix)\s+\d+',
            r'^[IVX]+\.\s+',             # Roman numerals
        ]
        
        # Words that indicate headings
        self.heading_indicators = [
            'introduction', 'background', 'summary', 'conclusion', 'methodology',
            'results', 'discussion', 'references', 'abstract', 'overview',
            'acknowledgements', 'table of contents', 'revision history',
            'appendix', 'bibliography', 'index'
        ]
        
        # Words/patterns that indicate NOT headings
        self.non_heading_indicators = [
            'page', 'figure', 'table', 'equation', 'note:', 'example:',
            'copyright', '©', 'all rights reserved', 'www.', 'http',
            'email', '@', '.com', '.org', '.net'
        ]
    
    def extract_text_blocks_enhanced(self, pdf_path):
        """Enhanced text extraction with better metadata"""
        try:
            doc = fitz.open(pdf_path)
            text_blocks = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")
                
                for block in blocks["blocks"]:
                    if "lines" not in block:
                        continue
                        
                    for line in block["lines"]:
                        line_text = ""
                        max_font_size = 0
                        is_bold = False
                        font_name = ""
                        total_flags = 0
                        
                        for span in line["spans"]:
                            line_text += span["text"]
                            if span["size"] > max_font_size:
                                max_font_size = span["size"]
                                font_name = span["font"]
                                total_flags = span["flags"]
                                is_bold = bool(span["flags"] & 16)  # Bold flag
                        
                        line_text = line_text.strip()
                        
                        # Skip very short, numeric-only, or irrelevant text
                        if (len(line_text) > 2 and 
                            not line_text.isdigit() and 
                            not re.match(r'^\s*$', line_text)):
                            
                            text_blocks.append({
                                'text': line_text,
                                'font_size': max_font_size,
                                'is_bold': is_bold,
                                'font_name': font_name,
                                'flags': total_flags,
                                'page': page_num + 1,
                                'bbox': line["bbox"],
                                'x0': line["bbox"][0],
                                'y0': line["bbox"][1],
                                'x1': line["bbox"][2], 
                                'y1': line["bbox"][3],
                                'width': line["bbox"][2] - line["bbox"][0],
                                'height': line["bbox"][3] - line["bbox"][1]
                            })
            
            doc.close()
            return text_blocks
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return []
    
    def is_likely_heading(self, text):
        """Determine if text is likely a heading based on content"""
        text_lower = text.lower().strip()
        
        # Check for non-heading indicators first
        for indicator in self.non_heading_indicators:
            if indicator in text_lower:
                return False
        
        # Check for heading patterns
        for pattern in self.heading_patterns:
            if re.match(pattern, text):
                return True
        
        # Check for heading indicator words
        for indicator in self.heading_indicators:
            if indicator in text_lower:
                return True
        
        # Check length and structure
        if len(text) > 200:  # Too long to be a heading
            return False
            
        # Check if it's mostly uppercase (excluding small words)
        words = text.split()
        if len(words) > 1:
            caps_words = [w for w in words if w.isupper() and len(w) > 2]
            if len(caps_words) >= len(words) * 0.7:  # 70% caps words
                return True
        
        return False
    
    def calculate_heading_score_improved(self, block, font_stats, whitespace_score, block_idx, total_blocks):
        """Improved heading score calculation"""
        score = 0.0
        text = block['text']
        
        # Font size score (relative to document)
        if font_stats['max'] > font_stats['mean']:
            size_ratio = (block['font_size'] - font_stats['mean']) / (font_stats['max'] - font_stats['mean'])
            score += self.weights['font_size'] * max(0, size_ratio)
        
        # Bold score
        if block['is_bold']:
            score += self.weights['bold']
        
        # Position score (prefer left-aligned, not heavily indented)
        if block['x0'] < 100:  # Not heavily indented
            score += self.weights['position'] * 0.8
        elif block['x0'] < 150:
            score += self.weights['position'] * 0.4
        
        # Whitespace score
        score += self.weights['whitespace'] * whitespace_score
        
        # Pattern/content score
        if self.is_likely_heading(text):
            score += self.weights['pattern']
        
        # Length penalty for very long text
        if len(text) > 100:
            score *= 0.5
        elif len(text) > 150:
            score *= 0.2
        
        # Early document bonus (headings often appear early)
        if block_idx < total_blocks * 0.1:  # First 10% of document
            score *= 1.2
        
        return score
    
    def classify_headings_improved(self, text_blocks):
        """Improved heading classification"""
        if not text_blocks:
            return []
        
        # Calculate font statistics
        font_sizes = [b['font_size'] for b in text_blocks if len(b['text']) > 3]
        if not font_sizes:
            return []
        
        font_stats = {
            'mean': np.mean(font_sizes),
            'std': np.std(font_sizes),
            'max': np.max(font_sizes),
            'min': np.min(font_sizes),
            'percentiles': np.percentile(font_sizes, [75, 90, 95])
        }
        
        candidates = []
        
        # Score all potential headings
        for i, block in enumerate(text_blocks):
            text = block['text'].strip()
            
            # More restrictive filtering
            if (len(text) < 3 or len(text) > 200 or 
                text.isdigit() or 
                text.count(' ') > 20):  # Skip very long paragraphs
                continue
            
            # Calculate whitespace score
            whitespace_score = self.calculate_whitespace_score(text_blocks, i)
            
            # Calculate heading score
            score = self.calculate_heading_score_improved(
                block, font_stats, whitespace_score, i, len(text_blocks)
            )
            
            # Higher threshold for better precision
            if score > 0.5:
                candidates.append({
                    'text': text,
                    'page': block['page'],
                    'score': score,
                    'font_size': block['font_size'],
                    'is_bold': block['is_bold'],
                    'order': i,
                    'x0': block['x0']
                })
        
        if not candidates:
            return []
        
        # Assign levels based on font size and content
        candidates.sort(key=lambda x: (-x['font_size'], x['order']))
        
        # Group by similar font sizes for level assignment
        levels_assigned = []
        font_groups = {}
        
        for candidate in candidates:
            font_size = candidate['font_size']
            
            # Group similar font sizes (within 1 point)
            group_key = None
            for existing_size in font_groups.keys():
                if abs(font_size - existing_size) <= 1.0:
                    group_key = existing_size
                    break
            
            if group_key is None:
                group_key = font_size
                font_groups[group_key] = []
            
            font_groups[group_key].append(candidate)
        
        # Sort groups by font size (descending) and assign levels
        sorted_groups = sorted(font_groups.items(), key=lambda x: x[0], reverse=True)
        
        for level_idx, (font_size, group_candidates) in enumerate(sorted_groups):
            level = f"H{min(level_idx + 1, 3)}"  # Max H3
            
            for candidate in group_candidates:
                candidate['level'] = level
                levels_assigned.append(candidate)
        
        # Sort by document order for final output
        levels_assigned.sort(key=lambda x: x['order'])
        
        return levels_assigned
    
    def calculate_whitespace_score(self, text_blocks, idx):
        """Calculate whitespace score for heading detection"""
        if idx == 0:
            return 0.8  # First block gets high score
        
        current = text_blocks[idx]
        prev = text_blocks[idx - 1]
        
        # Different page = high whitespace
        if current['page'] != prev['page']:
            return 1.0
        
        # Calculate vertical gap
        gap = current['y0'] - prev['y1']
        
        # Normalize gap
        if gap > 40:
            return 1.0
        elif gap > 25:
            return 0.8
        elif gap > 15:
            return 0.6
        elif gap > 5:
            return 0.3
        else:
            return 0.1
    
    def extract_title_improved(self, pdf_path, text_blocks):
        """Improved title extraction"""
        try:
            # Try metadata first
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            doc.close()
            
            if metadata and metadata.get('title'):
                title = metadata['title'].strip()
                if title and len(title) > 3 and not title.lower().endswith('.pdf'):
                    return title
            
            # Find title from document content
            if not text_blocks:
                return "Document"
            
            # Look in first few blocks of first page
            first_page_blocks = [b for b in text_blocks[:20] if b['page'] == 1]
            
            if first_page_blocks:
                # Find largest font or bold text in upper portion
                title_candidates = []
                
                for block in first_page_blocks:
                    text = block['text'].strip()
                    
                    # Skip very short or long text
                    if len(text) < 5 or len(text) > 150:
                        continue
                    
                    # Skip obvious non-titles
                    if any(word in text.lower() for word in 
                          ['page', 'chapter', 'section', 'figure', 'table']):
                        continue
                    
                    # Upper portion of page (y0 < 300)
                    if block['y0'] < 300:
                        title_candidates.append({
                            'text': text,
                            'font_size': block['font_size'],
                            'is_bold': block['is_bold'],
                            'y0': block['y0']
                        })
                
                if title_candidates:
                    # Prefer largest font, then bold, then position
                    title_candidates.sort(key=lambda x: (x['font_size'], x['is_bold'], -x['y0']), reverse=True)
                    return title_candidates[0]['text']
            
            # Fallback to filename without extension
            filename = Path(pdf_path).stem
            return filename.replace('_', ' ').replace('-', ' ').title()
            
        except Exception as e:
            logger.error(f"Error extracting title: {e}")
            return "Document"
    
    def process_pdf(self, pdf_path):
        """Main processing function"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing {pdf_path}")
            
            # Extract text blocks
            text_blocks = self.extract_text_blocks_enhanced(pdf_path)
            
            if not text_blocks:
                logger.warning(f"No text found in {pdf_path}")
                return {"title": "Document", "outline": []}
            
            # Extract title first
            title = self.extract_title_improved(pdf_path, text_blocks)
            
            # Extract headings
            headings = self.classify_headings_improved(text_blocks)
            
            # Format output to match expected schema
            outline = []
            for heading in headings:
                outline.append({
                    "level": heading['level'],
                    "text": heading['text'],
                    "page": heading['page']
                })
            
            result = {
                "title": title,
                "outline": outline
            }
            
            processing_time = time.time() - start_time
            logger.info(f"Processed {pdf_path} in {processing_time:.2f}s, found {len(outline)} headings")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return {"title": "Error", "outline": []}

def main():
    """Main entry point"""
    # Test with sample files first
    sample_dir = Path("d:/VS_CODE/Adobe/PS/Adobe-India-Hackathon25/Challenge_1a/sample_dataset/pdfs")
    output_dir = Path("d:/VS_CODE/Adobe/test_output")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize extractor
    extractor = ImprovedPDFExtractor()
    
    # Process sample PDFs
    if sample_dir.exists():
        pdf_files = list(sample_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} sample PDF files to process")
        
        for pdf_file in pdf_files:
            try:
                result = extractor.process_pdf(str(pdf_file))
                
                # Save result
                output_file = output_dir / f"{pdf_file.stem}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
                
                logger.info(f"Saved result to {output_file}")
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}")
    
    else:
        logger.warning("Sample directory not found")

if __name__ == "__main__":
    main()
