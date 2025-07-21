#!/usr/bin/env python3
"""
Final optimized Adobe Hackathon Round 1A: PDF Heading Extractor
Based on analysis of all sample outputs
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

class FinalPDFExtractor:
    def __init__(self):
        # Optimized weights
        self.weights = {
            'font_size': 0.40,
            'bold': 0.30,
            'position': 0.15,
            'whitespace': 0.10,
            'pattern': 0.05
        }
        
        # Precise heading patterns
        self.heading_patterns = [
            r'^\d+\.\s+[A-Z]',           # "1. Introduction"
            r'^\d+\.\d+\s+[A-Z]',        # "1.1 Background"
            r'^\d+\.\d+\.\d+\s+[A-Z]',   # "1.1.1 Details"
            r'^[A-Z][A-Z\s]{3,}:?\s*$',  # ALL CAPS headings
            r'^\d+\s+[A-Z][a-z]',        # "1 Introduction"
            r'^[A-Z]\.\s+[A-Z]',         # "A. Section"
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
    
    def extract_text_blocks(self, pdf_path):
        """Extract text blocks with metadata"""
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
                        
                        for span in line["spans"]:
                            line_text += span["text"]
                            if span["size"] > max_font_size:
                                max_font_size = span["size"]
                                font_name = span["font"]
                                is_bold = bool(span["flags"] & 16)
                        
                        line_text = line_text.strip()
                        
                        if (len(line_text) > 2 and 
                            not line_text.isdigit() and 
                            not re.match(r'^\s*$', line_text)):
                            
                            text_blocks.append({
                                'text': line_text,
                                'font_size': max_font_size,
                                'is_bold': is_bold,
                                'font_name': font_name,
                                'page': page_num + 1,
                                'bbox': line["bbox"],
                                'x0': line["bbox"][0],
                                'y0': line["bbox"][1],
                                'x1': line["bbox"][2], 
                                'y1': line["bbox"][3]
                            })
            
            doc.close()
            return text_blocks
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return []
    
    def is_form_document(self, text_blocks):
        """Detect if this is a form document"""
        all_text = ' '.join([block['text'].lower() for block in text_blocks[:15]])
        
        form_indicators = [
            'application form', 'form for', 'claim form', 'request form'
        ]
        
        field_indicators = [
            'name of', 'designation', 'date of', 'amount', 'employee'
        ]
        
        has_form_title = any(indicator in all_text for indicator in form_indicators)
        has_form_fields = sum(1 for indicator in field_indicators if indicator in all_text) >= 2
        
        return has_form_title and has_form_fields
    
    def is_likely_heading(self, text, is_form=False):
        """Determine if text is likely a heading"""
        text_lower = text.lower().strip()
        
        # Non-heading indicators
        for indicator in self.non_heading_indicators:
            if indicator in text_lower:
                return False
        
        # Form-specific exclusions
        if is_form:
            form_exclusions = [
                r'^\d+\.$',  # Just numbers
                r'^s\.no$', r'^name$', r'^age$', r'^date$',
                r'^relationship$', r'^designation$', r'^amount$',
                r'^[a-z]+:?\s*$',  # Single lowercase words
            ]
            
            for pattern in form_exclusions:
                if re.match(pattern, text_lower):
                    return False
        
        # Very short text is likely not a heading
        if len(text) < 5 and len(text.split()) <= 1:
            return False
        
        # Check heading patterns
        for pattern in self.heading_patterns:
            if re.match(pattern, text):
                return True
        
        # Check heading indicator words
        for indicator in self.heading_indicators:
            if indicator in text_lower:
                return True
        
        # Too long to be a heading
        if len(text) > 200:
            return False
            
        # Mostly uppercase check
        words = text.split()
        if len(words) > 1:
            caps_words = [w for w in words if w.isupper() and len(w) > 2]
            if len(caps_words) >= len(words) * 0.7:
                return True
        
        return False
    
    def extract_title(self, pdf_path, text_blocks):
        """Extract document title"""
        try:
            # First look in document content
            if text_blocks:
                first_page_blocks = [b for b in text_blocks[:15] if b['page'] == 1]
                
                if first_page_blocks:
                    title_candidates = []
                    
                    for block in first_page_blocks:
                        text = block['text'].strip()
                        
                        # Skip short or very long text
                        if len(text) < 8 or len(text) > 150:
                            continue
                        
                        # Skip obvious non-titles
                        if (text.isdigit() or 
                            any(word in text.lower() for word in 
                                ['page', 'chapter', 'section', 'figure', 'table', 'date:']) or
                            re.match(r'^\d+\.$', text) or
                            len(text.split()) == 1):
                            continue
                        
                        # Look for title-like patterns in upper area
                        if block['y0'] < 200:
                            score = block['font_size']
                            if block['is_bold']:
                                score += 20
                            score -= block['y0'] / 10  # Prefer higher position
                            
                            title_candidates.append({
                                'text': text,
                                'score': score
                            })
                    
                    if title_candidates:
                        title_candidates.sort(key=lambda x: x['score'], reverse=True)
                        best_title = title_candidates[0]['text']
                        
                        # Add trailing spaces to match expected format
                        if not best_title.endswith('  '):
                            best_title += '  '
                        
                        return best_title
            
            # Fallback to metadata
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            doc.close()
            
            if metadata and metadata.get('title'):
                title = metadata['title'].strip()
                if (title and len(title) > 3 and 
                    not any(ext in title.lower() for ext in ['.pdf', '.doc', '.docx']) and
                    'microsoft word' not in title.lower()):
                    return title
            
            # Last resort - return empty string for some documents
            return ""
            
        except Exception as e:
            logger.error(f"Error extracting title: {e}")
            return ""
    
    def calculate_whitespace_score(self, text_blocks, idx):
        """Calculate whitespace score"""
        if idx == 0:
            return 0.8
        
        current = text_blocks[idx]
        prev = text_blocks[idx - 1]
        
        if current['page'] != prev['page']:
            return 1.0
        
        gap = current['y0'] - prev['y1']
        
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
    
    def calculate_heading_score(self, block, font_stats, whitespace_score):
        """Calculate heading score"""
        score = 0.0
        
        # Font size score
        if font_stats['max'] > font_stats['mean']:
            size_ratio = (block['font_size'] - font_stats['mean']) / (font_stats['max'] - font_stats['mean'])
            score += self.weights['font_size'] * max(0, size_ratio)
        
        # Bold score
        if block['is_bold']:
            score += self.weights['bold']
        
        # Position score
        if block['x0'] < 100:
            score += self.weights['position'] * 0.8
        elif block['x0'] < 150:
            score += self.weights['position'] * 0.4
        
        # Whitespace score
        score += self.weights['whitespace'] * whitespace_score
        
        # Pattern score
        if self.is_likely_heading(block['text']):
            score += self.weights['pattern']
        
        return score
    
    def classify_headings(self, text_blocks):
        """Classify headings with form detection"""
        if not text_blocks:
            return []
        
        # Check if this is a form
        is_form = self.is_form_document(text_blocks)
        
        # Calculate font statistics
        font_sizes = [b['font_size'] for b in text_blocks if len(b['text']) > 3]
        if not font_sizes:
            return []
        
        font_stats = {
            'mean': np.mean(font_sizes),
            'max': np.max(font_sizes),
            'min': np.min(font_sizes)
        }
        
        candidates = []
        
        # Score potential headings
        for i, block in enumerate(text_blocks):
            text = block['text'].strip()
            
            # Basic filtering
            if (len(text) < 3 or len(text) > 200 or 
                text.isdigit() or 
                text.count(' ') > 20):
                continue
            
            # Skip if not likely a heading
            if not self.is_likely_heading(text, is_form):
                continue
            
            whitespace_score = self.calculate_whitespace_score(text_blocks, i)
            score = self.calculate_heading_score(block, font_stats, whitespace_score)
            
            # Threshold based on document type
            threshold = 0.95 if is_form else 0.7
            
            if score > threshold:
                candidates.append({
                    'text': text,
                    'page': block['page'],
                    'score': score,
                    'font_size': block['font_size'],
                    'is_bold': block['is_bold'],
                    'order': i
                })
        
        if not candidates:
            return []
        
        # Assign heading levels based on font size
        candidates.sort(key=lambda x: (-x['font_size'], x['order']))
        
        # Group by font size for level assignment
        levels_assigned = []
        font_groups = {}
        
        for candidate in candidates:
            font_size = candidate['font_size']
            
            # Group similar font sizes
            group_key = None
            for existing_size in font_groups.keys():
                if abs(font_size - existing_size) <= 1.5:
                    group_key = existing_size
                    break
            
            if group_key is None:
                group_key = font_size
                font_groups[group_key] = []
            
            font_groups[group_key].append(candidate)
        
        # Assign levels
        sorted_groups = sorted(font_groups.items(), key=lambda x: x[0], reverse=True)
        
        for level_idx, (font_size, group_candidates) in enumerate(sorted_groups):
            level = f"H{min(level_idx + 1, 3)}"
            
            for candidate in group_candidates:
                candidate['level'] = level
                levels_assigned.append(candidate)
        
        # Sort by document order
        levels_assigned.sort(key=lambda x: x['order'])
        
        return levels_assigned
    
    def process_pdf(self, pdf_path):
        """Main processing function"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing {pdf_path}")
            
            # Extract text blocks
            text_blocks = self.extract_text_blocks(pdf_path)
            
            if not text_blocks:
                logger.warning(f"No text found in {pdf_path}")
                return {"title": "", "outline": []}
            
            # Extract title
            title = self.extract_title(pdf_path, text_blocks)
            
            # Extract headings
            headings = self.classify_headings(text_blocks)
            
            # Format output
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
            return {"title": "", "outline": []}

def main():
    """Main entry point for production"""
    input_dir = Path("/app/input")  # Docker mount point
    output_dir = Path("/app/output")  # Docker mount point
    
    # For testing, use local paths
    if not input_dir.exists():
        input_dir = Path("D:/VS_CODE/Adobe/dataset-of-pdf-files/versions/1/Pdf")
        output_dir = Path("D:/VS_CODE/Adobe/output")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize extractor
    extractor = FinalPDFExtractor()
    
    # Process all PDFs
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning("No PDF files found in input directory")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        try:
            result = extractor.process_pdf(str(pdf_file))
            
            # Generate output filename
            output_file = output_dir / f"{pdf_file.stem}.json"
            
            # Save result
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved result to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_file}: {e}")

if __name__ == "__main__":
    main()
