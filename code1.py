#!/usr/bin/env python3
"""
Adobe Hackathon Round 1A: PDF Heading Extractor
Optimized for speed, accuracy, and hackathon constraints
"""

import fitz  # PyMuPDF - fastest PDF library
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

class HackathonPDFExtractor:
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
        
    def extract_text_blocks_fast(self, pdf_path):
        """Fast text extraction with essential metadata"""
        try:
            doc = fitz.open(pdf_path)
            text_blocks = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Use get_text("dict") for detailed formatting info
                blocks = page.get_text("dict")
                
                for block in blocks["blocks"]:
                    if "lines" not in block:
                        continue
                        
                    for line in block["lines"]:
                        line_text = ""
                        max_font_size = 0
                        is_bold = False
                        font_name = ""
                        
                        # Combine spans in the same line
                        for span in line["spans"]:
                            line_text += span["text"]
                            if span["size"] > max_font_size:
                                max_font_size = span["size"]
                                font_name = span["font"]
                                is_bold = span["flags"] & 16  # Bold flag
                        
                        line_text = line_text.strip()
                        if len(line_text) > 2:  # Skip very short text
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
    
    def analyze_font_distribution(self, text_blocks):
        """Quick font analysis for adaptive thresholds"""
        font_sizes = [b['font_size'] for b in text_blocks if len(b['text']) > 3]
        
        if not font_sizes:
            return {'percentiles': [10, 12, 14, 16]}
            
        return {
            'mean': np.mean(font_sizes),
            'std': np.std(font_sizes),
            'percentiles': np.percentile(font_sizes, [50, 75, 90, 95]),
            'max': np.max(font_sizes)
        }
    
    def calculate_whitespace_score(self, text_blocks, idx):
        """Calculate whitespace above current block"""
        if idx == 0:
            return 0.5  # First block gets medium score
            
        current = text_blocks[idx]
        prev = text_blocks[idx - 1]
        
        # Different page = high whitespace
        if current['page'] != prev['page']:
            return 1.0
            
        # Calculate vertical gap
        gap = current['y0'] - prev['y1']
        
        # Normalize gap (typical line height ~15-20)
        if gap > 30:
            return 1.0
        elif gap > 15:
            return 0.7
        elif gap > 5:
            return 0.3
        else:
            return 0.0
    
    def is_likely_heading(self, text):
        """Determine if text is likely a heading based on content"""
        text_lower = text.lower().strip()
        
        # Check for non-heading indicators first
        for indicator in self.non_heading_indicators:
            if indicator in text_lower:
                return False
        
        # Form-specific exclusions - only for obvious form elements
        form_exclusions = [
            r'^\d+\.$',  # Just numbers like "1.", "2."
            r'^\d+\.\s*$',  # Numbers with spaces
            r'^s\.no$',  # Serial number
            r'^name$',   # Single word field labels
            r'^age$',
            r'^date$',
            r'^relationship$',
            r'^designation$',
            r'^amount$',
            r'^[a-z]+:?\s*$',  # Single lowercase words
        ]
        
        for pattern in form_exclusions:
            if re.match(pattern, text_lower):
                return False
        
        # Skip very short standalone text that's likely form fields
        if len(text) < 5 and len(text.split()) <= 1:  # Only single very short words
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

    def detect_heading_patterns(self, text):
        """Fast pattern matching for headings"""
        return self.is_likely_heading(text)
    
    def calculate_heading_score(self, block, font_stats, whitespace_score):
        """Calculate comprehensive heading score"""
        score = 0.0
        
        # Font size score (normalized)
        if font_stats['max'] > 0:
            size_ratio = block['font_size'] / font_stats['max']
            score += self.weights['font_size'] * size_ratio
        
        # Bold score
        if block['is_bold']:
            score += self.weights['bold']
        
        # Position score (prefer left-aligned, not too indented)
        if block['x0'] < 100:  # Not heavily indented
            score += self.weights['position']
        
        # Whitespace score
        score += self.weights['whitespace'] * whitespace_score
        
        # Pattern score
        if self.detect_heading_patterns(block['text']):
            score += self.weights['pattern']
        
        return score
    
    def classify_headings_fast(self, text_blocks):
        """Fast heading classification with clustering"""
        if not text_blocks:
            return []
            
        font_stats = self.analyze_font_distribution(text_blocks)
        candidates = []
        
    def classify_headings_fast(self, text_blocks):
        """Fast heading classification with clustering"""
        if not text_blocks:
            return []
            
        font_stats = self.analyze_font_distribution(text_blocks)
        candidates = []
        
    def classify_headings_fast(self, text_blocks):
        """Fast heading classification with clustering"""
        if not text_blocks:
            return []
            
        font_stats = self.analyze_font_distribution(text_blocks)
        candidates = []
        
        # Check if this looks like a form document
        # Look for form indicators in the text
        all_text = ' '.join([block['text'].lower() for block in text_blocks[:10]])
        is_form_like = any(indicator in all_text for indicator in [
            'application form', 'form for', 'claim form', 'request form'
        ]) and any(indicator in all_text for indicator in [
            'name of', 'designation', 'date of', 'amount'
        ])
        
        # Score all potential headings
        for i, block in enumerate(text_blocks):
            # More restrictive filtering
            if (len(block['text']) < 3 or 
                len(block['text']) > 200 or  # Increased from 200 to filter paragraphs
                block['text'].isdigit() or 
                block['text'].strip().count(' ') > 15):  # Skip very long text
                continue
                
            whitespace_score = self.calculate_whitespace_score(text_blocks, i)
            score = self.calculate_heading_score(block, font_stats, whitespace_score)
            
            # Adjust threshold based on document type
            if is_form_like:
                threshold = 0.95  # Very high threshold for forms
            else:
                threshold = 0.7   # Normal threshold for documents
            
            # Higher threshold for better precision
            if score > threshold:
                candidates.append({
                    'text': block['text'],
                    'page': block['page'],
                    'score': score,
                    'font_size': block['font_size'],
                    'is_bold': block['is_bold'],
                    'order': i
                })
        
        if not candidates:
            return []
        
        # Fast clustering for level assignment
        if len(candidates) >= 3:
            # Use font size and score for clustering
            features = [[c['font_size'], c['score']] for c in candidates]
            
            # Determine optimal clusters (max 3 levels: H1, H2, H3)
            n_clusters = min(3, len(candidates))
            
            if n_clusters > 1:
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
                cluster_labels = kmeans.fit_predict(features_scaled)
                
                # Map clusters to heading levels based on average font size
                cluster_info = defaultdict(list)
                for i, label in enumerate(cluster_labels):
                    cluster_info[label].append(candidates[i]['font_size'])
                
                # Sort clusters by average font size (descending)
                sorted_clusters = sorted(cluster_info.keys(), 
                                       key=lambda x: np.mean(cluster_info[x]), 
                                       reverse=True)
                
                # Assign levels
                level_map = {cluster: f"H{i+1}" for i, cluster in enumerate(sorted_clusters)}
                
                for i, candidate in enumerate(candidates):
                    candidate['level'] = level_map[cluster_labels[i]]
            else:
                # Single cluster - all H1
                for candidate in candidates:
                    candidate['level'] = "H1"
        else:
            # Few candidates - simple assignment
            candidates.sort(key=lambda x: x['font_size'], reverse=True)
            for i, candidate in enumerate(candidates):
                candidate['level'] = f"H{min(i+1, 3)}"
        
        # Sort by document order
        candidates.sort(key=lambda x: x['order'])
        
        return candidates
    
    def extract_title_fast(self, pdf_path, text_blocks):
        """Improved title extraction"""
        try:
            # Find title from document content first (more reliable for forms)
            if text_blocks:
                # Look in first few blocks of first page
                first_page_blocks = [b for b in text_blocks[:15] if b['page'] == 1]
                
                if first_page_blocks:
                    # Find largest font or bold text in upper portion
                    title_candidates = []
                    
                    for block in first_page_blocks:
                        text = block['text'].strip()
                        
                        # Skip very short or long text
                        if len(text) < 8 or len(text) > 150:
                            continue
                        
                        # Skip obvious non-titles (form field numbers, single words)
                        if (text.isdigit() or 
                            any(word in text.lower() for word in 
                                ['page', 'chapter', 'section', 'figure', 'table', 'form number', 'date:']) or
                            re.match(r'^\d+\.$', text) or  # Just numbers like "1."
                            len(text.split()) == 1):  # Single words
                            continue
                        
                        # Look for title-like patterns
                        if (text.lower().startswith(('application', 'form', 'report', 'document', 'manual', 'guide')) or
                            'application' in text.lower() or
                            'form' in text.lower()):
                            
                            # Upper portion of page (y0 < 200)
                            if block['y0'] < 200:
                                title_candidates.append({
                                    'text': text,
                                    'font_size': block['font_size'],
                                    'is_bold': block['is_bold'],
                                    'y0': block['y0'],
                                    'score': block['font_size'] + (20 if block['is_bold'] else 0) - block['y0']/10
                                })
                    
                    if title_candidates:
                        # Sort by score (font size + bold bonus - position penalty)
                        title_candidates.sort(key=lambda x: x['score'], reverse=True)
                        return title_candidates[0]['text'].strip() + "  "  # Add trailing spaces like expected
            
            # Try metadata as fallback, but clean it up
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            doc.close()
            
            if metadata and metadata.get('title'):
                title = metadata['title'].strip()
                # Skip obvious filename-based titles
                if (title and len(title) > 3 and 
                    not title.lower().endswith('.pdf') and
                    not title.lower().endswith('.doc') and
                    not title.lower().endswith('.docx') and
                    'microsoft word' not in title.lower()):
                    return title
            
            # Fallback to filename without extension
            filename = Path(pdf_path).stem
            return filename.replace('_', ' ').replace('-', ' ').title()
            
        except Exception as e:
            logger.error(f"Error extracting title: {e}")
            return "Document"
    
    def process_pdf(self, pdf_path):
        """Main processing function optimized for hackathon"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing {pdf_path}")
            
            # Extract text blocks
            text_blocks = self.extract_text_blocks_fast(pdf_path)
            
            if not text_blocks:
                logger.warning(f"No text found in {pdf_path}")
                return {"title": "Document", "outline": []}
            
            # Extract headings
            headings = self.classify_headings_fast(text_blocks)
            
            # Extract title
            title = self.extract_title_fast(pdf_path, text_blocks)
            
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
            return {"title": "Error", "outline": []}

def main():
    """Main entry point for hackathon execution"""
    # For Docker execution
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # For local testing, fallback to local paths
    if not input_dir.exists():
        input_dir = Path("d:/VS_CODE/Adobe/PS/Adobe-India-Hackathon25/Challenge_1a/sample_dataset/pdfs")
        output_dir = Path("d:/VS_CODE/Adobe/test_output")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize extractor
    extractor = HackathonPDFExtractor()
    
    # Process all PDFs in input directory
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning("No PDF files found in input directory")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        try:
            # Process PDF
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