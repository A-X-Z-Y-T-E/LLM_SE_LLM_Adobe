import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import os
from datetime import datetime
import re
import math

# ========================================================================
# LABEL MAPPING SYSTEM FOR DOCUMENT STRUCTURE CLASSIFICATION
# ========================================================================

class LabelMappingSystem:
    """
    Manages label mappings for document structure classification.
    Handles conversion between string labels and numerical IDs for ML models.
    """
    
    def __init__(self):
        # Load existing label mappings from file
        self.load_existing_mappings()
    
    def load_existing_mappings(self):
        """
        Load label mappings from existing label_mappings.json file.
        """ 
        try:
            # Try to load from the existing file
            project_root = Path(__file__).parent.parent
            mappings_file = project_root / "data" / "label_mappings.json"
            
            if mappings_file.exists():
                with open(mappings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                mappings = data.get("mappings", {})
                self.label_to_id = mappings.get("label_to_id", {})
                
                # Convert string keys to int for id_to_label
                id_to_label_raw = mappings.get("id_to_label", {})
                self.id_to_label = {int(k): v for k, v in id_to_label_raw.items()}
                
                self.DOCUMENT_LABELS = mappings.get("labels_list", [])
                self.num_classes = len(self.DOCUMENT_LABELS)
                
                print("✅ Loaded existing label mappings:")
                for label, label_id in self.label_to_id.items():
                    print(f"   {label} → {label_id}")
                    
            else:
                print("⚠️ label_mappings.json not found, creating default mappings")
                self.create_default_mappings()
                
        except Exception as e:
            print(f"❌ Error loading label mappings: {e}")
            print("🔄 Creating default mappings...")
            self.create_default_mappings()
    
    def create_default_mappings(self):
        """
        Create default label mappings if file doesn't exist.
        """
        # Use the exact labels from the existing JSON file
        self.DOCUMENT_LABELS = ["BODY", "HH1", "HH2", "HH3", "H4", "TITLE"]
        
        self.label_to_id = {label: idx for idx, label in enumerate(self.DOCUMENT_LABELS)}
        self.id_to_label = {idx: label for idx, label in enumerate(self.DOCUMENT_LABELS)}
        self.num_classes = len(self.DOCUMENT_LABELS)
        
        print("📋 Created default label mappings:")
        for label, label_id in self.label_to_id.items():
            print(f"   {label} → {label_id}")

    def get_label_id(self, label: str) -> int:
        """
        Convert string label to numerical ID.
        
        Args:
            label (str): String label (e.g., "HH1", "BODY", "TITLE")
            
        Returns:
            int: Numerical ID for the label
            
        Raises:
            ValueError: If label is not recognized
        """
        # Handle both HH4 and H4 for backward compatibility
        if label == "HH4":
            label = "H4"
            
        if label not in self.label_to_id:
            raise ValueError(f"Unknown label: {label}. Valid labels: {list(self.label_to_id.keys())}")
        
        return self.label_to_id[label]
    
    def get_label_string(self, label_id: int) -> str:
        """
        Convert numerical ID back to string label.
        
        Args:
            label_id (int): Numerical ID
            
        Returns:
            str: String label
        """
        if label_id not in self.id_to_label:
            raise ValueError(f"Unknown label ID: {label_id}. Valid IDs: {list(self.id_to_label.keys())}")
        
        return self.id_to_label[label_id]

# ========================================================================
# FEATURE ENGINEERING FOR DOCUMENT LAYOUT ANALYSIS (DLA) GNN
# ========================================================================

def extract_and_normalize_features(all_text_blocks_raw_data: list) -> tuple:
    """
    Perform feature engineering on extracted text blocks from PDF documents.
    OPTIMIZED: Added validation, progress tracking, and error recovery.
    
    Args:
        all_text_blocks_raw_data (list): List of dictionaries representing text blocks
        
    Returns:
        tuple: (processed_blocks_list, normalization_params_dict)
    """
    if not all_text_blocks_raw_data:
        return [], {}
    
    print(f"🔧 Starting feature engineering for {len(all_text_blocks_raw_data)} text blocks...")
    
    # Initialize label mapping system
    try:
        label_mapper = LabelMappingSystem()
    except Exception as e:
        print(f"❌ Error initializing label mapper: {e}")
        return [], {}
    
    # Step 1: Calculate page dimensions and basic metrics
    print("📐 Calculating page dimensions...")
    page_dimensions = calculate_page_dimensions(all_text_blocks_raw_data)
    
    if not page_dimensions:
        print("⚠️ No valid page dimensions found, using defaults")
        page_dimensions = {1: {'page_width': 595.0, 'page_height': 842.0}}
    
    # Step 2: Calculate global statistics for normalization
    print("📊 Calculating global statistics...")
    global_stats = calculate_global_statistics(all_text_blocks_raw_data, page_dimensions)
    
    # Step 3: Process each block and extract features
    print("🔧 Extracting features from blocks...")
    processed_blocks = []
    error_count = 0
    
    # Progress tracking for large datasets
    total_blocks = len(all_text_blocks_raw_data)
    progress_interval = max(1, total_blocks // 20)  # Show progress every 5%
    
    for i, block in enumerate(all_text_blocks_raw_data):
        try:
            # Progress indicator
            if i % progress_interval == 0 and i > 0:
                print(f"   📈 Progress: {i}/{total_blocks} ({i/total_blocks*100:.1f}%)")
            
            # Get previous block for contextual features
            prev_block = all_text_blocks_raw_data[i-1] if i > 0 else None
            
            # Extract all features for current block
            processed_block = extract_block_features(
                block, 
                prev_block, 
                page_dimensions, 
                global_stats, 
                label_mapper
            )
            
            processed_blocks.append(processed_block)
            
        except Exception as e:
            error_count += 1
            print(f"⚠️ Error processing block {i}: {e}")
            
            # Skip problematic blocks but don't crash
            if error_count > total_blocks * 0.1:  # If >10% errors, stop
                print(f"❌ Too many errors ({error_count}), stopping processing")
                break
            continue
    
    # Step 4: Prepare normalization parameters for saving
    normalization_params = prepare_normalization_params(global_stats, page_dimensions)
    
    print(f"✅ Feature engineering complete!")
    print(f"   📊 Processed: {len(processed_blocks)} blocks")
    print(f"   ⚠️ Errors: {error_count} blocks")
    print(f"   🔧 Features per block: 20")
    print(f"   🏷️ Labels: {label_mapper.num_classes} classes")
    
    return processed_blocks, normalization_params

def calculate_page_dimensions(blocks: list) -> dict:
    """
    Calculate page dimensions from bounding boxes.
    OPTIMIZED: Single pass through blocks for O(n) complexity.
    
    Args:
        blocks (list): List of text block dictionaries
        
    Returns:
        dict: Page dimensions by page number
    """
    page_dims = {}
    
    for block in blocks:
        page_num = block.get('page_number', 1)
        bbox = block.get('bbox', [0, 0, 0, 0])
        
        # Skip invalid bboxes
        if len(bbox) != 4 or any(not isinstance(x, (int, float)) for x in bbox):
            continue
            
        x0, y0, x1, y1 = bbox
        
        # Ensure proper bbox ordering
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        
        if page_num not in page_dims:
            page_dims[page_num] = {
                'max_x': x1,
                'max_y': y1,
                'min_x': x0,
                'min_y': y0
            }
        else:
            dims = page_dims[page_num]
            dims['max_x'] = max(dims['max_x'], x1)
            dims['max_y'] = max(dims['max_y'], y1)
            dims['min_x'] = min(dims['min_x'], x0)
            dims['min_y'] = min(dims['min_y'], y0)
    
    # Calculate page width and height with validation
    for page_num in page_dims:
        dims = page_dims[page_num]
        dims['page_width'] = max(dims['max_x'] - dims['min_x'], 100.0)
        dims['page_height'] = max(dims['max_y'] - dims['min_y'], 100.0)
        
        # Use A4 defaults if calculated dimensions are unreasonable
        if dims['page_width'] < 100 or dims['page_width'] > 2000:
            dims['page_width'] = 595.0  # A4 width in points
        if dims['page_height'] < 100 or dims['page_height'] > 3000:
            dims['page_height'] = 842.0  # A4 height in points
    
    return page_dims

def calculate_global_statistics(blocks: list, page_dimensions: dict) -> dict:
    """
    Calculate global statistics for normalization.
    OPTIMIZED: Single pass with pre-allocated lists and vectorized operations.
    
    Args:
        blocks (list): List of text block dictionaries
        page_dimensions (dict): Page dimensions
        
    Returns:
        dict: Global statistics for normalization
    """
    # Pre-allocate lists for better memory performance
    font_sizes = []
    text_lengths = []
    y_offsets = []
    x_offsets = []
    font_size_ratios = []
    
    # Note: Python lists don't have reserve method, so we just use empty lists
    prev_block = None
    
    for block in blocks:
        # Font size - with validation
        font_size = block.get('font_size', 10.0)
        if isinstance(font_size, (int, float)) and 0.1 <= font_size <= 200:
            font_sizes.append(float(font_size))
        else:
            font_sizes.append(10.0)  # Default fallback
        
        # Text length
        text = block.get('text', '')
        if isinstance(text, str):
            text_lengths.append(len(text))
        else:
            text_lengths.append(0)
        
        # Contextual features (skip first block)
        if prev_block is not None:
            current_page = block.get('page_number', 1)
            page_dims = page_dimensions.get(current_page, {})
            page_height = page_dims.get('page_height', 842.0)
            page_width = page_dims.get('page_width', 595.0)
            
            # Get bboxes with validation
            current_bbox = block.get('bbox', [0, 0, 0, 0])
            prev_bbox = prev_block.get('bbox', [0, 0, 0, 0])
            
            if len(current_bbox) == 4 and len(prev_bbox) == 4:
                # Y offset to previous block (normalized)
                if page_height > 0:
                    y_offset = (current_bbox[1] - prev_bbox[3]) / page_height
                    # Clamp to reasonable range
                    y_offset = max(-2.0, min(2.0, y_offset))
                    y_offsets.append(y_offset)
                
                # X offset to previous block (normalized)
                if page_width > 0:
                    x_offset = (current_bbox[0] - prev_bbox[0]) / page_width
                    # Clamp to reasonable range
                    x_offset = max(-2.0, min(2.0, x_offset))
                    x_offsets.append(x_offset)
            
            # Font size ratio with validation
            prev_font_size = prev_block.get('font_size', 10.0)
            current_font_size = font_sizes[-1]  # Use validated font size
            
            if isinstance(prev_font_size, (int, float)) and prev_font_size > 0.1:
                font_ratio = current_font_size / prev_font_size
                # Clamp to reasonable range (0.1x to 10x)
                font_ratio = max(0.1, min(10.0, font_ratio))
                font_size_ratios.append(font_ratio)
        
        prev_block = block
    
    # Calculate min/max with error handling
    def safe_min_max(values, default_min=0.0, default_max=1.0):
        if not values:
            return default_min, default_max
        
        # Filter out invalid values
        valid_values = [v for v in values if isinstance(v, (int, float)) and not (math.isnan(v) or math.isinf(v))]
        
        if not valid_values:
            return default_min, default_max
        
        return float(min(valid_values)), float(max(valid_values))
    
    # Calculate statistics
    font_min, font_max = safe_min_max(font_sizes, 8.0, 24.0)
    length_min, length_max = safe_min_max(text_lengths, 0.0, 500.0)
    y_offset_min, y_offset_max = safe_min_max(y_offsets, -1.0, 1.0)
    x_offset_min, x_offset_max = safe_min_max(x_offsets, -1.0, 1.0)
    ratio_min, ratio_max = safe_min_max(font_size_ratios, 0.5, 2.0)
    
    return {
        'font_sizes_min': font_min,
        'font_sizes_max': font_max,
        'text_lengths_min': length_min,
        'text_lengths_max': length_max,
        'y_offsets_min': y_offset_min,
        'y_offsets_max': y_offset_max,
        'x_offsets_min': x_offset_min,
        'x_offsets_max': x_offset_max,
        'font_size_ratios_min': ratio_min,
        'font_size_ratios_max': ratio_max
    }

def extract_block_features(block: dict, prev_block: dict, page_dimensions: dict, 
                         global_stats: dict, label_mapper: LabelMappingSystem) -> dict:
    """
    Extract all 20 features for a single text block.
    OPTIMIZED: Reduced redundant calculations and added validation.
    
    Args:
        block (dict): Current text block
        prev_block (dict): Previous text block (None for first block)
        page_dimensions (dict): Page dimensions
        global_stats (dict): Global statistics for normalization
        label_mapper (LabelMappingSystem): Label mapping system
        
    Returns:
        dict: Processed block with features
    """
    # Start with original block data
    processed_block = block.copy()
    
    # Add label mapping with error handling
    raw_label = block.get('label', 'BODY')
    processed_block['raw_label_string'] = raw_label
    
    try:
        processed_block['label_id'] = label_mapper.get_label_id(raw_label)
    except ValueError:
        print(f"⚠️ Unknown label '{raw_label}', using BODY (0)")
        processed_block['label_id'] = 0  # Default to BODY
    
    # Get page dimensions with fallback
    page_num = block.get('page_number', 1)
    page_dims = page_dimensions.get(page_num, {})
    page_width = page_dims.get('page_width', 595.0)
    page_height = page_dims.get('page_height', 842.0)
    
    # Validate page dimensions
    if page_width <= 0 or page_height <= 0:
        page_width, page_height = 595.0, 842.0
    
    # Extract and validate bounding box
    bbox = block.get('bbox', [0, 0, 0, 0])
    if len(bbox) != 4:
        bbox = [0, 0, 0, 0]
    
    x0, y0, x1, y1 = [float(x) for x in bbox]
    
    # Ensure proper bbox ordering
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    
    # Initialize features dictionary
    features = {}
    
    # ========================================================================
    # A. GEOMETRIC FEATURES (8 features) - OPTIMIZED
    # ========================================================================
    
    # Pre-calculate normalized coordinates (reused multiple times)
    x0_norm = x0 / page_width
    y0_norm = y0 / page_height
    x1_norm = x1 / page_width
    y1_norm = y1 / page_height
    
    features['x0_norm'] = float(x0_norm)
    features['y0_norm'] = float(y0_norm)
    features['x1_norm'] = float(x1_norm)
    features['y1_norm'] = float(y1_norm)
    
    # Normalized dimensions
    features['width_norm'] = float((x1 - x0) / page_width)
    features['height_norm'] = float((y1 - y0) / page_height)
    
    # Alignment features (optimized with pre-calculated values)
    features['is_left_aligned'] = 1.0 if abs(x0_norm) < 0.01 else 0.0
    
    # Center horizontally (reuse x0_norm, x1_norm)
    center_x_norm = (x0_norm + x1_norm) / 2
    features['is_centered_horizontally'] = 1.0 if abs(center_x_norm - 0.5) < 0.05 else 0.0
    
    # ========================================================================
    # B. STYLISTIC FEATURES (3 features) - OPTIMIZED
    # ========================================================================
    
    # Get and validate font size
    font_size = block.get('font_size', 10.0)
    if not isinstance(font_size, (int, float)) or font_size <= 0:
        font_size = 10.0
    
    # Normalized font size with safe division
    min_font = global_stats.get('font_sizes_min', font_size)
    max_font = global_stats.get('font_sizes_max', font_size)
    
    if max_font > min_font:
        features['font_size_norm'] = float((font_size - min_font) / (max_font - min_font))
    else:
        features['font_size_norm'] = 0.5  # Default if all fonts same size
    
    # Font style features (case-insensitive, optimized)
    font_name_lower = str(block.get('font_name', '')).lower()
    features['is_bold'] = 1.0 if ('bold' in font_name_lower or 'black' in font_name_lower) else 0.0
    features['is_italic'] = 1.0 if ('italic' in font_name_lower or 'oblique' in font_name_lower) else 0.0
    
    # ========================================================================
    # C. TEXTUAL/LEXICAL FEATURES (5 features) - OPTIMIZED
    # ========================================================================
    
    text = str(block.get('text', '')).strip()
    text_length = len(text)
    
    # Normalized text length with safe division
    min_length = global_stats.get('text_lengths_min', 0)
    max_length = global_stats.get('text_lengths_max', 1)
    
    if max_length > min_length:
        features['text_length_chars_norm'] = float((text_length - min_length) / (max_length - min_length))
    else:
        features['text_length_chars_norm'] = 0.0
    
    # Text pattern features (optimized with early returns)
    features['starts_with_bullet'] = 1.0 if text and text[0] in '•-*' else 0.0
    features['ends_with_colon'] = 1.0 if text and text[-1] == ':' else 0.0
    features['is_all_caps'] = 1.0 if (text and len(text) > 2 and text.isupper()) else 0.0
    
    # Number prefix pattern (compiled regex for better performance)
    features['contains_number_prefix'] = 1.0 if text and re.match(r'^\s*\d+\.?\s', text) else 0.0
    
    # ========================================================================
    # D. CONTEXTUAL FEATURES (6 features) - OPTIMIZED
    # ========================================================================
    
    if prev_block is None:
        # First block - set contextual features to 0 (vectorized assignment)
        contextual_defaults = {
            'y_offset_to_prev_block_norm': 0.0,
            'x_offset_to_prev_block_norm': 0.0,
            'font_size_ratio_to_prev_block_norm': 0.0,
            'is_prev_block_same_font_size': 0.0,
            'is_prev_block_same_bold_status': 0.0,
            'is_prev_block_same_indentation': 0.0
        }
        features.update(contextual_defaults)
    else:
        # Calculate contextual features efficiently
        prev_bbox = prev_block.get('bbox', [0, 0, 0, 0])
        if len(prev_bbox) == 4:
            prev_x0, prev_y0, prev_x1, prev_y1 = [float(x) for x in prev_bbox]
            
            # Y offset (normalized) with safe division
            y_offset_raw = (y0 - prev_y1) / page_height
            min_y_offset = global_stats.get('y_offsets_min', 0)
            max_y_offset = global_stats.get('y_offsets_max', 1)
            
            if max_y_offset > min_y_offset:
                features['y_offset_to_prev_block_norm'] = float((y_offset_raw - min_y_offset) / (max_y_offset - min_y_offset))
            else:
                features['y_offset_to_prev_block_norm'] = 0.0
            
            # X offset (normalized) with safe division
            x_offset_raw = (x0 - prev_x0) / page_width
            min_x_offset = global_stats.get('x_offsets_min', 0)
            max_x_offset = global_stats.get('x_offsets_max', 1)
            
            if max_x_offset > min_x_offset:
                features['x_offset_to_prev_block_norm'] = float((x_offset_raw - min_x_offset) / (max_x_offset - min_x_offset))
            else:
                features['x_offset_to_prev_block_norm'] = 0.0
        else:
            features['y_offset_to_prev_block_norm'] = 0.0
            features['x_offset_to_prev_block_norm'] = 0.0
        
        # Font size ratio (normalized) with validation
        prev_font_size = prev_block.get('font_size', 10.0)
        if isinstance(prev_font_size, (int, float)) and prev_font_size > 0:
            font_ratio_raw = font_size / prev_font_size
            min_ratio = global_stats.get('font_size_ratios_min', 1)
            max_ratio = global_stats.get('font_size_ratios_max', 1)
            
            if max_ratio > min_ratio:
                features['font_size_ratio_to_prev_block_norm'] = float((font_ratio_raw - min_ratio) / (max_ratio - min_ratio))
            else:
                features['font_size_ratio_to_prev_block_norm'] = 0.5
        else:
            features['font_size_ratio_to_prev_block_norm'] = 0.0
        
        # Boolean comparisons (optimized)
        features['is_prev_block_same_font_size'] = 1.0 if abs(font_size - prev_font_size) < 1.0 else 0.0
        
        # Same bold status (reuse previous calculation)
        prev_font_name_lower = str(prev_block.get('font_name', '')).lower()
        prev_is_bold = 1.0 if ('bold' in prev_font_name_lower or 'black' in prev_font_name_lower) else 0.0
        features['is_prev_block_same_bold_status'] = 1.0 if features['is_bold'] == prev_is_bold else 0.0
        
        # Same indentation (reuse x0_norm)
        if len(prev_bbox) == 4:
            prev_x0_norm = prev_x0 / page_width
            features['is_prev_block_same_indentation'] = 1.0 if abs(x0_norm - prev_x0_norm) < 0.01 else 0.0
        else:
            features['is_prev_block_same_indentation'] = 0.0
    
    # Add features to processed block
    processed_block['features'] = features
    
    return processed_block

def prepare_normalization_params(global_stats: dict, page_dimensions: dict) -> dict:
    """
    Prepare normalization parameters for saving.
    
    Args:
        global_stats (dict): Global statistics
        page_dimensions (dict): Page dimensions
        
    Returns:
        dict: Normalization parameters
    """
    return {
        "metadata": {
            "description": "Normalization parameters for document layout analysis features",
            "feature_count": 20,
            "created_timestamp": datetime.now().isoformat(),
            "version": "1.0"
        },
        "global_statistics": global_stats,
        "page_dimensions_sample": {
            "typical_page_width": 595.0,
            "typical_page_height": 842.0,
            "note": "Page dimensions calculated dynamically per page"
        },
        "feature_descriptions": {
            "geometric": ["x0_norm", "y0_norm", "x1_norm", "y1_norm", "width_norm", "height_norm", "is_left_aligned", "is_centered_horizontally"],
            "stylistic": ["font_size_norm", "is_bold", "is_italic"],
            "textual": ["text_length_chars_norm", "starts_with_bullet", "ends_with_colon", "is_all_caps", "contains_number_prefix"],
            "contextual": ["y_offset_to_prev_block_norm", "x_offset_to_prev_block_norm", "font_size_ratio_to_prev_block_norm", "is_prev_block_same_font_size", "is_prev_block_same_bold_status", "is_prev_block_same_indentation"]
        }
    }

def load_json_files_from_directory(directory_path: str) -> List[dict]:
    """
    Load all JSON files from the specified directory.
    
    Args:
        directory_path (str): Path to directory containing JSON files
        
    Returns:
        List[dict]: Combined list of all text blocks from all JSON files
    """
    directory = Path(directory_path)
    all_blocks = []
    
    if not directory.exists():
        print(f"❌ Directory not found: {directory_path}")
        return []
    
    json_files = list(directory.glob("*.json"))
    
    if not json_files:
        print(f"❌ No JSON files found in: {directory_path}")
        return []
    
    print(f"📂 Loading {len(json_files)} JSON files from: {directory_path}")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Handle different JSON structures
            if isinstance(data, list):
                all_blocks.extend(data)
            elif isinstance(data, dict) and 'annotations' in data:
                all_blocks.extend(data['annotations'])
            else:
                print(f"⚠️ Unexpected JSON structure in {json_file.name}")
                
        except Exception as e:
            print(f"❌ Error loading {json_file.name}: {e}")
    
    print(f"✅ Loaded {len(all_blocks)} total text blocks")
    return all_blocks

def process_json_files_and_extract_features(input_directory: str = None, output_path: str = None) -> None:
    """
    Main function to process JSON files and extract features.
    
    Args:
        input_directory (str, optional): Directory containing JSON files
        output_path (str, optional): Path to save processed features
    """
    # Set default paths
    if input_directory is None:
        project_root = Path(__file__).parent.parent
        input_directory = project_root / "data" / "new_new_json_files"
    
    if output_path is None:
        project_root = Path(__file__).parent.parent
        output_path = project_root / "training_data" / "processed_features.json"
    
    input_directory = Path(input_directory)
    output_path = Path(output_path)
    
    print("🔧 DOCUMENT LAYOUT ANALYSIS - FEATURE ENGINEERING")
    print("=" * 70)
    print(f"📂 Input directory: {input_directory}")
    print(f"💾 Output path: {output_path}")
    
    # Load all JSON files
    all_blocks = load_json_files_from_directory(str(input_directory))
    
    if not all_blocks:
        print("❌ No data to process!")
        return
    
    # Sort blocks by reading order (page_number, then y0, then x0)
    print("📑 Sorting blocks by reading order...")
    all_blocks.sort(key=lambda x: (x.get('page_number', 0), x.get('bbox', [0, 0, 0, 0])[1], x.get('bbox', [0, 0, 0, 0])[0]))
    
    # Extract features
    processed_blocks, normalization_params = extract_and_normalize_features(all_blocks)
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save processed features
    output_data = {
        "metadata": {
            "description": "Processed features for document layout analysis",
            "total_blocks": len(processed_blocks),
            "feature_count": 20,
            "label_classes": 6,
            "processing_timestamp": datetime.now().isoformat()
        },
        "processed_blocks": processed_blocks,
        "normalization_parameters": normalization_params
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Feature extraction complete!")
    print(f"💾 Saved {len(processed_blocks)} processed blocks to: {output_path}")
    
    # Save normalization parameters separately
    norm_params_path = output_path.parent / "normalization_parameters.json"
    with open(norm_params_path, 'w', encoding='utf-8') as f:
        json.dump(normalization_params, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Saved normalization parameters to: {norm_params_path}")

def save_simple_format_blocks(blocks: List[dict], output_path: str) -> None:
    """
    Save text blocks in simple format without features for basic annotation workflow.
    
    Args:
        blocks (List[dict]): List of text blocks with basic information
        output_path (str): Path to save the JSON file
    """
    # Extract only the core fields for simple format
    simple_blocks = []
    
    for block in blocks:
        simple_block = {
            "pdf_id": block.get("pdf_id", "unknown"),
            "page_number": block.get("page_number", 1),
            "block_id": block.get("block_id", ""),
            "text": block.get("text", ""),
            "bbox": block.get("bbox", [0, 0, 0, 0]),
            "font_size": float(block.get("font_size", 10.0)),
            "font_name": block.get("font_name", ""),
            "label": block.get("label", "BODY")
        }
        simple_blocks.append(simple_block)
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(simple_blocks, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(simple_blocks)} blocks in simple format to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error saving simple format blocks: {e}")
        raise

def save_processed_features_per_document(all_blocks: List[dict], output_directory: str = None) -> None:
    """
    Save processed blocks with features grouped by document (pdf_id).
    Creates one JSON file per document containing all blocks from that document.
    
    Args:
        all_blocks (List[dict]): List of processed blocks with features
        output_directory (str, optional): Directory to save document files
    """
    if output_directory is None:
        project_root = Path(__file__).parent.parent
        output_directory = project_root / "training_data" / "processed_by_document"
    
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group blocks by pdf_id
    documents = {}
    for block in all_blocks:
        pdf_id = block.get("pdf_id", "unknown")
        if pdf_id not in documents:
            documents[pdf_id] = []
        documents[pdf_id].append(block)
    
    print(f"📄 Saving {len(documents)} documents with processed features...")
    
    # Save each document to its own file
    for pdf_id, doc_blocks in documents.items():
        # Sort blocks by reading order within document
        doc_blocks.sort(key=lambda x: (
            x.get('page_number', 0), 
            x.get('bbox', [0, 0, 0, 0])[1], 
            x.get('bbox', [0, 0, 0, 0])[0]
        ))
        
        # Generate filename
        safe_pdf_id = "".join(c for c in pdf_id if c.isalnum() or c in ('_', '-'))
        output_file = output_dir / f"{safe_pdf_id}_processed_features.json"
        
        # Prepare document data
        document_data = {
            "metadata": {
                "pdf_id": pdf_id,
                "total_blocks": len(doc_blocks),
                "pages": list(set(block.get('page_number', 1) for block in doc_blocks)),
                "feature_count": 20,
                "label_classes": 6,
                "processing_timestamp": datetime.now().isoformat()
            },
            "blocks": doc_blocks
        }
        
        # Save document
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(document_data, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ {pdf_id}: {len(doc_blocks)} blocks → {output_file.name}")
            
        except Exception as e:
            print(f"   ❌ Error saving {pdf_id}: {e}")
    
    print(f"🎉 Saved all documents to: {output_dir}")

def save_simple_format_per_document(all_blocks: List[dict], output_directory: str = None) -> None:
    """
    Save blocks in simple format grouped by document (pdf_id).
    Creates one JSON file per document with only basic fields (no features).
    
    Args:
        all_blocks (List[dict]): List of blocks (processed or raw)
        output_directory (str, optional): Directory to save document files
    """
    if output_directory is None:
        project_root = Path(__file__).parent.parent
        output_directory = project_root / "data" / "simple_format_by_document"
    
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group blocks by pdf_id
    documents = {}
    for block in all_blocks:
        pdf_id = block.get("pdf_id", "unknown")
        if pdf_id not in documents:
            documents[pdf_id] = []
        documents[pdf_id].append(block)
    
    print(f"📄 Saving {len(documents)} documents in simple format...")
    
    # Save each document to its own file
    for pdf_id, doc_blocks in documents.items():
        # Sort blocks by reading order within document
        doc_blocks.sort(key=lambda x: (
            x.get('page_number', 0), 
            x.get('bbox', [0, 0, 0, 0])[1], 
            x.get('bbox', [0, 0, 0, 0])[0]
        ))
        
        # Convert to simple format
        simple_blocks = []
        for block in doc_blocks:
            simple_block = {
                "pdf_id": block.get("pdf_id", "unknown"),
                "page_number": block.get("page_number", 1),
                "block_id": block.get("block_id", ""),
                "text": block.get("text", ""),
                "bbox": block.get("bbox", [0, 0, 0, 0]),
                "font_size": float(block.get("font_size", 10.0)),
                "font_name": block.get("font_name", ""),
                "label": block.get("label", "BODY")
            }
            simple_blocks.append(simple_block)
        
        # Generate filename
        safe_pdf_id = "".join(c for c in pdf_id if c.isalnum() or c in ('_', '-'))
        output_file = output_dir / f"{safe_pdf_id}_simple.json"
        
        # Save document
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(simple_blocks, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ {pdf_id}: {len(simple_blocks)} blocks → {output_file.name}")
            
        except Exception as e:
            print(f"   ❌ Error saving {pdf_id}: {e}")
    
    print(f"🎉 Saved all documents to: {output_dir}")

def convert_processed_to_simple_format(input_file: str, output_file: str = None) -> None:
    """
    Convert a processed features file back to simple format.
    
    Args:
        input_file (str): Path to processed features JSON file
        output_file (str, optional): Path for simple format output
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_file}")
        return
    
    if output_file is None:
        output_file = input_path.parent / f"{input_path.stem}_simple.json"
    
    # Load processed data
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract blocks
        if isinstance(data, dict) and 'processed_blocks' in data:
            blocks = data['processed_blocks']
        elif isinstance(data, dict) and 'blocks' in data:
            blocks = data['blocks']
        elif isinstance(data, list):
            blocks = data
        else:
            print(f"❌ Unexpected data structure in {input_file}")
            return
        
        # Save in simple format
        save_simple_format_blocks(blocks, output_file)
        
    except Exception as e:
        print(f"❌ Error converting to simple format: {e}")

def save_features_as_vectors_per_document(all_blocks: List[dict], output_directory: str = None) -> None:
    """
    Save processed blocks with features as vectors grouped by document (pdf_id).
    Creates one JSON file per document where each block contains all original data + features vector.
    
    This format is optimized for ML training with features as [num_blocks, num_features] vectors.
    
    Args:
        all_blocks (List[dict]): List of processed blocks with features
        output_directory (str, optional): Directory to save document files
    """
    if output_directory is None:
        project_root = Path(__file__).parent.parent
        output_directory = project_root / "training_data" / "features_per_document"
    
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group blocks by pdf_id
    documents = {}
    for block in all_blocks:
        pdf_id = block.get("pdf_id", "unknown")
        if pdf_id not in documents:
            documents[pdf_id] = []
        documents[pdf_id].append(block)
    
    print(f"📄 Saving {len(documents)} documents with feature vectors...")
    
    # Save each document to its own file
    for pdf_id, doc_blocks in documents.items():
        # Sort blocks by reading order within document
        doc_blocks.sort(key=lambda x: (
            x.get('page_number', 0), 
            x.get('bbox', [0, 0, 0, 0])[1], 
            x.get('bbox', [0, 0, 0, 0])[0]
        ))
        
        # Convert to the specified format
        formatted_blocks = []
        for block in doc_blocks:
            # Extract features dictionary or create empty one
            features = block.get('features', {})
            
            # Ensure all 20 features are present with default values
            complete_features = {
                # Geometric features (8)
                "x0_norm": features.get("x0_norm", 0.0),
                "y0_norm": features.get("y0_norm", 0.0),
                "x1_norm": features.get("x1_norm", 0.0),
                "y1_norm": features.get("y1_norm", 0.0),
                "width_norm": features.get("width_norm", 0.0),
                "height_norm": features.get("height_norm", 0.0),
                "is_left_aligned": features.get("is_left_aligned", 0.0),
                "is_centered_horizontally": features.get("is_centered_horizontally", 0.0),
                
                # Stylistic features (3)
                "font_size_norm": features.get("font_size_norm", 0.0),
                "is_bold": features.get("is_bold", 0.0),
                "is_italic": features.get("is_italic", 0.0),
                
                # Textual features (5)
                "text_length_chars_norm": features.get("text_length_chars_norm", 0.0),
                "starts_with_bullet": features.get("starts_with_bullet", 0.0),
                "ends_with_colon": features.get("ends_with_colon", 0.0),
                "is_all_caps": features.get("is_all_caps", 0.0),
                "contains_number_prefix": features.get("contains_number_prefix", 0.0),
                
                # Contextual features (6)
                "y_offset_to_prev_block_norm": features.get("y_offset_to_prev_block_norm", 0.0),
                "x_offset_to_prev_block_norm": features.get("x_offset_to_prev_block_norm", 0.0),
                "font_size_ratio_to_prev_block_norm": features.get("font_size_ratio_to_prev_block_norm", 0.0),
                "is_prev_block_same_font_size": features.get("is_prev_block_same_font_size", 0.0),
                "is_prev_block_same_bold_status": features.get("is_prev_block_same_bold_status", 0.0),
                "is_prev_block_same_indentation": features.get("is_prev_block_same_indentation", 0.0)
            }
            
            # Create formatted block with all required fields
            formatted_block = {
                "pdf_id": block.get("pdf_id", "unknown"),
                "page_number": block.get("page_number", 1),
                "block_id": block.get("block_id", ""),
                "text": block.get("text", ""),
                "bbox": block.get("bbox", [0, 0, 0, 0]),
                "font_size": float(block.get("font_size", 10.0)),
                "font_name": block.get("font_name", ""),
                "raw_label_string": block.get("raw_label_string", block.get("label", "BODY")),
                "label_id": block.get("label_id", 0),
                "features": complete_features
            }
            
            formatted_blocks.append(formatted_block)
        
        # Generate filename based on pdf_id
        safe_pdf_id = "".join(c for c in pdf_id if c.isalnum() or c in ('_', '-'))
        output_file = output_dir / f"{safe_pdf_id}_features.json"
        
        # Save document with metadata
        document_data = {
            "metadata": {
                "pdf_id": pdf_id,
                "total_blocks": len(formatted_blocks),
                "pages": list(set(block.get('page_number', 1) for block in formatted_blocks)),
                "feature_count": 20,
                "label_classes": 6,
                "processing_timestamp": datetime.now().isoformat(),
                "format": "features_per_document",
                "description": "Document blocks with 20 normalized features as vectors"
            },
            "blocks": formatted_blocks
        }
        
        # Save document
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(document_data, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ {pdf_id}: {len(formatted_blocks)} blocks → {output_file.name}")
            
        except Exception as e:
            print(f"   ❌ Error saving {pdf_id}: {e}")
    
    print(f"🎉 Saved all feature vectors to: {output_dir}")

def save_features_as_flat_vectors_per_document(all_blocks: List[dict], output_directory: str = None) -> None:
    """
    Save processed blocks as pure feature vectors per document for direct ML consumption.
    Creates files with format: [num_blocks, num_features] arrays + separate labels.
    
    Args:
        all_blocks (List[dict]): List of processed blocks with features
        output_directory (str, optional): Directory to save document files
    """
    if output_directory is None:
        project_root = Path(__file__).parent.parent
        output_directory = project_root / "training_data" / "ml_ready_vectors"
    
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group blocks by pdf_id
    documents = {}
    for block in all_blocks:
        pdf_id = block.get("pdf_id", "unknown")
        if pdf_id not in documents:
            documents[pdf_id] = []
        documents[pdf_id].append(block)
    
    print(f"📄 Saving {len(documents)} documents as ML-ready feature vectors...")
    
    # Feature names in order (for reference)
    feature_names = [
        "x0_norm", "y0_norm", "x1_norm", "y1_norm", "width_norm", "height_norm", 
        "is_left_aligned", "is_centered_horizontally", "font_size_norm", "is_bold", 
        "is_italic", "text_length_chars_norm", "starts_with_bullet", "ends_with_colon", 
        "is_all_caps", "contains_number_prefix", "y_offset_to_prev_block_norm", 
        "x_offset_to_prev_block_norm", "font_size_ratio_to_prev_block_norm", 
        "is_prev_block_same_font_size", "is_prev_block_same_bold_status", 
        "is_prev_block_same_indentation"
    ]
    
    # Save each document
    for pdf_id, doc_blocks in documents.items():
        # Sort blocks by reading order within document
        doc_blocks.sort(key=lambda x: (
            x.get('page_number', 0), 
            x.get('bbox', [0, 0, 0, 0])[1], 
            x.get('bbox', [0, 0, 0, 0])[0]
        ))
        
        # Extract features as vectors
        feature_vectors = []
        labels = []
        block_ids = []
        
        for block in doc_blocks:
            features = block.get('features', {})
            
            # Extract feature vector in consistent order
            feature_vector = [
                float(features.get(feature_name, 0.0)) 
                for feature_name in feature_names
            ]
            
            feature_vectors.append(feature_vector)
            labels.append(block.get("label_id", 0))
            block_ids.append(block.get("block_id", ""))
        
        # Generate filename
        safe_pdf_id = "".join(c for c in pdf_id if c.isalnum() or c in ('_', '-'))
        output_file = output_dir / f"{safe_pdf_id}_vectors.json"
        
        # Prepare ML-ready data
        ml_data = {
            "metadata": {
                "pdf_id": pdf_id,
                "num_blocks": len(feature_vectors),
                "num_features": len(feature_names),
                "feature_names": feature_names,
                "label_classes": 6,
                "processing_timestamp": datetime.now().isoformat(),
                "format": "ml_ready_vectors"
            },
            "feature_vectors": feature_vectors,  # [num_blocks, 20] array
            "labels": labels,                    # [num_blocks] array
            "block_ids": block_ids              # [num_blocks] array for reference
        }
        
        # Save document
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(ml_data, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ {pdf_id}: [{len(feature_vectors)}, {len(feature_names)}] → {output_file.name}")
            
        except Exception as e:
            print(f"   ❌ Error saving {pdf_id}: {e}")
    
    print(f"🎉 Saved all ML-ready vectors to: {output_dir}")

def process_and_save_by_document(input_directory: str = None, output_directory: str = None, 
                                save_simple: bool = True, save_features: bool = True,
                                save_vectors: bool = True) -> None:
    """
    Enhanced processing function that saves data in multiple formats organized by document.
    
    Args:
        input_directory (str, optional): Directory containing JSON files
        output_directory (str, optional): Base directory for outputs
        save_simple (bool): Whether to save simple format files
        save_features (bool): Whether to save processed features files
        save_vectors (bool): Whether to save feature vectors
    """
    # Set default paths
    if input_directory is None:
        project_root = Path(__file__).parent.parent
        input_directory = project_root / "data" / "new_new_json_files"
    
    if output_directory is None:
        project_root = Path(__file__).parent.parent
        output_directory = project_root / "training_data"
    
    input_directory = Path(input_directory)
    output_directory = Path(output_directory)
    
    print("🔧 ENHANCED DOCUMENT PROCESSING WITH MULTIPLE OUTPUT FORMATS")
    print("=" * 70)
    print(f"📂 Input directory: {input_directory}")
    print(f"📁 Output directory: {output_directory}")
    print(f"📄 Save simple format: {save_simple}")
    print(f"🔧 Save features: {save_features}")
    print(f"📊 Save vectors: {save_vectors}")
    
    # Load all JSON files
    all_blocks = load_json_files_from_directory(str(input_directory))
    
    if not all_blocks:
        print("❌ No data to process!")
        return
    
    # Sort blocks by reading order
    print("📑 Sorting blocks by reading order...")
    all_blocks.sort(key=lambda x: (
        x.get('page_number', 0), 
        x.get('bbox', [0, 0, 0, 0])[1], 
        x.get('bbox', [0, 0, 0, 0])[0]
    ))
    
    # Save simple format by document (before processing)
    if save_simple:
        print("\n💾 Saving simple format by document...")
        simple_output_dir = output_directory / "simple_format_by_document"
        save_simple_format_per_document(all_blocks, str(simple_output_dir))
    
    # Extract features if requested
    if save_features or save_vectors:
        print("\n🔧 Extracting features...")
        processed_blocks, normalization_params = extract_and_normalize_features(all_blocks)
        
        if save_features:
            # Save processed features by document
            print("\n💾 Saving processed features by document...")
            features_output_dir = output_directory / "processed_by_document"
            save_processed_features_per_document(processed_blocks, str(features_output_dir))
        
        if save_vectors:
            # Save feature vectors in the requested format
            print("\n📊 Saving feature vectors per document...")
            vectors_output_dir = output_directory / "features_per_document"
            save_features_as_vectors_per_document(processed_blocks, str(vectors_output_dir))
            
            # Also save ML-ready format
            print("\n🤖 Saving ML-ready vectors...")
            ml_output_dir = output_directory / "ml_ready_vectors"
            save_features_as_flat_vectors_per_document(processed_blocks, str(ml_output_dir))
        
        # Save combined processed features file
        combined_output_path = output_directory / "combined_processed_features.json"
        output_data = {
            "metadata": {
                "description": "Combined processed features for all documents",
                "total_blocks": len(processed_blocks),
                "feature_count": 20,
                "label_classes": 6,
                "processing_timestamp": datetime.now().isoformat()
            },
            "processed_blocks": processed_blocks,
            "normalization_parameters": normalization_params
        }
        
        with open(combined_output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Saved combined features to: {combined_output_path}")
        
        # Save normalization parameters separately
        norm_params_path = output_directory / "normalization_parameters.json"
        with open(norm_params_path, 'w', encoding='utf-8') as f:
            json.dump(normalization_params, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Saved normalization parameters to: {norm_params_path}")

# ========================================================================
# MAIN EXECUTION
# ========================================================================

if __name__ == "__main__":
    print("🏷️  DOCUMENT STRUCTURE FEATURE ENGINEERING")
    print("=" * 80)
    
    # Choose what to run
    print("Choose an option:")
    print("1. Process JSON files and extract features (old format)")
    print("2. Process and save by document (enhanced format with vectors)")
    print("3. Convert processed file to simple format")
    print("4. All - process by document with all formats")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        process_json_files_and_extract_features()
        
    elif choice == "2":
        save_simple = input("Save simple format? (y/n, default: y): ").strip().lower() != 'n'
        save_features = input("Save processed features? (y/n, default: y): ").strip().lower() != 'n'
        save_vectors = input("Save feature vectors? (y/n, default: y): ").strip().lower() != 'n'
        process_and_save_by_document(save_simple=save_simple, save_features=save_features, save_vectors=save_vectors)
        
    elif choice == "3":
        input_file = input("Enter path to processed features file: ").strip()
        output_file = input("Enter output path (or Enter for auto): ").strip() or None
        convert_processed_to_simple_format(input_file, output_file)
        
    elif choice == "4":
        print("🔄 Processing all formats by document...")
        process_and_save_by_document()
        
    else:
        print("Invalid choice. Running enhanced processing...")
        process_and_save_by_document()
    
    print("\n🎉 Feature engineering operations complete!")
    
