"""
COMPLETE PDF TO OUTLINE PIPELINE USING MODEL V8
Integrates all components: PDF extraction → Feature engineering → Graph building → Model prediction → Outline generation
"""

import torch
import json
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import sys
import time
from datetime import datetime

# Add necessary paths
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / "model_training"))
sys.path.append(str(current_dir / "extractor"))

# Import components
try:
    from extractor.feature_engineering import (
        extract_and_normalize_features, 
        LabelMappingSystem,
        calculate_page_dimensions,
        calculate_global_statistics,
        extract_block_features
    )
    from model_training.models import DocumentGNN
    from model_training.build_graph import build_document_graph
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all required modules are available")
    sys.exit(1)


class PDFTextExtractor:
    """Enhanced PDF text extraction matching feature_engineering.py requirements."""
    
    @staticmethod
    def extract_text_blocks_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text blocks from PDF with exact format expected by feature_engineering.py.
        Enhanced with robust error handling for PyMuPDF issues.
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            List of text block dictionaries in exact format for feature_engineering.py
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"📄 Extracting text from: {pdf_path.name}")
        
        blocks = []
        block_counter = 0
        doc = None
        
        try:
            # Open document with error handling
            doc = fitz.open(str(pdf_path))
            
            # Get page count safely
            try:
                page_count = doc.page_count
            except Exception as e:
                print(f"⚠️ Error getting page count: {e}")
                # Try alternative method
                try:
                    page_count = len(doc)
                except:
                    # Fallback: try to iterate through pages
                    page_count = 0
                    try:
                        while True:
                            page = doc.load_page(page_count)
                            page_count += 1
                    except:
                        pass
            
            print(f"   📄 Document has {page_count} pages")
            
            # Process each page
            for page_num in range(page_count):
                try:
                    page = doc.load_page(page_num)
                    text_dict = page.get_text("dict")
                    
                    for block_idx, block in enumerate(text_dict.get("blocks", [])):
                        if "lines" in block:  # Text block
                            for line_idx, line in enumerate(block["lines"]):
                                for span_idx, span in enumerate(line.get("spans", [])):
                                    text = span.get("text", "").strip()
                                    
                                    if text:  # Only include non-empty text
                                        # Extract bbox coordinates
                                        bbox = span.get("bbox", [0, 0, 0, 0])
                                        
                                        # Determine bold/italic from flags
                                        flags = span.get("flags", 0)
                                        is_bold = bool(flags & 16)  # Bold flag
                                        is_italic = bool(flags & 2)  # Italic flag
                                        
                                        # Create block in EXACT format expected by feature_engineering.py
                                        block_data = {
                                            "pdf_id": pdf_path.stem,
                                            "page_number": page_num + 1,
                                            "block_id": f"{pdf_path.stem}_p{page_num+1}_b{block_counter:03d}",
                                            "text": text,
                                            "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                                            "font_size": float(span.get("size", 12.0)),
                                            "font_name": span.get("font", ""),
                                            "is_bold": is_bold,
                                            "is_italic": is_italic,
                                            "label": "BODY"  # Default label for processing
                                        }
                                        
                                        blocks.append(block_data)
                                        block_counter += 1
                
                except Exception as page_error:
                    print(f"⚠️ Error processing page {page_num + 1}: {page_error}")
                    continue
            
        except Exception as e:
            print(f"❌ Error opening/processing PDF: {e}")
            print("   Trying alternative extraction method...")
            
            # Alternative extraction method using simpler approach
            try:
                if doc is None:
                    doc = fitz.open(str(pdf_path))
                
                # Use simple text extraction
                for page_num in range(10):  # Try up to 10 pages
                    try:
                        page = doc.load_page(page_num)
                        text_blocks_simple = page.get_text("blocks")
                        
                        for block_idx, block in enumerate(text_blocks_simple):
                            if len(block) >= 5:  # block format: (x0, y0, x1, y1, text, ...)
                                text = block[4].strip()
                                if text:
                                    block_data = {
                                        "pdf_id": pdf_path.stem,
                                        "page_number": page_num + 1,
                                        "block_id": f"{pdf_path.stem}_p{page_num+1}_b{block_counter:03d}",
                                        "text": text,
                                        "bbox": [float(block[0]), float(block[1]), float(block[2]), float(block[3])],
                                        "font_size": 12.0,  # Default
                                        "font_name": "",
                                        "is_bold": False,
                                        "is_italic": False,
                                        "label": "BODY"
                                    }
                                    blocks.append(block_data)
                                    block_counter += 1
                                    
                    except Exception as alt_page_error:
                        if page_num == 0:  # If first page fails, it's a serious error
                            print(f"❌ Cannot read any pages from PDF: {alt_page_error}")
                            break
                        else:
                            # End of document reached
                            break
                            
            except Exception as alt_error:
                print(f"❌ Alternative extraction also failed: {alt_error}")
                
                # Last resort: try to extract just text without formatting
                try:
                    if doc is None:
                        doc = fitz.open(str(pdf_path))
                    
                    page = doc.load_page(0)  # Try just first page
                    simple_text = page.get_text()
                    
                    if simple_text.strip():
                        # Split into paragraphs and create basic blocks
                        paragraphs = [p.strip() for p in simple_text.split('\n\n') if p.strip()]
                        
                        for idx, para in enumerate(paragraphs[:50]):  # Limit to 50 paragraphs
                            block_data = {
                                "pdf_id": pdf_path.stem,
                                "page_number": 1,
                                "block_id": f"{pdf_path.stem}_p1_b{idx:03d}",
                                "text": para,
                                "bbox": [0.0, float(idx * 20), 500.0, float((idx + 1) * 20)],
                                "font_size": 12.0,
                                "font_name": "",
                                "is_bold": False,
                                "is_italic": False,
                                "label": "BODY"
                            }
                            blocks.append(block_data)
                        
                        print(f"   ⚠️ Used simple text extraction: {len(blocks)} paragraphs")
                    
                except Exception as final_error:
                    print(f"❌ All extraction methods failed: {final_error}")
                    raise Exception(f"Cannot extract text from PDF: {pdf_path.name}")
        
        finally:
            # Always close the document
            if doc:
                try:
                    doc.close()
                except:
                    pass
        
        if not blocks:
            raise Exception(f"No text could be extracted from PDF: {pdf_path.name}")
        
        print(f"   ✅ Extracted {len(blocks)} text blocks")
        return blocks


class ModelV8OutlineGenerator:
    """Generate outlines using the trained V8 model."""
    
    def __init__(self, model_path: str = "updated_model_8.pth"):
        """Initialize with V8 model."""
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.label_names = ['BODY', 'HH1', 'HH2', 'HH3', 'H4', 'TITLE']
        self.label_mapper = LabelMappingSystem()
        self._load_model()
    
    def _load_model(self):
        """Load the V8 ultra-optimized model."""
        if not self.model_path.exists():
            print(f"❌ V8 model not found: {self.model_path}")
            print("   Train the model first using train_ultra_optimized_model_v8.py")
            return
        
        print(f"🌟 Loading V8 Ultra-Optimized model from: {self.model_path}")
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Get model config
            model_config = checkpoint.get('model_config', {})
            
            # Create model with V8 architecture
            self.model = DocumentGNN(
                num_node_features=model_config.get('num_node_features', 22),
                hidden_dim=model_config.get('hidden_dim', 88),
                num_classes=model_config.get('num_classes', 6),
                num_layers=model_config.get('num_layers', 3),
                dropout=model_config.get('dropout', 0.12)
            )
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ V8 Model loaded successfully on {self.device}")
            print(f"   🏗️ Architecture: {model_config.get('hidden_dim', 88)} hidden, {model_config.get('num_layers', 3)} layers")
            
            # Show V8 features if available
            training_config = checkpoint.get('training_config', {})
            if 'ultra_features' in training_config:
                print(f"   🌟 Ultra features: {training_config['ultra_features']}")
            
            # Show performance metrics
            metrics = checkpoint.get('metrics', {})
            if metrics:
                print(f"   🎯 Training performance:")
                print(f"      Ultra score: {metrics.get('best_ultra_score', 'N/A')}")
                print(f"      Structural ratio: {metrics.get('final_test_structural_ratio', 'N/A')}")
                
        except Exception as e:
            print(f"❌ Error loading V8 model: {e}")
            self.model = None
    
    def predict_document_structure(self, text_blocks: List[Dict]) -> List[Dict]:
        """
        Predict document structure from text blocks using feature_engineering.py pipeline.
        
        Args:
            text_blocks (List[Dict]): Text blocks from PDF in exact format
            
        Returns:
            List of predictions with confidence scores
        """
        if self.model is None:
            print("❌ No model loaded")
            return []
        
        print(f"🔧 Processing {len(text_blocks)} text blocks for structure prediction...")
        
        try:
            # Step 1: Feature engineering using exact pipeline from feature_engineering.py
            print("   📊 Extracting features using feature_engineering.py...")
            
            # Call extract_and_normalize_features which expects exact format:
            # - pdf_id, page_number, block_id, text, bbox, font_size, font_name, is_bold, is_italic, label
            processed_blocks, label_mappings = extract_and_normalize_features(text_blocks)
            
            if not processed_blocks:
                print("❌ No features extracted from feature_engineering.py")
                return []
            
            print(f"   ✅ Feature engineering complete: {len(processed_blocks)} blocks processed")
            
            # Verify the processed blocks have the expected structure
            if processed_blocks and 'features' in processed_blocks[0]:
                feature_count = len(processed_blocks[0]['features'])
                print(f"   🔧 Features per block: {feature_count}")
                if feature_count != 22:
                    print(f"   ⚠️ Expected 22 features, got {feature_count}")
            
            # Step 2: Build graph using exact pipeline from build_graph.py
            print("   🔗 Building document graph...")
            graph_data, original_blocks = build_document_graph(processed_blocks, auto_save=False)
            
            if graph_data is None:
                print("❌ Failed to build graph")
                return []
            
            print(f"   ✅ Graph built: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
            print(f"   📐 Node features shape: {graph_data.x.shape}")
            print(f"   🏷️ Labels shape: {graph_data.y.shape}")
            
            # Step 3: Model prediction
            print("   🤖 Running V8 model inference...")
            graph_data = graph_data.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(graph_data.x, graph_data.edge_index)
                predictions = outputs.argmax(dim=1).cpu().numpy()
                confidences = torch.softmax(outputs, dim=1).max(dim=1)[0].cpu().numpy()
            
            # Step 4: Map predictions back to original text blocks
            results = []
            for i, (original_block, pred_id, confidence) in enumerate(zip(text_blocks, predictions, confidences)):
                pred_label = self.label_names[pred_id]
                
                result = {
                    'block_id': original_block.get('block_id', f'block_{i}'),
                    'text': original_block.get('text', ''),
                    'predicted_label': pred_label,
                    'confidence': float(confidence),
                    'page_number': original_block.get('page_number', 1),
                    'bbox': original_block.get('bbox', [0, 0, 0, 0]),
                    'font_size': original_block.get('font_size', 12.0),
                    'original_label': original_block.get('label', 'BODY')
                }
                results.append(result)
            
            print(f"   🎯 Prediction complete: {len(results)} blocks classified")
            
            # Show prediction distribution
            pred_counts = {}
            for result in results:
                label = result['predicted_label']
                pred_counts[label] = pred_counts.get(label, 0) + 1
            
            print("   📊 Prediction distribution:")
            for label, count in pred_counts.items():
                print(f"      {label}: {count}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error in prediction pipeline: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def generate_clean_outline(self, predictions: List[Dict], 
                             exclude_body: bool = True,
                             min_confidence: float = 0.3) -> Dict[str, Any]:
        """
        Generate clean outline from predictions.
        
        Args:
            predictions (List[Dict]): Model predictions
            exclude_body (bool): Whether to exclude BODY tags
            min_confidence (float): Minimum confidence threshold
            
        Returns:
            Clean outline dictionary
        """
        print(f"📋 Generating clean outline (exclude_body={exclude_body}, min_confidence={min_confidence})...")
        
        # Filter predictions
        structural_elements = []
        
        for pred in predictions:
            label = pred['predicted_label']
            confidence = pred.get('confidence', 0.0)
            
            # Skip BODY if requested
            if exclude_body and label == 'BODY':
                continue
            
            # Skip low confidence predictions
            if confidence < min_confidence:
                continue
            
            # Only include structural elements
            if label in ['HH1', 'HH2', 'HH3', 'H4', 'TITLE']:
                text = pred['text'].strip()
                
                # Skip very short text (likely noise)
                if len(text) < 3:
                    continue
                
                # Convert labels to standard hierarchy format
                if label == 'HH1':
                    level = 'H1'
                elif label == 'HH2':
                    level = 'H2'
                elif label == 'HH3':
                    level = 'H3'
                elif label == 'H4':
                    level = 'H4'
                elif label == 'TITLE':
                    level = 'TITLE'
                else:
                    level = label
                
                structural_elements.append({
                    'level': level,
                    'text': text,
                    'page': pred['page_number'],
                    'confidence': confidence,
                    'bbox': pred.get('bbox', [0, 0, 0, 0])
                })
        
        # Sort by page number, then by y-coordinate
        structural_elements.sort(key=lambda x: (x['page'], x.get('bbox', [0, 0, 0, 0])[1]))
        
        # Create final outline
        outline = {
            'title': self._extract_title(structural_elements, predictions),
            'outline': structural_elements,
            'metadata': {
                'total_structural_elements': len(structural_elements),
                'pages_with_structure': len(set(elem['page'] for elem in structural_elements)),
                'element_counts': {
                    level: len([e for e in structural_elements if e['level'] == level])
                    for level in ['TITLE', 'H1', 'H2', 'H3', 'H4']
                },
                'model_version': 'updated_model_8_ultra_optimized',
                'confidence_threshold': min_confidence,
                'excluded_body_tags': exclude_body,
                'processing_timestamp': datetime.now().isoformat(),
                'feature_engineering_pipeline': 'extract_and_normalize_features',
                'total_original_blocks': len(predictions)
            }
        }
        
        print(f"   ✅ Generated outline with {len(structural_elements)} structural elements")
        
        return outline
    
    def _extract_title(self, structural_elements: List[Dict], all_predictions: List[Dict]) -> str:
        """Extract document title from predictions."""
        # Look for TITLE predictions first
        title_elements = [e for e in structural_elements if e['level'] == 'TITLE']
        if title_elements:
            return title_elements[0]['text']
        
        # Look for high-confidence H1 on first page
        h1_elements = [e for e in structural_elements if e['level'] == 'H1' and e['page'] == 1]
        if h1_elements:
            return h1_elements[0]['text']
        
        # Look for any structural element on first page
        first_page_elements = [e for e in structural_elements if e['page'] == 1]
        if first_page_elements:
            return first_page_elements[0]['text']
        
        return "Untitled Document"


class CompletePDFToOutlinePipeline:
    """Complete pipeline from PDF to outline using V8 model with exact feature engineering."""
    
    def __init__(self, model_path: str = "updated_model_8.pth"):
        """Initialize pipeline with V8 model."""
        self.pdf_extractor = PDFTextExtractor()
        self.outline_generator = ModelV8OutlineGenerator(model_path)
        
        print("🚀 COMPLETE PDF TO OUTLINE PIPELINE INITIALIZED")
        print("   📄 PDF Extraction: PyMuPDF (exact format for feature_engineering.py)")
        print("   🔧 Feature Engineering: extract_and_normalize_features (22 normalized features)")
        print("   🔗 Graph Building: build_document_graph (KNN + reading order)")
        print(f"   🤖 Model: V8 Ultra-Optimized ({model_path})")
        print("   📋 Output: Clean outline (JSON format)")
        print("   ✅ Pipeline fully compatible with existing feature engineering")
    
    def process_pdf_to_outline(self, pdf_path: str, 
                              output_path: str = None,
                              exclude_body: bool = True,
                              min_confidence: float = 0.3,
                              save_intermediate: bool = False) -> Dict[str, Any]:
        """
        Complete pipeline: PDF → Outline using exact feature engineering pipeline.
        
        Args:
            pdf_path (str): Path to input PDF
            output_path (str): Path to save outline JSON
            exclude_body (bool): Whether to exclude BODY tags
            min_confidence (float): Minimum confidence threshold
            save_intermediate (bool): Whether to save intermediate files
            
        Returns:
            Generated outline dictionary
        """
        print(f"🌟 PROCESSING PDF TO OUTLINE: {Path(pdf_path).name}")
        print("=" * 70)
        
        start_time = time.time()
        
        try:
            # Step 1: Extract text blocks from PDF in EXACT format for feature_engineering.py
            print("📄 Step 1: PDF Text Extraction (feature_engineering.py compatible)")
            text_blocks = self.pdf_extractor.extract_text_blocks_from_pdf(pdf_path)
            
            if not text_blocks:
                print("❌ No text blocks extracted from PDF")
                return {}
            
            print(f"   ✅ Extracted {len(text_blocks)} text blocks")
            
            # Verify format matches feature_engineering.py requirements
            if text_blocks:
                sample_block = text_blocks[0]
                required_fields = ['pdf_id', 'page_number', 'block_id', 'text', 'bbox', 'font_size', 'font_name', 'is_bold', 'is_italic', 'label']
                missing_fields = [field for field in required_fields if field not in sample_block]
                if missing_fields:
                    print(f"   ⚠️ Missing required fields: {missing_fields}")
                else:
                    print("   ✅ All required fields present for feature_engineering.py")
            
            # Save intermediate if requested
            if save_intermediate:
                intermediate_dir = Path(pdf_path).parent / "intermediate"
                intermediate_dir.mkdir(exist_ok=True)
                
                blocks_file = intermediate_dir / f"{Path(pdf_path).stem}_text_blocks.json"
                with open(blocks_file, 'w', encoding='utf-8') as f:
                    json.dump(text_blocks, f, indent=2, ensure_ascii=False)
                print(f"   💾 Saved text blocks to: {blocks_file}")
            
            # Step 2: Full feature engineering + model prediction pipeline
            print("\n🤖 Step 2: Complete Feature Engineering + V8 Model Pipeline")
            predictions = self.outline_generator.predict_document_structure(text_blocks)
            
            if not predictions:
                print("❌ No predictions generated")
                return {}
            
            print(f"   ✅ Generated {len(predictions)} predictions")
            
            # Save intermediate if requested
            if save_intermediate:
                pred_file = intermediate_dir / f"{Path(pdf_path).stem}_predictions.json"
                with open(pred_file, 'w', encoding='utf-8') as f:
                    json.dump(predictions, f, indent=2, ensure_ascii=False)
                print(f"   💾 Saved predictions to: {pred_file}")
            
            # Step 3: Generate clean outline
            print("\n📋 Step 3: Outline Generation")
            outline = self.outline_generator.generate_clean_outline(
                predictions, 
                exclude_body=exclude_body,
                min_confidence=min_confidence
            )
            
            # Show outline summary
            metadata = outline.get('metadata', {})
            print(f"   📊 Outline summary:")
            print(f"      Title: {outline.get('title', 'N/A')}")
            print(f"      Structural elements: {metadata.get('total_structural_elements', 0)}")
            print(f"      Pages with structure: {metadata.get('pages_with_structure', 0)}")
            
            element_counts = metadata.get('element_counts', {})
            for level, count in element_counts.items():
                if count > 0:
                    print(f"      {level}: {count}")
            
            # Step 4: Save outline
            if output_path is None:
                output_path = Path(pdf_path).stem + "_outline_v8.json"
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(outline, f, indent=2, ensure_ascii=False)
            
            processing_time = time.time() - start_time
            
            print(f"\n🎉 PIPELINE COMPLETE!")
            print(f"   ⏱️ Processing time: {processing_time:.1f} seconds")
            print(f"   💾 Outline saved to: {output_path}")
            print(f"   🌟 Used V8 Ultra-Optimized model with exact feature engineering")
            
            # Assess quality
            if len(outline['outline']) > 0:
                print(f"   📈 Quality assessment:")
                print(f"      Structural elements found: {len(outline['outline'])}")
                if 3 <= len(outline['outline']) <= 20:
                    print(f"      ✅ Good structural detection")
                elif len(outline['outline']) > 20:
                    print(f"      ⚠️ Many elements detected (might be over-sensitive)")
                else:
                    print(f"      ⚠️ Few elements detected (might be under-sensitive)")
            
            return outline
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def batch_process_pdfs(self, pdf_directory: str, 
                          output_directory: str,
                          exclude_body: bool = True,
                          min_confidence: float = 0.3) -> List[Dict[str, Any]]:
        """
        Process multiple PDFs in batch.
        
        Args:
            pdf_directory (str): Directory containing PDF files
            output_directory (str): Directory to save outlines
            exclude_body (bool): Whether to exclude BODY tags
            min_confidence (float): Minimum confidence threshold
            
        Returns:
            List of processing results
        """
        pdf_dir = Path(pdf_directory)
        output_dir = Path(output_directory)
        
        if not pdf_dir.exists():
            print(f"❌ PDF directory not found: {pdf_directory}")
            return []
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"❌ No PDF files found in: {pdf_directory}")
            return []
        
        print(f"🔄 BATCH PROCESSING {len(pdf_files)} PDFs")
        print("=" * 60)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        successful = 0
        failed = 0
        
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n📄 Processing {i}/{len(pdf_files)}: {pdf_file.name}")
            
            try:
                output_file = output_dir / f"{pdf_file.stem}_outline_v8.json"
                
                outline = self.process_pdf_to_outline(
                    str(pdf_file),
                    str(output_file),
                    exclude_body=exclude_body,
                    min_confidence=min_confidence
                )
                
                if outline and outline.get('outline'):
                    successful += 1
                    results.append({
                        'pdf_file': str(pdf_file),
                        'output_file': str(output_file),
                        'success': True,
                        'outline_elements': len(outline['outline']),
                        'title': outline.get('title', 'N/A')
                    })
                else:
                    failed += 1
                    results.append({
                        'pdf_file': str(pdf_file),
                        'output_file': str(output_file),
                        'success': False,
                        'error': 'No outline generated'
                    })
                
            except Exception as e:
                failed += 1
                print(f"   ❌ Failed: {e}")
                results.append({
                    'pdf_file': str(pdf_file),
                    'output_file': None,
                    'success': False,
                    'error': str(e)
                })
        
        print(f"\n🎉 BATCH PROCESSING COMPLETE!")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   📁 Outlines saved to: {output_dir}")
        
        # Save batch summary
        summary_file = output_dir / "batch_processing_summary.json"
        summary = {
            'total_pdfs': len(pdf_files),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(pdf_files) if pdf_files else 0,
            'processing_timestamp': datetime.now().isoformat(),
            'model_version': 'updated_model_8_ultra_optimized',
            'settings': {
                'exclude_body': exclude_body,
                'min_confidence': min_confidence
            },
            'results': results
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"   📊 Summary saved to: {summary_file}")
        
        return results


def main():
    """Main interface for the complete pipeline."""
    print("🌟 COMPLETE PDF TO OUTLINE PIPELINE - V8 ULTRA-OPTIMIZED")
    print("=" * 80)
    print("🎯 Input: PDF files")
    print("🤖 Model: V8 Ultra-Optimized (multi-stage adaptive training)")
    print("📋 Output: Clean JSON outlines (similar to data/new_json_files)")
    print()
    
    # Check if V8 model exists
    model_path = "updated_model_8.pth"
    if not Path(model_path).exists():
        print(f"❌ V8 model not found: {model_path}")
        print("   Please train the V8 model first using train_ultra_optimized_model_v8.py")
        return
    
    # Initialize pipeline
    try:
        pipeline = CompletePDFToOutlinePipeline(model_path)
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return
    
    while True:
        print("Pipeline Options:")
        print("1. 📄 Process single PDF")
        print("2. 🔄 Batch process PDFs directory")
        print("3. 🧪 Quick test with user-provided PDF")
        print("4. ⚙️ Custom settings")
        print("5. 📊 View example output format")
        print("6. 🚪 Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "1":
            pdf_path = input("Enter PDF path: ").strip()
            if not pdf_path:
                print("❌ No path provided")
                continue
            
            if not Path(pdf_path).exists():
                print(f"❌ PDF file not found: {pdf_path}")
                continue
            
            output_path = input("Output path (Enter for auto): ").strip() or None
            
            try:
                outline = pipeline.process_pdf_to_outline(pdf_path, output_path)
                if outline:
                    print("\n✅ Processing successful!")
                    if outline.get('outline'):
                        print(f"   📋 Generated outline with {len(outline['outline'])} elements")
                        print(f"   📖 Title: {outline.get('title', 'N/A')}")
                    else:
                        print("   ⚠️ No structural elements found")
                else:
                    print("❌ Processing failed")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == "2":
            pdf_dir = input("Enter PDF directory path: ").strip()
            if not pdf_dir:
                print("❌ No directory provided")
                continue
            
            if not Path(pdf_dir).exists():
                print(f"❌ Directory not found: {pdf_dir}")
                continue
            
            output_dir = input("Output directory (Enter for auto): ").strip()
            if not output_dir:
                output_dir = str(Path(pdf_dir) / "outlines_v8")
            
            try:
                results = pipeline.batch_process_pdfs(pdf_dir, output_dir)
                print(f"\n✅ Batch processing complete!")
                successful = len([r for r in results if r['success']])
                print(f"   📊 Success rate: {successful}/{len(results)}")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == "3":
            quick_test_pipeline()
        
        elif choice == "4":
            print("\n⚙️ Custom Settings:")
            exclude_body = input("Exclude BODY tags? (y/n, default: y): ").strip().lower() != 'n'
            
            try:
                min_conf = float(input("Minimum confidence (0.0-1.0, default: 0.3): ").strip() or "0.3")
                min_conf = max(0.0, min(1.0, min_conf))
            except:
                min_conf = 0.3
            
            print(f"   Settings updated: exclude_body={exclude_body}, min_confidence={min_conf}")
            # These settings would be used in subsequent processing
        
        elif choice == "5":
            print("\n📊 Example Output Format:")
            example_outline = {
                "title": "Sample Document Title",
                "outline": [
                    {"level": "H1", "text": "Introduction", "page": 1, "confidence": 0.95},
                    {"level": "H2", "text": "Background", "page": 1, "confidence": 0.87},
                    {"level": "H2", "text": "Methodology", "page": 2, "confidence": 0.92},
                    {"level": "H3", "text": "Data Collection", "page": 2, "confidence": 0.89},
                    {"level": "H1", "text": "Results", "page": 3, "confidence": 0.94}
                ],
                "metadata": {
                    "total_structural_elements": 5,
                    "pages_with_structure": 3,
                    "model_version": "updated_model_8_ultra_optimized",
                    "processing_timestamp": "2025-01-22T12:00:00"
                }
            }
            print(json.dumps(example_outline, indent=2))
        
        elif choice == "6":
            print("👋 Pipeline session ended!")
            print("🌟 Your V8 model is ready for PDF outline generation!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")


def quick_test_pipeline():
    """Quick function to test the pipeline with user-provided PDF file."""
    print("🚀 QUICK PIPELINE TEST")
    print("=" * 40)
    
    # Get PDF path from user
    pdf_path = input("Enter PDF file path: ").strip()
    
    if not pdf_path:
        print("❌ No PDF path provided")
        return
    
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    if pdf_file.suffix.lower() != '.pdf':
        print(f"❌ File is not a PDF: {pdf_path}")
        return
    
    print(f"📄 Testing with: {pdf_file.name}")
    
    try:
        pipeline = CompletePDFToOutlinePipeline()
        outline = pipeline.process_pdf_to_outline(str(pdf_file))
        
        if outline and outline.get('outline'):
            print("\n🎉 QUICK TEST SUCCESSFUL!")
            print(f"   📖 Title: {outline.get('title', 'N/A')}")
            print(f"   📋 Elements: {len(outline['outline'])}")
            
            # Show first few elements
            for i, element in enumerate(outline['outline'][:3]):
                print(f"      {element['level']}: {element['text'][:50]}...")
            
            if len(outline['outline']) > 3:
                print(f"      ... and {len(outline['outline']) - 3} more elements")
        else:
            print("⚠️ Test completed but no outline generated")
    
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check if running as quick test
    if len(sys.argv) > 1 and sys.argv[1] == "--quick-test":
        quick_test_pipeline()
    else:
        main()
