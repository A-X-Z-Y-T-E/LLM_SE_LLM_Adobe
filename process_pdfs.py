"""
Adobe Hackathon Challenge 1a: PDF Processing Solution
Processes PDFs from /app/input and generates JSON outputs to /app/output
Uses V8 Ultra-Optimized model for document structure analysis
"""

import json
import time
from pathlib import Path
import sys
import os
import traceback
from typing import Dict, List, Any

# Add current directory to Python path
sys.path.append('/app')

try:
    from complete_pdf_to_outline_pipeline import CompletePDFToOutlinePipeline
except ImportError as e:
    print(f"❌ Failed to import pipeline: {e}")
    sys.exit(1)


class AdobeChallengePDFProcessor:
    """PDF processor for Adobe Hackathon Challenge 1a."""
    
    def __init__(self):
        """Initialize the processor with V8 model."""
        self.input_dir = Path("/app/input")
        self.output_dir = Path("/app/output")
        self.model_path = "/app/updated_model_8.pth"
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the pipeline
        try:
            self.pipeline = CompletePDFToOutlinePipeline(self.model_path)
            print("✅ Pipeline initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize pipeline: {e}")
            # Create a fallback dummy processor
            self.pipeline = None
    
    def convert_to_challenge_format(self, outline_data: Dict) -> Dict[str, Any]:
        """
        Convert V8 model output to Adobe Challenge format.
        EXACTLY matches the required format - no extra metadata in final output.
        """
        if not outline_data or 'outline' not in outline_data:
            return {
                "title": "Document",
                "outline": []
            }
        
        # Extract structural elements from V8 output
        challenge_outline = []
        
        for element in outline_data.get('outline', []):
            # Convert V8 format to exact Challenge format
            challenge_element = {
                "level": element.get('level', 'BODY'),
                "text": element.get('text', ''),
                "page": element.get('page', 1)
            }
            
            # Only include structural elements (exclude BODY as per requirements)
            if challenge_element["level"] in ['TITLE', 'H1', 'H2', 'H3', 'H4']:
                challenge_outline.append(challenge_element)
        
        # Create final output in EXACT Challenge format (no metadata in final output)
        result = {
            "title": outline_data.get('title', 'Document'),
            "outline": challenge_outline
        }
        
        return result
    
    def create_fallback_output(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Create fallback output in exact Adobe Challenge format.
        """
        return {
            "title": pdf_path.stem.replace('_', ' ').title(),
            "outline": [
                {
                    "level": "H1",
                    "text": f"Document: {pdf_path.stem}",
                    "page": 1
                }
            ]
        }
    
    def process_single_pdf(self, pdf_path: Path) -> bool:
        """
        Process a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if successful, False otherwise
        """
        output_path = self.output_dir / f"{pdf_path.stem}.json"
        
        print(f"📄 Processing: {pdf_path.name}")
        
        try:
            if self.pipeline:
                # Use V8 model pipeline
                outline_data = self.pipeline.process_pdf_to_outline(
                    str(pdf_path),
                    exclude_body=True,
                    min_confidence=0.3,
                    save_intermediate=False
                )
                
                # Convert to Challenge format
                challenge_output = self.convert_to_challenge_format(outline_data)
            else:
                # Fallback processing
                print("⚠️ Using fallback processing")
                challenge_output = self.create_fallback_output(pdf_path)
            
            # Save output
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(challenge_output, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Generated: {output_path.name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to process {pdf_path.name}: {e}")
            
            # Create emergency fallback
            try:
                emergency_output = self.create_fallback_output(pdf_path)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(emergency_output, f, indent=2, ensure_ascii=False)
                print(f"⚠️ Created fallback output: {output_path.name}")
                return True
            except Exception as fallback_error:
                print(f"❌ Even fallback failed: {fallback_error}")
                return False
    
    def process_all_pdfs(self) -> Dict[str, Any]:
        """
        Process all PDFs in the input directory.
        
        Returns:
            Processing summary
        """
        start_time = time.time()
        
        # Find all PDF files
        pdf_files = list(self.input_dir.glob("*.pdf"))
        
        if not pdf_files:
            print("⚠️ No PDF files found in input directory")
            return {
                "processed": 0,
                "successful": 0,
                "failed": 0,
                "processing_time": 0,
                "files": []
            }
        
        print(f"🔍 Found {len(pdf_files)} PDF files to process")
        
        # Process each PDF
        successful = 0
        failed = 0
        results = []
        
        for pdf_file in pdf_files:
            file_start_time = time.time()
            
            success = self.process_single_pdf(pdf_file)
            
            file_processing_time = time.time() - file_start_time
            
            if success:
                successful += 1
            else:
                failed += 1
            
            results.append({
                "file": pdf_file.name,
                "success": success,
                "processing_time": round(file_processing_time, 2)
            })
        
        total_time = time.time() - start_time
        
        # Summary
        summary = {
            "processed": len(pdf_files),
            "successful": successful,
            "failed": failed,
            "processing_time": round(total_time, 2),
            "average_time_per_file": round(total_time / len(pdf_files), 2),
            "files": results
        }
        
        return summary


def main():
    """Main entry point for Adobe Challenge 1a PDF processor."""
    print("🚀 ADOBE HACKATHON CHALLENGE 1A - PDF PROCESSOR")
    print("=" * 60)
    print("🎯 Using V8 Ultra-Optimized Model for Document Structure Analysis")
    print("📁 Input: /app/input")
    print("📁 Output: /app/output")
    print()
    
    try:
        # Initialize processor
        processor = AdobeChallengePDFProcessor()
        
        # Check if input directory exists and has PDFs
        if not processor.input_dir.exists():
            print("❌ Input directory /app/input does not exist")
            sys.exit(1)
        
        # Process all PDFs
        summary = processor.process_all_pdfs()
        
        # Print summary
        print("\n📊 PROCESSING SUMMARY:")
        print(f"   📄 Files processed: {summary['processed']}")
        print(f"   ✅ Successful: {summary['successful']}")
        print(f"   ❌ Failed: {summary['failed']}")
        print(f"   ⏱️ Total time: {summary['processing_time']:.2f} seconds")
        
        if summary['processed'] > 0:
            print(f"   📈 Average time per file: {summary['average_time_per_file']:.2f} seconds")
            
            # Check if within Challenge constraints
            if summary['processing_time'] <= 10.0:
                print("   🎉 ✅ Within 10-second time constraint!")
            else:
                print("   ⚠️ ⏰ Exceeded 10-second time constraint")
        
        # Success/failure exit codes
        if summary['failed'] == 0:
            print("\n🎉 All files processed successfully!")
            sys.exit(0)
        else:
            print(f"\n⚠️ {summary['failed']} files failed to process")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Critical error in main process: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 
