import os
import json
import base64
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from pathlib import Path
import fitz  # PyMuPDF
from groq import Groq  # Changed from openai to groq
from google.cloud import vision
from datasets import Dataset
import pandas as pd
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time
from dotenv import load_dotenv
import os

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HeadingInfo:
    level: str
    text: str
    page: int

class PDFOutlineExtractor:
    def __init__(self, groq_api_key: str, google_credentials_path: str):
        """
        Initialize the PDF outline extractor with API credentials
        
        Args:
            groq_api_key: Groq API key
            google_credentials_path: Path to Google Cloud Vision API credentials JSON
        """
        # Groq setup
        self.groq_client = Groq(api_key=groq_api_key)
        
        # Google Vision setup (with error handling)
        self.vision_client = None
        self.google_vision_enabled = False
        
        try:
            if google_credentials_path and google_credentials_path != "dummy_path" and os.path.exists(google_credentials_path):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_credentials_path
                self.vision_client = vision.ImageAnnotatorClient()
                self.google_vision_enabled = True
                logger.info("Google Vision API initialized successfully")
            else:
                logger.warning("Google Vision API not initialized - credentials path not provided or invalid")
        except Exception as e:
            logger.warning(f"Failed to initialize Google Vision API: {str(e)}")
            logger.info("Continuing with Groq and heuristic methods only")
        
        # Hugging Face will be used for hosting the dataset
        
    def extract_pdf_pages_as_images(self, pdf_path: str) -> List[bytes]:
        """Extract PDF pages as images for Google Vision API"""
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # 2x zoom
            img_data = pix.tobytes("png")
            images.append(img_data)
        
        doc.close()
        return images
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """Extract text from each page of the PDF"""
        doc = fitz.open(pdf_path)
        pages_text = {}
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            pages_text[page_num + 1] = text  # 1-indexed pages
        
        doc.close()
        return pages_text
    
    async def extract_with_groq(self, pdf_path: str) -> Dict[str, Any]:
        """Extract outline using Groq API"""
        try:
            pages_text = self.extract_text_from_pdf(pdf_path)
            
            # Combine first few pages for title extraction
            combined_text = ""
            for page_num in sorted(pages_text.keys())[:5]:  # First 5 pages
                combined_text += f"Page {page_num}:\n{pages_text[page_num]}\n\n"
            
            prompt = f"""
            Analyze this PDF content and extract the document structure in the following JSON format:
            {{
                "title": "Document Title",
                "outline": [
                    {{"level": "H1", "text": "Main Heading", "page": 1}},
                    {{"level": "H2", "text": "Sub Heading", "page": 2}},
                    {{"level": "H3", "text": "Sub-sub Heading", "page": 3}}
                ]
            }}
            
            Rules:
            1. Extract the main document title
            2. Identify headings at H1, H2, and H3 levels based on formatting, font size, and context
            3. Include the page number where each heading appears
            4. Maintain hierarchical structure (H1 > H2 > H3)
            5. Only include actual headings, not regular text
            
            PDF Content:
            {combined_text[:8000]}  # Limit to avoid token limits
            
            Return only valid JSON:
            """
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Updated to current available model
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=4096
            )
            
            result_text = response.choices[0].message.content.strip()
            # Clean up the response to extract JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1]
            
            return json.loads(result_text)
            
        except Exception as e:
            logger.error(f"Groq extraction failed for {pdf_path}: {str(e)}")
            return {"title": "Unknown", "outline": []}
    
    async def extract_with_google_vision(self, pdf_path: str) -> Dict[str, Any]:
        """Extract outline using Google Cloud Vision API"""
        if not self.google_vision_enabled or not self.vision_client:
            logger.info(f"Google Vision API not available for {pdf_path}, skipping")
            return {"title": "Unknown", "outline": []}
            
        try:
            images = self.extract_pdf_pages_as_images(pdf_path)
            all_text_blocks = []
            
            for page_num, image_data in enumerate(images[:10], 1):  # Limit to first 10 pages
                image = vision.Image(content=image_data)
                response = self.vision_client.document_text_detection(image=image)
                
                if response.full_text_annotation:
                    # Analyze text blocks for heading detection
                    for page in response.full_text_annotation.pages:
                        for block in page.blocks:
                            block_text = ""
                            avg_font_size = 0
                            font_sizes = []
                            
                            for paragraph in block.paragraphs:
                                for word in paragraph.words:
                                    word_text = ''.join([symbol.text for symbol in word.symbols])
                                    block_text += word_text + " "
                                    
                                    # Get font size if available
                                    if word.symbols and hasattr(word.symbols[0], 'property'):
                                        if hasattr(word.symbols[0].property, 'detected_break'):
                                            # This is a simplified approach - Vision API structure varies
                                            pass
                            
                            if block_text.strip():
                                all_text_blocks.append({
                                    "text": block_text.strip(),
                                    "page": page_num,
                                    "confidence": block.confidence if hasattr(block, 'confidence') else 0.8
                                })
            
            # Use heuristics to identify headings
            outline = self._identify_headings_from_blocks(all_text_blocks)
            title = self._extract_title_from_blocks(all_text_blocks)
            
            return {"title": title, "outline": outline}
            
        except Exception as e:
            logger.error(f"Google Vision extraction failed for {pdf_path}: {str(e)}")
            return {"title": "Unknown", "outline": []}
    
    def _identify_headings_from_blocks(self, text_blocks: List[Dict]) -> List[Dict]:
        """Identify headings using heuristics"""
        headings = []
        
        for block in text_blocks:
            text = block["text"]
            
            # Heuristics for heading detection
            is_heading = (
                len(text) < 100 and  # Short text
                not text.endswith('.') and  # Doesn't end with period
                len(text.split()) <= 10 and  # Not too many words
                any(c.isupper() for c in text) and  # Contains uppercase
                text.strip() != ""
            )
            
            if is_heading:
                # Determine heading level based on position and characteristics
                level = "H1"
                if block["page"] > 1:
                    if len(text) < 30:
                        level = "H3"
                    elif len(text) < 60:
                        level = "H2"
                
                headings.append({
                    "level": level,
                    "text": text,
                    "page": block["page"]
                })
        
        return headings[:20]  # Limit number of headings
    
    def _extract_title_from_blocks(self, text_blocks: List[Dict]) -> str:
        """Extract document title from first page blocks"""
        if not text_blocks:
            return "Unknown Document"
        
        # Find blocks on first page
        first_page_blocks = [b for b in text_blocks if b["page"] == 1]
        
        if first_page_blocks:
            # Return the first substantial text block as title
            for block in first_page_blocks:
                if len(block["text"]) > 10 and len(block["text"]) < 200:
                    return block["text"]
        
        return "Unknown Document"
    
    async def extract_with_fallback_heuristics(self, pdf_path: str) -> Dict[str, Any]:
        """Fallback method using PyMuPDF heuristics"""
        try:
            doc = fitz.open(pdf_path)
            outline = []
            title = "Unknown Document"
            
            # Try to get title from metadata or first page
            metadata = doc.metadata
            if metadata.get("title"):
                title = metadata["title"]
            else:
                # Extract from first page
                first_page = doc.load_page(0)
                first_page_text = first_page.get_text()
                lines = first_page_text.split('\n')
                for line in lines:
                    if line.strip() and len(line) > 10 and len(line) < 200:
                        title = line.strip()
                        break
            
            # Extract headings using font size analysis
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                font_size = span["size"]
                                font_flags = span["flags"]
                                
                                # Heading heuristics
                                is_bold = font_flags & 2**4  # Bold flag
                                is_potential_heading = (
                                    len(text) > 5 and
                                    len(text) < 100 and
                                    not text.endswith('.') and
                                    (font_size > 12 or is_bold)
                                )
                                
                                if is_potential_heading:
                                    # Determine level based on font size
                                    level = "H1"
                                    if font_size >= 18:
                                        level = "H1"
                                    elif font_size >= 14:
                                        level = "H2"
                                    else:
                                        level = "H3"
                                    
                                    outline.append({
                                        "level": level,
                                        "text": text,
                                        "page": page_num + 1
                                    })
            
            doc.close()
            
            # Remove duplicates and limit
            seen = set()
            unique_outline = []
            for item in outline:
                if item["text"] not in seen:
                    seen.add(item["text"])
                    unique_outline.append(item)
            
            return {"title": title, "outline": unique_outline[:15]}
            
        except Exception as e:
            logger.error(f"Fallback extraction failed for {pdf_path}: {str(e)}")
            return {"title": "Unknown Document", "outline": []}
    
    async def process_single_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF with all methods and create consensus"""
        logger.info(f"Processing {pdf_path}")
        
        # Run all extraction methods
        tasks = [
            self.extract_with_groq(pdf_path),
            self.extract_with_google_vision(pdf_path),
            self.extract_with_fallback_heuristics(pdf_path)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Create consensus result
        consensus = self._create_consensus(results)
        
        return {
            "pdf_path": pdf_path,
            "groq_result": results[0] if not isinstance(results[0], Exception) else None,
            "google_vision_result": results[1] if not isinstance(results[1], Exception) else None,
            "heuristic_result": results[2] if not isinstance(results[2], Exception) else None,
            "consensus_result": consensus
        }
    
    def _create_consensus(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create consensus from multiple extraction results"""
        valid_results = [r for r in results if isinstance(r, dict) and "outline" in r]
        
        if not valid_results:
            return {"title": "Unknown Document", "outline": []}
        
        # Take title from first valid result
        title = next((r["title"] for r in valid_results if r["title"] != "Unknown"), "Unknown Document")
        
        # Merge outlines (simplified approach - in practice, you'd want more sophisticated merging)
        all_headings = []
        for result in valid_results:
            all_headings.extend(result["outline"])
        
        # Remove duplicates based on text similarity
        unique_headings = []
        seen_texts = set()
        
        for heading in all_headings:
            text_lower = heading["text"].lower().strip()
            if text_lower not in seen_texts and len(text_lower) > 3:
                seen_texts.add(text_lower)
                unique_headings.append(heading)
        
        return {"title": title, "outline": unique_headings[:10]}  # Limit to 10 headings

class DatasetCreator:
    def __init__(self, extractor: PDFOutlineExtractor):
        self.extractor = extractor
    
    async def create_dataset(self, pdf_directory: str, output_directory: str) -> str:
        """Create training dataset from PDF directory"""
        pdf_files = list(Path(pdf_directory).glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        os.makedirs(output_directory, exist_ok=True)
        
        all_results = []
        
        # Process PDFs in batches to avoid overwhelming APIs
        batch_size = 5
        for i in range(0, len(pdf_files), batch_size):
            batch = pdf_files[i:i+batch_size]
            batch_tasks = [self.extractor.process_single_pdf(str(pdf_path)) for pdf_path in batch]
            
            batch_results = await asyncio.gather(*batch_tasks)
            all_results.extend(batch_results)
            
            # Add delay between batches to respect API limits
            if i + batch_size < len(pdf_files):
                await asyncio.sleep(2)
        
        # Save individual JSON files
        dataset_records = []
        
        for result in all_results:
            pdf_path = Path(result["pdf_path"])
            json_filename = pdf_path.stem + ".json"
            json_path = Path(output_directory) / json_filename
            
            # Save consensus result as ground truth
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result["consensus_result"], f, indent=2, ensure_ascii=False)
            
            # Create dataset record
            dataset_records.append({
                "pdf_filename": pdf_path.name,
                "json_filename": json_filename,
                "title": result["consensus_result"]["title"],
                "num_headings": len(result["consensus_result"]["outline"]),
                "groq_available": result["groq_result"] is not None,
                "google_vision_available": result["google_vision_result"] is not None,
                "heuristic_available": result["heuristic_result"] is not None,
                "extraction_methods": [
                    method for method, available in [
                        ("groq", result["groq_result"] is not None),
                        ("google_vision", result["google_vision_result"] is not None),
                        ("heuristic", result["heuristic_result"] is not None)
                    ] if available
                ]
            })
        
        # Create and save Hugging Face dataset
        df = pd.DataFrame(dataset_records)
        dataset = Dataset.from_pandas(df)
        
        dataset_path = Path(output_directory) / "huggingface_dataset"
        dataset.save_to_disk(str(dataset_path))
        
        # Save summary statistics
        summary = {
            "total_documents": len(all_results),
            "successful_extractions": len([r for r in all_results if r["consensus_result"]["outline"]]),
            "average_headings_per_doc": sum(len(r["consensus_result"]["outline"]) for r in all_results) / len(all_results),
            "extraction_method_success_rates": {
                "groq": len([r for r in all_results if r["groq_result"]]) / len(all_results),
                "google_vision": len([r for r in all_results if r["google_vision_result"]]) / len(all_results),
                "heuristic": len([r for r in all_results if r["heuristic_result"]]) / len(all_results)
            }
        }
        
        with open(Path(output_directory) / "dataset_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Dataset created successfully in {output_directory}")
        return str(dataset_path)

# Usage example
async def main():
    # Initialize the extractor
    extractor = PDFOutlineExtractor(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        google_credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    )
    
    # Create dataset
    dataset_creator = DatasetCreator(extractor)

    pdf_directory = "D:\\VS_CODE\\Adobe\\LLM_SE_LLM_Adobe\\training data\\Pdf"
    output_directory = "D:\\VS_CODE\\Adobe\\LLM_SE_LLM_Adobe\\training data\\new_json_files"

    dataset_path = await dataset_creator.create_dataset(pdf_directory, output_directory)
    print(f"Dataset saved to: {dataset_path}")

if __name__ == "__main__":
    asyncio.run(main())