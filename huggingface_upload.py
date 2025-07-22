
import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from datasets import Dataset, DatasetDict, Features, Value, Sequence
import pandas as pd

class HuggingFaceDatasetUploader:
    def __init__(self, token: str):
        """
        Initialize uploader with Hugging Face token
        
        Args:
            token: Hugging Face API token
        """
        self.api = HfApi(token=token)
        self.token = token
    
    def create_structured_dataset(self, dataset_directory: str) -> DatasetDict:
        """Create a structured dataset from the generated files"""
        
        dataset_dir = Path(dataset_directory)
        pdf_files = list(dataset_dir.glob("*.pdf"))
        json_files = list(dataset_dir.glob("*.json"))
        
        # Filter out summary files
        json_files = [f for f in json_files if f.name != "dataset_summary.json"]
        
        records = []
        
        for json_file in json_files:
            pdf_file = dataset_dir / (json_file.stem + ".pdf")
            
            if pdf_file.exists():
                with open(json_file, 'r', encoding='utf-8') as f:
                    outline_data = json.load(f)
                
                record = {
                    "pdf_filename": pdf_file.name,
                    "title": outline_data.get("title", ""),
                    "outline": outline_data.get("outline", []),
                    "num_headings": len(outline_data.get("outline", [])),
                    "heading_levels": list(set([h.get("level", "") for h in outline_data.get("outline", [])])),
                    "page_range": {
                        "min_page": min([h.get("page", 1) for h in outline_data.get("outline", [])], default=1),
                        "max_page": max([h.get("page", 1) for h in outline_data.get("outline", [])], default=1)
                    }
                }
                records.append(record)
        
        df = pd.DataFrame(records)
        
        # Define features schema for better dataset structure
        features = Features({
            "pdf_filename": Value("string"),
            "title": Value("string"),
            "outline": Sequence({
                "level": Value("string"),
                "text": Value("string"),
                "page": Value("int32")
            }),
            "num_headings": Value("int32"),
            "heading_levels": Sequence(Value("string")),
            "page_range": {
                "min_page": Value("int32"),
                "max_page": Value("int32")
            }
        })
        
        dataset = Dataset.from_pandas(df, features=features)
        
        # Split dataset (80% train, 20% test)
        split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
        
        return DatasetDict({
            "train": split_dataset["train"],
            "test": split_dataset["test"]
        })
    
    def upload_dataset(self, 
                      dataset_directory: str, 
                      repo_name: str, 
                      private: bool = False) -> str:
        """Upload dataset to Hugging Face Hub"""
        
        # Create structured dataset
        dataset_dict = self.create_structured_dataset(dataset_directory)
        
        # Create repository
        repo_url = create_repo(
            repo_name, 
            token=self.token, 
            repo_type="dataset",
            private=private
        )
        
        # Upload dataset
        dataset_dict.push_to_hub(
            repo_name,
            token=self.token,
            private=private
        )
        
        # Create and upload README
        readme_content = self._generate_readme(dataset_dict, dataset_directory)
        
        with open("README.md", "w") as f:
            f.write(readme_content)
        
        self.api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="dataset",
            token=self.token
        )
        
        os.remove("README.md")  # Clean up
        
        return repo_url
    
    def _generate_readme(self, dataset_dict: DatasetDict, dataset_directory: str) -> str:
        """Generate README for the dataset"""
        
        # Load summary if available
        summary_path = Path(dataset_directory) / "dataset_summary.json"
        summary = {}
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
        
        train_size = len(dataset_dict["train"])
        test_size = len(dataset_dict["test"])
        
        readme = f"""---
license: mit
task_categories:
- text-extraction
- document-parsing
language:
- en
tags:
- pdf
- document-structure
- heading-extraction
- outline-extraction
size_categories:
- 1K<n<10K
---

# PDF Outline Extraction Dataset

## Dataset Description

This dataset contains PDF documents paired with their extracted structured outlines, designed for training models to understand document structure and extract hierarchical headings (Title, H1, H2, H3) from PDF files.

## Dataset Statistics

- **Total Documents**: {train_size + test_size}
- **Training Set**: {train_size} documents
- **Test Set**: {test_size} documents
- **Average Headings per Document**: {summary.get('average_headings_per_doc', 'N/A')}

## Data Structure

Each example contains:

- `pdf_filename`: Name of the source PDF file
- `title`: Main document title
- `outline`: List of headings with structure:
  - `level`: Heading level (H1, H2, H3)
  - `text`: Heading text content
  - `page`: Page number where heading appears
- `num_headings`: Total number of headings extracted
- `heading_levels`: List of heading levels present in document
- `page_range`: Min and max page numbers with headings

## Example

```json
{{
  "pdf_filename": "sample_document.pdf",
  "title": "Machine Learning Fundamentals",
  "outline": [
    {{"level": "H1", "text": "Introduction", "page": 1}},
    {{"level": "H2", "text": "What is Machine Learning?", "page": 2}},
    {{"level": "H3", "text": "Types of Learning", "page": 3}}
  ],
  "num_headings": 3,
  "heading_levels": ["H1", "H2", "H3"],
  "page_range": {{"min_page": 1, "max_page": 3}}
}}
```

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("your-username/{repo_name.split('/')[-1]}")

# Access training data
train_data = dataset["train"]
test_data = dataset["test"]

# Example usage
for example in train_data:
    print(f"Document: {{example['pdf_filename']}}")
    print(f"Title: {{example['title']}}")
    print(f"Headings: {{len(example['outline'])}}")
```

## Dataset Creation

This dataset was created using multiple extraction methods:
- OpenAI GPT-4 Vision API
- Google Cloud Vision API  
- Heuristic-based extraction using PyMuPDF

The final ground truth represents a consensus from these methods to ensure quality and accuracy.

## Intended Use

This dataset is designed for:
- Training document structure understanding models
- Developing PDF parsing systems
- Research in document AI and information extraction
- Benchmarking heading extraction algorithms

## Limitations

- Currently focused on English-language documents
- Limited to documents with clear hierarchical structure
- May not perform well on highly visual or complex layout documents

## Citation

If you use this dataset, please cite:

```
@dataset{{pdf_outline_extraction_dataset,
  title={{PDF Outline Extraction Dataset}},
  author={{Your Name}},
  year={{2025}},
  url={{https://huggingface.co/datasets/{repo_name.split('/')[-1] if '/' in repo_name else repo_name}}}
}}
```
"""
        return readme

# Usage example
def upload_example():
    """Example of how to upload the dataset"""
    
    # Initialize uploader
    uploader = HuggingFaceDatasetUploader(token="your-huggingface-token")
    
    # Upload dataset
    repo_url = uploader.upload_dataset(
        dataset_directory="path/to/your/dataset/output",
        repo_name="your-username/pdf-outline-extraction-dataset",
        private=False  # Set to True for private dataset
    )
    
    print(f"Dataset uploaded successfully: {repo_url}")

if __name__ == "__main__":
    upload_example()