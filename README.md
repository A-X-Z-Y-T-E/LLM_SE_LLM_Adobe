# Adobe Hackathon Challenge 1A: PDF Outline Extraction

## Overview
Advanced PDF outline extraction using a custom-trained Graph Neural Network (GNN) that achieves high accuracy on document structure analysis.

## Approach
- **Feature Engineering**: 22 normalized features (geometric, stylistic, textual, contextual)
- **Graph Neural Network**: 3-layer DocumentGNN with spatial KNN + reading order edges
- **V8 Ultra-Optimized Model**: Multi-stage adaptive training with breakthrough detection
- **Robust PDF Processing**: Multiple fallback methods for text extraction

## Models & Libraries Used
- **PyTorch 2.5.0** (CPU-only): Deep learning framework
- **PyTorch Geometric 2.5.0**: Graph neural network operations
- **PyMuPDF 1.23.17**: PDF text extraction with error handling
- **Scikit-learn 1.3.2**: Feature normalization and KNN
- **Custom DocumentGNN**: 3-layer GNN (88 hidden dim, 22 input features, 6 output classes)

## Model Details
- **Architecture**: DocumentGNN with 3 GCN layers
- **Input**: 22 engineered features per text block
- **Output**: 6 classes (BODY, HH1→H1, HH2→H2, HH3→H3, H4, TITLE)
- **Size**: <200MB (compliant)
- **Training**: 964 document graphs with ultra-adaptive loss

## Build & Run

### Build
```bash
docker build --platform linux/amd64 -t pdf-processor .
```

### Run
```bash
docker run --rm -v $(pwd)/input:/app/input:ro -v $(pwd)/output:/app/output --network none pdf-processor
```

## Performance
- **Speed**: <10 seconds for 50-page PDFs
- **Accuracy**: High precision/recall on heading detection
- **Resources**: CPU-only, 8 cores, 16GB RAM compatible
- **Offline**: No internet required during execution

## Technical Highlights
- **22 Engineered Features**: Comprehensive text block characterization
- **Spatial Graph Construction**: KNN + reading order relationships
- **Multi-stage Training**: Conservative start + aggressive adaptation
- **Robust Error Handling**: Multiple PDF extraction fallbacks
- **Schema Compliance**: Exact Adobe Challenge JSON format

---

**Adobe India Hackathon 2025 - Challenge 1A Solution**
```bash
docker run --rm -v $(pwd)/input:/app/input:ro -v $(pwd)/output:/app/output --network none pdf-processor
```

## Performance Specifications

### Constraint Compliance
- ✅ **Execution Time**: <10 seconds for 50-page PDF
- ✅ **Model Size**: <200MB
- ✅ **Network**: No internet access during runtime
- ✅ **Platform**: AMD64 CPU execution
- ✅ **Resources**: 8 CPUs, 16GB RAM compatible
- ✅ **Libraries**: 100% open source

### Dependencies
- Python 3.10
- PyTorch 2.1.0 (CPU)
- PyTorch Geometric 2.4.0
- PyMuPDF 1.23.8
- NumPy 1.24.3
- Scikit-learn 1.3.0

## Usage

### Input Format
- Place PDF files in `/app/input` directory
- Files must have `.pdf` extension
- Read-only access (as per Challenge requirements)

### Output Format
- JSON files generated in `/app/output` directory
- One `filename.json` for each `filename.pdf`
- Schema-compliant structure with title, outline, and metadata

### Example Output
```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Introduction",
      "page": 1,
      "confidence": 0.95
    },
    {
      "level": "H2", 
      "text": "Background",
      "page": 1,
      "confidence": 0.87
    }
  ],
  "metadata": {
    "processing_method": "V8_ultra_optimized_model",
    "total_elements": 15,
    "pages_processed": 5
  }
}
```

## Model Training Pipeline

### Training Data
- 964 document graphs (674 train, 144 val, 146 test)
- Multi-stage adaptive training with UltraAdaptiveLoss
- Conservative start + aggressive adaptation strategy

### Training Features
- Real-time parameter adjustment
- Intelligent weight scheduling
- Breakthrough structural element detection (80-120% target ratio)
- Multi-document cross-validation

## Technical Implementation

### PDF Processing Pipeline
1. **Text Extraction**: PyMuPDF with multiple fallback methods
2. **Feature Engineering**: 22-feature normalization per text block
3. **Graph Construction**: Spatial KNN + reading order edges
4. **Model Inference**: V8 DocumentGNN classification
5. **Output Generation**: JSON formatting with confidence scores

### Error Handling
- Robust PDF parsing with multiple extraction methods
- Graceful degradation for problematic PDFs
- Fallback outputs maintain JSON schema compliance
- Comprehensive error logging and recovery

### Performance Optimizations
- Efficient memory management for large PDFs
- CPU-optimized inference (no GPU required)
- Batch processing capabilities
- Resource-constrained execution

## Testing and Validation

### Test Coverage
- Simple PDFs: Basic text documents
- Complex PDFs: Multi-column layouts, tables, images
- Large PDFs: 50+ page documents
- Edge Cases: Corrupted or unusual PDF formats

### Quality Metrics
- Structural detection accuracy: 80-120% target ratio
- Processing speed: <10 seconds per 50-page PDF
- Memory efficiency: <16GB RAM usage
- Model confidence: Weighted scoring system

## File Structure
```
Challenge_1a/
├── Dockerfile                           # Container configuration
├── requirements.txt                     # Python dependencies
├── process_pdfs.py                     # Main processing script
├── complete_pdf_to_outline_pipeline.py # V8 model pipeline
├── updated_model_8.pth                 # Trained V8 model
├── extractor/                          # Feature engineering
├── model_training/                     # GNN model components
├── utils/                              # Utility functions
├── data/                               # Configuration data
└── README.md                           # This documentation
```

## Innovation Highlights

### V8 Ultra-Optimization
- Multi-stage adaptive training approach
- Real-time loss parameter adjustment
- Breakthrough structural detection targeting
- Conservative start + aggressive adaptation strategy

### Advanced Feature Engineering
- 22 comprehensive features covering geometric, stylistic, textual, and contextual aspects
- Normalized feature space for robust cross-document performance
- Context-aware block relationship modeling

### Graph Neural Network Architecture
- Custom DocumentGNN designed for document layout analysis
- Spatial KNN edges + reading order relationships
- 3-layer architecture optimized for structural hierarchy detection

## License and Compliance
- 100% open source libraries and frameworks
- No proprietary or licensed components
- Full compliance with Adobe Hackathon guidelines
- Containerized for consistent cross-platform execution

---

**Adobe India Hackathon 2025 - Challenge 1a Solution**  
**Team**: Document Structure Analysis using V8 Ultra-Optimized GNN
