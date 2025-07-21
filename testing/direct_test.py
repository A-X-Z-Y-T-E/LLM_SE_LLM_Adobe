import fitz
import json

# Test file01.pdf directly
pdf_path = r"d:\VS_CODE\Adobe\PS\Adobe-India-Hackathon25\Challenge_1a\sample_dataset\pdfs\file01.pdf"

print("Testing PDF processing...")

try:
    # Open PDF
    doc = fitz.open(pdf_path)
    print(f"PDF opened successfully. Pages: {len(doc)}")
    
    # Extract text from all pages
    all_text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        all_text += text
        print(f"Page {page_num + 1}: {len(text)} characters")
        
        # Show first page content
        if page_num == 0:
            print("First page content:")
            print(text[:500])
    
    # Check metadata
    metadata = doc.metadata
    print(f"\nMetadata: {metadata}")
    
    doc.close()
    
    # Expected result
    expected_path = r"d:\VS_CODE\Adobe\PS\Adobe-India-Hackathon25\Challenge_1a\sample_dataset\outputs\file01.json"
    with open(expected_path, 'r') as f:
        expected = json.load(f)
    
    print(f"\nExpected result:")
    print(json.dumps(expected, indent=2))
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
