#!/usr/bin/env python3
import json
import os

def compare_final_results():
    """Compare our final results with expected outputs"""
    
    our_dir = r"d:\VS_CODE\Adobe\test_output"
    expected_dir = r"d:\VS_CODE\Adobe\PS\Adobe-India-Hackathon25\Challenge_1a\sample_dataset\outputs"
    
    total_score = 0
    max_score = 0
    
    for filename in ['file01.json', 'file02.json', 'file03.json', 'file04.json', 'file05.json']:
        our_path = os.path.join(our_dir, filename)
        expected_path = os.path.join(expected_dir, filename)
        
        print(f"\n=== {filename} ===")
        
        with open(our_path, 'r', encoding='utf-8') as f:
            our_result = json.load(f)
        
        with open(expected_path, 'r', encoding='utf-8') as f:
            expected = json.load(f)
        
        # Title comparison
        our_title = our_result['title'].strip()
        expected_title = expected['title'].strip()
        title_match = our_title == expected_title
        
        # Outline comparison
        our_count = len(our_result['outline'])
        expected_count = len(expected['outline'])
        outline_match = our_count == expected_count
        
        print(f"Title: {'✓' if title_match else '✗'}")
        print(f"  Our: '{our_title}'")
        print(f"  Expected: '{expected_title}'")
        
        print(f"Outline: {'✓' if outline_match else '✗'}")
        print(f"  Our count: {our_count}")
        print(f"  Expected count: {expected_count}")
        
        # Calculate score
        file_score = (1 if title_match else 0) + (1 if outline_match else 0)
        total_score += file_score
        max_score += 2
        
        print(f"File score: {file_score}/2")
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total score: {total_score}/{max_score}")
    print(f"Accuracy: {total_score/max_score*100:.1f}%")

if __name__ == "__main__":
    compare_final_results()
