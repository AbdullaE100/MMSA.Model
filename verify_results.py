#!/usr/bin/env python3
"""
Simple script to verify video sentiment classification results
"""

print("Checking final classification results...\n")

try:
    with open('final_check_results.txt', 'r') as f:
        content = f.read()
        print(content)
except Exception as e:
    print(f"Error reading results: {e}") 