#!/usr/bin/env python3
"""
Simple script to print test summary from results file
"""

def main():
    try:
        with open('test_results.txt', 'r') as f:
            content = f.read()
            
        # Find the summary section
        if 'Summary:' in content:
            summary_start = content.find('Summary:')
            summary_end = content.find('\n\n', summary_start)
            
            if summary_end == -1:  # If there's no double newline, go to the end
                summary_text = content[summary_start:]
            else:
                summary_text = content[summary_start:summary_end]
            
            print(summary_text)
        
        # Find and print the detailed results section
        if 'Detailed results:' in content:
            results_start = content.find('Detailed results:')
            
            print("\n" + content[results_start:])
        
    except Exception as e:
        print(f"Error reading results file: {e}")

if __name__ == "__main__":
    main() 