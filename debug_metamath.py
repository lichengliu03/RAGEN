
from datasets import load_dataset
import re

def extract_answer(response):
    match = re.search(r"The answer is: (.*?)$", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def normalize_text(text):
    if text is None:
        return ""
    text = text.strip().lower()
    text = re.sub(r'\s+', '', text)
    return text

try:
    ds = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
    print("Dataset loaded (streaming).")
    
    count = 0
    for item in ds:
        if not item['type'].startswith('MATH_'):
            continue
            
        response = item['response']
        query = item['query']
        extracted = extract_answer(response)
        
        print(f"--- Item {count} ---")
        print(f"Response end: ...{response[-50:]}")
        print(f"Extracted: {extracted}")
        
        if extracted is None:
            print("WARNING: Failed to extract answer from ground truth!")
        
        count += 1
        if count >= 5:
            break

except Exception as e:
    print(f"Error: {e}")

