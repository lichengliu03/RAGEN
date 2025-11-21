import sys
import json
from pathlib import Path
from verl import DataProto
import numpy as np

def check_rollout(path_str):
    path = Path(path_str)
    if not path.exists():
        print(f"File not found: {path}")
        return

    print(f"Checking {path}")
    dp = DataProto.load_from_disk(str(path))
    
    # 1. Check basic info
    print(f"Total items: {len(dp)}")
    
    # 2. Check metrics in meta_info
    if hasattr(dp, 'meta_info'):
        metrics = dp.meta_info.get('metrics', {})
        print("\nMetrics in meta_info:")
        for k, v in metrics.items():
            if 'MetamathQA' in k:
                print(f"  {k}: {v}")

    # 3. Check a few samples for truncation
    print("\nChecking sample responses for truncation...")
    tokenizer_path = "Qwen/Qwen2.5-3B-Instruct"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    count = 0
    for i in range(min(5, len(dp))):
        item = dp[i]
        # response_ids = item.batch['responses'][0] # Assuming batch size 1 slice
        # Or if batch is stripped, we might need to look at non_tensor_batch['response_texts'] if available
        # But usually 'responses' tensor is kept.
        
        if 'responses' in item.batch.keys():
             res = item.batch['responses']
             # handle dimension
             if res.dim() == 2: res = res[0]
             
             text = tokenizer.decode(res, skip_special_tokens=False)
             
             # Check for answer tag
             has_answer = "<answer>" in text and "</answer>" in text
             print(f"  Sample {i}: Length {len(res)} tokens. Has <answer>: {has_answer}")
             if not has_answer:
                 print(f"    -> Content: {text[-200:]}") # Print last 200 chars
        else:
             print(f"  Sample {i}: No responses tensor found")

path = "result/eval/r2w7/Qwen-Qwen2.5-3B-Instruct/grpo/step_10_turn_1/rollouts.pkl"
check_rollout(path)
