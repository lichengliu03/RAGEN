
import pickle
from verl import DataProto
from transformers import AutoTokenizer
import sys
import os
import torch

# Use Qwen tokenizer as in the script
model_path = "Qwen/Qwen2.5-3B-Instruct"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dp = DataProto.load_from_disk("result/eval/r2w7/Qwen-Qwen2.5-3B-Instruct/grpo/step_10_turn_1/rollouts.pkl")
    print(f"Loaded DataProto with {len(dp)} items")
    
    # Print metrics
    print("\nMeta Info Metrics:")
    metrics = dp.meta_info.get("metrics", {})
    print(metrics)

    print("\n--- First 3 Responses ---")
    for i in range(min(3, len(dp))):
        item = dp[i]
        batch = item.batch
        
        if batch is None:
            print(f"[Item {i}] Batch is None")
            continue

        # Check responses
        if "responses" in batch.keys():
            responses = batch["responses"]
            print(f"[Item {i}] Responses type: {type(responses)}")
            if hasattr(responses, "shape"):
                print(f"[Item {i}] Responses shape: {responses.shape}")
            
            if isinstance(responses, torch.Tensor):
                # If it's 1D, it's just a sequence. If 2D, it's batch x sequence.
                # dp[i] usually slices so batch dim might be gone or 1.
                if responses.dim() == 1:
                    response_ids = responses
                elif responses.dim() == 2:
                    response_ids = responses[0]
                else:
                    print(f"Unexpected dimension: {responses.dim()}")
                    continue
                    
                response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
                env_id = item.non_tensor_batch['env_ids']
                # env_id might be array or scalar
                if hasattr(env_id, '__iter__') and len(env_id) > 0:
                   env_id = env_id[0]
                
                print(f"\n[Item {i}] EnvID: {env_id}")
                print(f"Response: {response_text[:500]}...") 
            else:
                print(f"[Item {i}] Responses is not a tensor: {responses}")
        else:
            print(f"No 'responses' key in batch for item {i}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
