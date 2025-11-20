import os
import pickle
from verl import DataProto

# 指定你要检查的 rollout 文件路径
rollout_path = "/home/ubuntu/RAGEN/result/eval/r1w1/LichengLiu03-Qwen2.5-3B-UFO/rollouts.pkl"

print(f"Loading rollout from: {rollout_path}")

try:
    data = DataProto.load_from_disk(rollout_path)
    print(f"Loaded {len(data)} trajectories.")

    # 打印前 3 个样本的详细信息
    for i in range(min(3, len(data))):
        item = data[i]
        print(f"\n{'='*20} Sample {i} {'='*20}")
        
        # 尝试提取 observation / history
        # 数据结构通常存储在 non_tensor_batch 中
        if hasattr(item, 'non_tensor_batch'):
            batch = item.non_tensor_batch
            # 常见的键可能包括 'history', 'messages', 'prompt', 'response' 等
            # 这里我们打印所有键以便确认
            # print(f"Keys: {batch.keys()}")
            
            # 如果有 history 或 messages，打印出来
            if 'history' in batch:
                print(f"[History]:\n{batch['history']}")
            elif 'messages_list' in batch:
                 # messages_list 通常是 list of dict
                msgs = batch['messages_list']
                print("[Messages List]:")
                for m in msgs:
                    role = m.get('role', 'unknown')
                    content = m.get('content', '')
                    print(f"  [{role}]: {content[:500]}..." if len(content) > 500 else f"  [{role}]: {content}")
            else:
                print("Could not find 'history' or 'messages_list'. content of batch:")
                for k, v in batch.items():
                     print(f"  {k}: {str(v)[:200]}...")
        else:
            print("No non_tensor_batch found.")

except Exception as e:
    print(f"Error loading or parsing rollout: {e}")

