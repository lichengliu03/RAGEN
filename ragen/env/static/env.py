import numpy as np
from datasets import load_dataset
import re
import random
from typing import Dict, Any, Optional, List, Tuple, Callable
from ragen.env.base import BaseLanguageBasedEnv
from ragen.utils import all_seed
from .config import StaticEnvConfig
from .utils import REGISTERD_STATIC_ENV
from huggingface_hub import HfApi
import huggingface_hub
import os
import threading

_DATASET_CACHE = {}
_DATASET_LOCK = threading.Lock()

def _freeze(x):
    # Make nested dict/list hashable
    if isinstance(x, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in x.items()))
    if isinstance(x, (list, tuple)):
        return tuple(_freeze(v) for v in x)
    return x

def fast_check_dataset_info(repo_id: str, token=None):
    try:
        api = HfApi()
        api.dataset_info(repo_id, token=token)
        return True
    except huggingface_hub.utils.RepositoryNotFoundError:
        return False
    except huggingface_hub.utils.HfHubHTTPError as e:
        print("[HF Permission Check Error]", e)
        return False

def get_shared_dataset(dataset_config, cache_dir, trust_remote_code=True, fast_check=True, token=None):

    repo_id = dataset_config.get("path") or dataset_config.get("repo_id") or dataset_config.get("name")
    
    if fast_check:
        if repo_id and not fast_check_dataset_info(repo_id, token=token):
            raise RuntimeError(f"No HF access to dataset: {repo_id}")

    key = (_freeze(dataset_config), cache_dir, trust_remote_code)
    with _DATASET_LOCK:
        if key not in _DATASET_CACHE:
            _DATASET_CACHE[key] = load_dataset(
                **dataset_config,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
            )
        return _DATASET_CACHE[key]

class StaticEnv(BaseLanguageBasedEnv):
    """
    A general environment for evaluating language models on Hugging Face datasets.
    Supports multiple datasets: MetaMathQA, TheoremQA, MATH, MMLU-STEM, GSM8K, etc.
    """
    def __init__(self, config: StaticEnvConfig):
        super(StaticEnv, self).__init__()
        
        self.config = config
        dataset_config=getattr(config, "dataset_config", None)
        if dataset_config is None:
            dataset_config=REGISTERD_STATIC_ENV[self.config.dataset_name]["config"]
        
        hf_token = os.environ.get("HUGGINGFACE_USER_KEY", None)
        self.dataset = get_shared_dataset(dataset_config, self.config.cache_dir, True, True, hf_token)
        
        if self.config.split is None:
            self.split = list(self.dataset.keys())[0]
        else:
            self.split = self.config.split
            
        self.current_question_idx = None
        self.current_question = None
        self.correct_answer = None
        self.step_num = None
        self.current_image = None  # Store current image if multimodal
        
        self.processor = REGISTERD_STATIC_ENV[self.config.dataset_name]["processor"]
        self.compute_score= REGISTERD_STATIC_ENV[self.config.dataset_name]["compute_score"]
        
        self.last_observation = None
        
    def reset(self, seed=None, mode=None):
        """Reset the environment and get a new question."""
        dataset_split = self.dataset[self.split]
        with all_seed(seed):
            self.current_question_idx = random.randint(0, len(dataset_split) - 1)
        question_data = dataset_split[self.current_question_idx]
        self.current_question, self.correct_answer = self.processor(question_data)
        self.step_num = 0
        
        # Store current image if it exists (for multimodal datasets)
        self.current_image = question_data.get("Picture", None)
        
        self.last_observation = self.current_question
        
        return self.current_question
        
    def step(self, action):
        """Take a step in the environment with the given action (answer)."""
        score_result = self.compute_score(action,self.correct_answer)
        is_correct = score_result["is_correct"]
        is_valid = score_result["is_valid"]
        reward = 1.0 / (2 ** self.step_num) if is_correct else 0.0
        if is_correct:
            observation = "Correct!"
            done = True
        else:
            observation = "Incorrect. Please think again."
            # If GPT feedback is enabled, override the default observation
            try:
                from ragen.utils.gpt_feedback import get_fuzzy_feedback
                import os
                if os.environ.get("ENABLE_GPT_FEEDBACK", "0") == "1":
                    # print(f"[DEBUG] GPT feedback enabled. Fetching...")
                    gpt_obs = get_fuzzy_feedback(self.current_question, action)
                    if gpt_obs:
                        observation = gpt_obs
                    else:
                        print("[DEBUG] GPT feedback returned empty.")
            except ImportError as e:
                print(f"[ERROR] Failed to import gpt_feedback: {e}")
            except Exception as e:
                print(f"[ERROR] Error getting GPT feedback: {e}")
            done = False
            
        self.step_num += 1
        info = {
            "success": is_correct,
            "is_valid": is_valid,
        }
        
        self.last_observation = observation
        
        return observation, reward, done, info

    def render(self, mode: str = 'text'):
        return self.last_observation
    
    def get_current_image(self):
        """Get the current image if it exists (for multimodal support)."""
        return self.current_image
    
    def has_image(self):
        """Check if the current question has an associated image."""
        return self.current_image is not None

if __name__ == "__main__":
    # Example usage
    
    
    
    for dataset_name in REGISTERD_STATIC_ENV.keys():
        config = StaticEnvConfig(
            dataset_name=dataset_name,
            cache_dir="./data",
        )
        
        # Initialize the environment
        env = StaticEnv(config)
        
        # Reset the environment to get the first question
        print("\n--- New Question ---")
        obs = env.reset(seed=42)
        print(obs)
        
        print("\n--- Correct Answer ---")
        print(env.correct_answer)
        
        # Interactive loop for testing
        while True:
            user_answer = input("\nEnter your answer (or 'q' to quit): ")
            if user_answer.lower() == 'q':
                break
            
            # Take a step in the environment with the user's answer
            obs, reward, done, info = env.step(user_answer)
            
            # Print the results
            print(f"\n{obs}")
            
            # If the episode is done, reset the environment for a new question
            if done:
                print(f"\ntotal step: {env.step_num}, reward: {reward}")
                print("\n--- New Question ---")
                question = env.reset()
                print(question)
                print("\n--- Correct Answer ---")
                print(env.correct_answer)