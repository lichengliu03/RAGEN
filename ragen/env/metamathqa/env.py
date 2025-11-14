import gym
from gym import spaces
import numpy as np
from datasets import load_dataset
import re
import random
from ragen.env.base import BaseLanguageBasedEnv
from ragen.utils import all_seed
from .config import MetaMathQAEnvConfig
class MetaMathQAEnv(BaseLanguageBasedEnv):
    def __init__(self, config: MetaMathQAEnvConfig):
        super(MetaMathQAEnv, self).__init__()
        
        self.config = config
        self.dataset = load_dataset(path=self.config.dataset_path, cache_dir=self.config.cache_dir)
        self.current_question_idx = None
        self.current_question = None
        self.correct_answer = None
        self.step_num = None
        self.render_cache = None
        self._attempt_history = []
        self._attempt_distinct = set()
        # Multi-question episode tracking (only used when config.multi_question_mode is True)
        self._episode_indices = []
        self._episode_ptr = 0
        self._episode_num_correct = 0
        self._episode_num_attempted = 0
        
        
    def _extract_answer(self, response):
        match = re.search(r"The answer is: (.*?)$", response, re.DOTALL)
        print(response)
        if match:
            return match.group(1).strip()
        return None
    
    def _load_question_by_index(self, idx: int):
        dataset = self.dataset[self.config.split]
        self.current_question_idx = idx
        question_data = dataset[self.current_question_idx]
        self.current_question = question_data['query']
        self.correct_answer = self._extract_answer(question_data['response'])
    
    def _normalize_text(self, text: str) -> str:
        if text is None:
            return ""
        text = text.strip().lower()
        # remove all whitespaces
        text = re.sub(r'\s+', '', text)
        return text
        
    def reset(self,seed=None, mode=None):
        if not self.config.multi_question_mode:
            dataset = self.dataset[self.config.split]
            with all_seed(seed):
                self.current_question_idx = random.randint(0, len(dataset) - 1)
            question_data = dataset[self.current_question_idx]
            self.current_question = question_data['query']
            self.correct_answer = self._extract_answer(question_data['response'])
            self.step_num = 0
            self.render_cache = self.current_question
            self._attempt_history = []
            self._attempt_distinct = set()
            return self.render_cache
        else:
            dataset = self.dataset[self.config.split]
            n_total = len(dataset)
            nq = max(1, int(self.config.n_questions_per_episode))
            with all_seed(seed):
                if nq <= n_total:
                    self._episode_indices = random.sample(range(n_total), nq)
                else:
                    self._episode_indices = [random.randint(0, n_total - 1) for _ in range(nq)]
            self._episode_ptr = 0
            self._episode_num_correct = 0
            self._episode_num_attempted = 0
            self.step_num = 0
            self._attempt_history = []
            self._attempt_distinct = set()
            self._load_question_by_index(self._episode_indices[self._episode_ptr])
            self.render_cache = self.current_question
            return self.render_cache
        
    def step(self, action):
        if not self.config.multi_question_mode:
            is_correct, is_valid = self._check_answer(action)
            # record attempt for repetition penalty
            normalized_attempt = self._normalize_text(action)
            if normalized_attempt:
                self._attempt_history.append(normalized_attempt)
                self._attempt_distinct.add(normalized_attempt)
            T = max(1, len(self._attempt_history))
            distinct_fraction = len(self._attempt_distinct) / T
            rep_penalty = float(self.config.lambda_rep) * (1.0 - distinct_fraction)

            base_reward = (self.config.gamma ** self.step_num) if is_correct else 0.0
            reward = base_reward - rep_penalty
            if is_correct:
                observation = "Correct!"
                done = True
            else:
                observation = "Incorrect. Please think again."
                done = False
            self.step_num += 1
            info = {
                "action_is_valid": is_valid,
                "success": is_correct,
                "rep_distinct_fraction": distinct_fraction,
                "rep_penalty": rep_penalty,
            }
            self.render_cache = observation
            return self.render_cache, reward, done, info
        else:
            is_correct, is_valid = self._check_answer(action)
            normalized_attempt = self._normalize_text(action)
            if normalized_attempt:
                self._attempt_history.append(normalized_attempt)
                self._attempt_distinct.add(normalized_attempt)
            T = max(1, len(self._attempt_history))
            distinct_fraction = len(self._attempt_distinct) / T
            rep_penalty = float(self.config.lambda_rep) * (1.0 - distinct_fraction)

            base_reward = (self.config.gamma ** self.step_num) if is_correct else 0.0
            reward = base_reward - rep_penalty
            # one attempt per question, then move on
            self._episode_num_attempted += 1
            if is_correct:
                self._episode_num_correct += 1
            self._episode_ptr += 1
            done = (self._episode_ptr >= len(self._episode_indices))
            acc = (self._episode_num_correct / max(1, self._episode_num_attempted))
            acc_pct = int(round(acc * 100))
            if self._episode_num_correct == self._episode_num_attempted:
                phrase = "All correct!"
            else:
                phrase = f"{acc_pct}% correct"
            if done:
                observation = phrase
            else:
                # load next question
                self._load_question_by_index(self._episode_indices[self._episode_ptr])
                observation = f"{self.current_question}\n\n{phrase}"
            self.step_num += 1
            info = {
                "action_is_valid": is_valid,
                "success": is_correct,
                "rep_distinct_fraction": distinct_fraction,
                "rep_penalty": rep_penalty,
                "episode_total_questions": len(self._episode_indices),
                "episode_num_correct": self._episode_num_correct,
                "episode_num_attempted": self._episode_num_attempted,
                "episode_accuracy": acc,
            }
            self.render_cache = observation
            return self.render_cache, reward, done, info
    
    def _check_answer(self, user_answer):
        """Check if the user's answer matches the correct answer."""
        user_answer = user_answer.strip()
        normalized_answer = self._normalize_text(user_answer)
        if self.correct_answer:
            normalized_label = self._normalize_text(self.correct_answer)
            is_correct = normalized_answer == normalized_label
        else:
            is_correct = False
        is_valid = normalized_answer != ""
        return is_correct, is_valid

    def render(self):
        return self.render_cache


if __name__ == "__main__":
    # Create the environment configuration
    config = MetaMathQAEnvConfig(
        dataset_path="meta-math/MetaMathQA",
        cache_dir="./data",
        split="train"
    )
    
    # Initialize the environment
    env = MetaMathQAEnv(config)
    
    # Reset the environment to get the first question
    print("Question:")
    question = env.reset(seed=42)
    print(question)
    print("\nCorrect answer (for testing purposes):")
    print(env.correct_answer)
    
    # Interactive loop for testing
    while True:
        user_answer = input("\nEnter your answer (or 'q' to quit): ")
        if user_answer.lower() == 'q':
            break
        
        # Take a step in the environment with the user's answer
        #breakpoint()
        obs, reward, done, info = env.step(user_answer)
        
        
        # Print the results
        print("\nFeedback:", obs)
        print("Reward:", reward)
        print("Done:", done)
        print("Info:", info)
        
        # If the episode is done, reset the environment for a new question
        if done:
            print("\n--- New Question ---")
            question = env.reset()
            print(question)
            print("\nCorrect answer (for testing purposes):")
            print(env.correct_answer)