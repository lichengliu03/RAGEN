from typing import Optional, List, Dict
from dataclasses import dataclass, field

@dataclass
class MetaMathQAEnvConfig:
    """Configuration for FrozenLake environment"""
    # Map config
    dataset_path: str = field(default="meta-math/MetaMathQA")
    cache_dir:str = field(default="./data")
    split: str = field(default="train")
    gamma: float = field(default=0.5)
    lambda_rep: float = field(default=1.0)
    # Number of questions per episode
    n_questions_per_episode: int = field(default=5)
    # Toggle: default single-question episode; when True, one episode contains multiple questions
    multi_question_mode: bool = field(default=False)
