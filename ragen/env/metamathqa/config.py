from typing import Optional, List, Dict
from dataclasses import dataclass, field

@dataclass
class MetaMathQAEnvConfig:
    """Configuration for MetaMathQA environment"""
    dataset_path: str = field(default="meta-math/MetaMathQA")
    cache_dir: str = field(default="./data")
    split: str = field(default="train")
    gamma: float = field(default=0.5)
    lambda_rep: float = field(default=1.0)
    # Episode question count (effective when multi_question_mode is True)
    n_questions_per_episode: int = field(default=1)
    # Toggle multi-question episodes; default single-question
    multi_question_mode: bool = field(default=False)

    def __post_init__(self):
        # Guard invalid counts
        if not isinstance(self.n_questions_per_episode, int) or self.n_questions_per_episode < 1:
            self.n_questions_per_episode = max(1, int(self.n_questions_per_episode or 1))
        # Auto-enable multi-question mode when users request >1 question
        if not self.multi_question_mode and self.n_questions_per_episode > 1:
            self.multi_question_mode = True
