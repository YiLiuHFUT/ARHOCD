import tensorflow as tf
from textattack.commands.augment_command import AugmentCommand
import argparse

# bert
# 测试集
args_dict = {
    "input_csv": "./attack_results/ethos/ST/trepat/attack_bert_test.csv",
    "output_csv": "./attack_results/ethos/ST_trans/trepat/attack_bert_test.csv",
    "input_column": "perturbed_text",
    "recipe": "back_trans",
    "pct_words_to_swap": 0.1,
    "transformations_per_example": 3,
    "exclude_original": False,
    "high_yield": False,
    "fast_augment": True,
    "enable_advanced_metrics": False,
    "interactive": False,
    "random_seed": 42,
    "overwrite": True,
}

args = argparse.Namespace(**args_dict)
AugmentCommand().run(args)

