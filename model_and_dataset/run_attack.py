from textattack.commands.attack_command import AttackCommand
from textattack.commands.augment_command import AugmentCommand
import argparse

args_dict = {
    "model_from_file": "./model_and_dataset/my_roberta_model.py",
    "dataset_from_file": "./model_and_dataset/white_train.py",
    "attack_recipe": "tfadjusted",
    "log_to_csv": "./attack_results/white/STmodel/tfadjusted/attack_roberta_train.csv",
    "num_examples": 2752,
    "num_examples_offset": 2753,
    "csv_coloring_style": "plain",
}

args = argparse.Namespace(**args_dict)
AttackCommand().run(args)
