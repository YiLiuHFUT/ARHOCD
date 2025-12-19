# Adversarially Robust Harmful Online Content Detector (ARHOCD) 

Main Code for our paper (Under second round review at ISR): Adversarially Robust Detection of Harmful Online Content: A Computational Design Science Approach

## Dataset

* `dataset/ethos/`: This directory contains the files of the ETHOS dataset, which is designed for hate speech detection.
* `dataset/PHEME/`: This directory contains the files of the PHEME dataset, which is designed for rumor detection.
* `dataset/white/`: This directory contains the files of the White Supremacist dataset, which focuses is designed for extremist content detection.


## OverView
* `data`:
This directory contains the files for dataset preprocessing.

* `model_and_dataset`:
This directory contains a code example for adversarial sample generation.
Adversarial samples are generated based on the TextAttack framework: https://github.com/QData/TextAttack.

* `Method/standard_training.py`: Contains the standard training procedure without adversarial samples.
* `Method/llama_sample_generation.py`: Contains the proposed LLM-based (LLaMA) adversarial sample generation method.
* `Method/ours_bayesian_ensemble.py`: Contains the proposed Bayesian two-dimensional weight assignment method.
* `Method/iterative_adversarial_training.py`: Contains the proposed iterative adversarial training strategy.

## Dependencies
* Python 3.11.14.
* Versions of all depending libraries are specified in `requirements.txt`.

## Usage

1. Train base detectors: 
To train multiple initial base detectors, run the script Method/standard_training.py.
This step produces several independently trained base detectors.

2. Generate adversarial samples: 
To generate adversarial samples for each base detectors, use the TextAttack framework: https://github.com/QData/TextAttack.
This step produces adversarial examples.

3. Generate samples with the same meaning by LLaMA: 
To generate samples with the same semantic meaning, use Method/llama_sample_generation.py.
This step produces multiple samples with the same meaning for each original input.

4. Train the proposed weight assignment method: 
To perform the proposed weight assignment method for the base detectors, use the Method/ours_bayesian_ensemble.py.
This step produces the trained weight assignor.

5. Iterative adversarial training: 
To perform the proposed iterative adversarial training strategy for both the base detectors and the weight assignor, alternately run
Method/iterative_adversarial_training.py and Method/ours_bayesian_ensemble.py.
This step produces the trained ARHOCD model.

  
