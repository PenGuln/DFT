# Discriminative Finetuning (DFT) of Large Language Models

This repository contains the implementation for the paper "Discriminative Finetuning of Generative Large Language Models without Reward Models and Preference Data".

## Overview

DFT is a novel approach for finetuning large language models (LLMs) that eliminates the need for preference data or reward models while achieving performance competitive with SFT→PO methods. Unlike standard supervised fine-tuning (SFT) which uses a generative approach, DFT adopts a discriminative paradigm that:

- Pushes up the likelihood of positive answers
- Explicitly suppresses negative examples generated by the base model
- Uses an efficient optimization algorithm with moving average estimators

## Models

- [Mistral-7B-DFT](https://huggingface.co/siqi00/Mistral-7B-DFT)
- [Mistral-7B-DFT2](https://huggingface.co/siqi00/Mistral-7B-DFT2)
- [MetaMath-7B-DFT](https://huggingface.co/ilgee/metamath-dft1)
- [MetaMath-7B-DFT2](https://huggingface.co/ilgee/metamath-dft2)

## Results

DFT achieves strong performance across multiple benchmarks:

### Mathematical Reasoning

| Method  | GSM8K | MATH |
|---------|-------|------|
| RFT-7B | 50.3 | -- |
| Qwen-7B | 51.6 | -- |
| Mistral-7B-base | 52.2 | 13.1 |
| WizardMath-7B | 54.9 | 10.7 |
| LLaMA-2-70B | 56.8 | 13.5 |
| MetaMath-7B | 66.5 | 19.8 |
| MetaMath-Mistral-7B | 77.7 | 28.2 |
| DFT (Mistral-7B-base) | **79.15** | 28.34 |
| DFT2 (Mistral-7B-base) | 78.77 | **28.62** |

### General Language Tasks

Competitive performance with SFT→PO methods across MMLU, TruthfulQA, HellaSwag, WinoGrande, GSM8K, ARC, and IFEval benchmarks. See paper for detailed results.


## Install Dependencies

```bash
# Clone the repository
git clone https://github.com/username/dft.git
cd dft

# Install dependencies
conda env create -f dft.yml
conda activate dft
git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
git checkout ae3f44fc7d8003d706752ca06f689574dffa3b76
python -m pip install .
```

## Usage

### Training

DFT:

```bash
accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dft.py recipes/dft/mistral_base.yaml \
    --learning_rate=2e-6 \
    --tau=1.0 \
    --u_init=0 \
    --output_dir=./ckpts/dft \
    --samples_per_prompt=2 \
    --gamma=0.85 \
    --ref_free=false
```

DFT2:

```bash
accelerate launch --main_process_port 29507 --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dft.py recipes/dft/mistral_base.yaml \
    --learning_rate=2e-6 \
    --tau=0.3 \
    --u_init=0 \
    --output_dir=./ckpts/dft2 \
    --samples_per_prompt=2 \
    --gamma=0.9 \
    --ref_free=true
```

### Precompute Log Probabilities

```bash
accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dft.py recipes/dft/mistral_base.yaml \
    --output_dir=./ckpts/dft \
    --samples_per_prompt=2 \
    --precompute_offline_ref_log_probs=true \
    --save_strategy=no
```

```bash
accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dft.py recipes/dft/mistral_base.yaml \
    --learning_rate=2e-6 \
    --tau=1.0 \
    --u_init=0 \
    --output_dir=./ckpts/dft \
    --samples_per_prompt=2 \
    --gamma=0.85 \
    --ref_free=false \
    --probs_dir=./ckpts/logps.pt
```

### Generating Data
Basic usage:

```bash
python generator/gen.py \
    --model mistralai/Mistral-7B-v0.1 \
    --revision 7231864981174d9bee8c7687c24c8344414eae6b \
    --seed 0 \
    --chat \
    --system_message "As an AI assistant, it's your job to ensure that the information you provide to users is accurate, current, and relevant. Offer comprehensive details to educate and engage the user." \
    --temp 0.7 \
    --top_p 1.0 \
    --top_k 50 \
    --max_new_tokens 320 \
    --output_prefix "mistral_ultrafeedback"
```

To create a self-play dataset with multiple negative samples, run the generator with different seeds, then merge the outputs with `generator/merge_and_upload.py`. Example:
```bash
python generator/gen.py --seed 0 \  # ... other arguments remain the same
python generator/gen.py --seed 1 \  # ... other arguments remain the same
python generator/gen.py --seed 2 \  # ... other arguments remain the same
python generator/gen.py --seed 3 \  # ... other arguments remain the same
python generator/merge_and_upload.py \
    --dataset "mistral_ultrafeedback" \
    --push_to_hub                   # Add this argument to push the dataset to huggingface
```

To replicate our UF self-play datasets:

```bash
bash generator/gen_uf.sh
```

To replicate our MetaMath self-play datasets:

```bash
bash generator/gen_mm.sh
```

## Evaluation
We use lmeval-harness and alpacaEval to complete the evaluation tasks. To install the dependence:
```
pip install lm_eval==0.4.5
pip install alpaca_eval==0.6.2
```



## Citation

```bibtex
@article{guo2025discriminative,
  title={Discriminative Finetuning of Generative Large Language Models without Reward Models and Preference Data},
  author={Guo, Siqi and Hong, Ilgee and Balmaseda, Vicente and Zhao, Tuo and Yang, Tianbao},
  journal={Preprint},
  year={2025}
}
```
