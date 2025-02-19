# Discriminative Finetuning (DFT)

This repository contains the implementation for the paper "Discriminative Finetuning of Generative Large Language Models without Reward Models and Preference Data".


## Model Checkpoints

- Mistral-7B-DFT - [siqi00/Mistral-7B-DFT](https://huggingface.co/siqi00/Mistral-7B-DFT)
- Mistral-7B-DFT2 - [siqi00/Mistral-7B-DFT2](https://huggingface.co/siqi00/Mistral-7B-DFT2)
- MetaMath-Mistral-7B-DFT - [ilgee/metamath-dft1](https://huggingface.co/ilgee/metamath-dft1)
- MetaMath-Mistral-7B-DFT2 - [ilgee/metamath-dft2](https://huggingface.co/ilgee/metamath-dft2)

## Performance


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
| MetaMath-Mistral-7B-DFT  | **79.15** | 28.34 |
| MetaMath-Mistral-7B-DFT2 | 78.77 | **28.62** |

### General Language Tasks

| Method              | MMLU  | TruthfulQA | HellaSwag | Winogrande | GSM8k | ARC   | IFEval |
|---------------------|-------|------------|-----------|------------|-------|-------|--------|
| Mistral-7B-DFT      | 61.69 | 52.23      | 83.95     | 78.37      | 48.22 | 64.25 | 51.20  |
| Mistral-7B-DFT2     | 61.66 | 54.14      | 83.20     | 77.82      | 45.49 | 64.42 | 51.20  |

## Install Dependencies

```bash
# Clone the repository
git clone https://github.com/username/dft.git
cd dft

# Install dependencies
conda env create -f dft.yml
conda activate dft
git clone https://github.com/huggingface/alignment-handbook.git
cd alignment-handbook
git checkout ae3f44fc7d8003d706752ca06f689574dffa3b76
python -m pip install .
cd ..
rm -r alignment-handbook
```

## Training

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

For scenarios with limited GPU memory, we provide an option to precompute log probabilities for the reference model. This allows training without keeping the reference model in memory.

To use this feature, run the training script with `--precompute_offline_ref_log_probs` enabled:
```bash
accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dft.py recipes/dft/mistral_base.yaml \
    --output_dir=./ckpts/dft \
    --samples_per_prompt=2 \
    --precompute_offline_ref_log_probs=true \
    --save_strategy=no
```

This will create a `logps.pt` file in your output directory. Next, load the `logps.pt` file for reference using `--probs_dir` argument:

```bash
accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dft.py recipes/dft/mistral_base.yaml \
    --learning_rate=2e-6 \
    --tau=1.0 \
    --u_init=0 \
    --output_dir=./ckpts/dft \
    --samples_per_prompt=2 \
    --gamma=0.85 \
    --ref_free=false \
    --probs_dir=./ckpts/dft/logps.pt
```

## Generating
A [gen.py](https://github.com/PenGuln/DFT/blob/main/generator/gen.py) is provided to generate samples using the base model:

```bash
python generator/gen.py \
    --model mistralai/Mistral-7B-v0.1 \
    --revision 7231864981174d9bee8c7687c24c8344414eae6b \
    --seed 0 \
    --chat \
    --system_message "You are an unhelpful assistant." \
    --temp 0.7 \
    --top_p 1.0 \
    --top_k 50 \
    --max_new_tokens 320 \
    --output_prefix "mistral_ultrafeedback"
```

To create a 1-to-m dataset (m negative samples per prompt), run the generator with different seeds, then merge the outputs with `generator/merge_and_upload.py`. The following example shows how to create an 1-to-8 dataset:

```bash
# Generate samples with 8 different seeds
for seed in {0..7}; do
    python generator/gen.py \
        --model mistralai/Mistral-7B-v0.1 \
        --revision 7231864981174d9bee8c7687c24c8344414eae6b \
        --seed $seed \
        --chat \
        --system_message "You are an unhelpful assistant." \
        --temp 0.7 \
        --top_p 1.0 \
        --top_k 50 \
        --max_new_tokens 320 \
        --output_prefix "mistral_ultrafeedback"
done
python generator/merge_and_upload.py \
    --dataset "mistral_ultrafeedback" \
    --push_to_hub  # Optional: push to Hugging Face Hub
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
We use lmeval-harness and alpacaEval to complete the evaluation tasks. Make sure you are using the same version to reproduce our results:

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
