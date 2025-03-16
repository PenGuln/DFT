# Discriminative Finetuning (DFT)

This repository contains the implementation for the paper [Discriminative Finetuning of Generative Large Language Models without Reward Models and Preference Data](https://arxiv.org/abs/2502.18679).


## Model Checkpoints

- DFT-Mistral-7B - [siqi00/Mistral-7B-DFT](https://huggingface.co/siqi00/Mistral-7B-DFT)
- DFT2-Mistral-7B - [siqi00/Mistral-7B-DFT2](https://huggingface.co/siqi00/Mistral-7B-DFT2)
- DFT-MetaMath-Mistral-7B - [ilgee/metamath-dft1](https://huggingface.co/ilgee/MetaMath-Mistral-7B-DFT)
- DFT2-MetaMath-Mistral-7B - [ilgee/metamath-dft2](https://huggingface.co/ilgee/MetaMath-Mistral-7B-DFT2)

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

All models are trained on the SFT split of [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized), i.e., regarding the winning responses $\mathbf{y}_w$ as the ground-truth and discard all losing responses $\mathbf{y}_l$. All models use [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) as the base model. The generated negative samples $\mathbf{y}'$ can be found at [siqi00/mistral_ultrafeedback_unhelpful_chatprompt_0.7_1.0_50_320](https://huggingface.co/datasets/siqi00/mistral_ultrafeedback_unhelpful_chatprompt_0.7_1.0_50_320).

| Method | MMLU | TruthfulQA | HellaSwag | Winogrande | GSM8k | ARC | IFEval | Avg. |
|--------|-------|------------|-----------|------------|--------|-----|---------|-------|
| SFT | 62.18 | 50.04 | 83.59 | 78.06 | 45.26 | 63.65 | 49.72 | 61.79 |
| SPIN | 61.99 | 49.91 | 83.75 | 77.90 | 46.02 | 61.95 | 23.11 | 57.80 |
| SimPO | 62.39 | 52.08 | 83.89 | 78.14 | 2.58 | 61.86 | 18.85 | 51.40 |
| SimPO-SFT | 62.28 | 49.59 | 83.46 | 77.90 | 42.53 | 61.52 | 43.62 | 60.13 |
| KTO | 61.59 | 49.32 | 82.88 | 79.24 | 43.97 | 61.60 | 38.08 | 59.53 |
| Mistral-7B-DFT | 61.69 | 52.23 | 83.95 | 78.37 | 48.22 | 64.25 | 51.20 | 62.84 |
| Mistral-7B-DFT2 | 61.66 | 54.14 | 83.20 | 77.82 | 45.49 | 64.42 | 51.20 | 62.56 |


### Detailed IFEval Performance

| Method              | prompt level strict  | prompt level loose | inst level strict | inst level loose | 
|---------------------|-------|------------|-----------|------------|
| SFT                 | 49.72 | 53.23      | 60.07     | 63.55      |  
| Mistral-7B-DFT      | 51.20 | 55.27      | 61.51     | 65.47      |  
| Mistral-7B-DFT2     | 51.20 | 55.08      | 61.39     | 65.59      |

## Installation

```bash
# Clone the repository
git clone https://github.com/PenGuln/DFT.git
cd DFT
conda env create -f dft.yml
conda activate dft
```

Next, install alignment-handbook:
```bash
git clone https://github.com/huggingface/alignment-handbook.git
cd alignment-handbook
git checkout ae3f44fc7d8003d706752ca06f689574dffa3b76
python -m pip install .
cd ..
rm -r alignment-handbook
```

Finally, log into your Hugging Face and Weights and Biases accounts as follows:
```bash
huggingface-cli login
wandb login
```

## Training

DFT:

```bash
accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dft.py recipes/dft/mistral_base_dft.yaml
```

DFT2:

```bash
accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dft.py recipes/dft/mistral_base_dft2.yaml
```

For scenarios with limited GPU memory, we provide an option to precompute log probabilities for the reference model. This allows training without keeping the reference model in memory.

To use this feature, run the training script with `--precompute_offline_ref_log_probs` enabled:
```bash
accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dft.py recipes/dft/mistral_base_dft.yaml \
    --output_dir=./ckpts/dft \
    --precompute_offline_ref_log_probs=true \
    --save_strategy=no
```

This will create a `logps.pt` file in your output directory. Next, load the `logps.pt` file for reference using `--probs_dir` argument:

```bash
accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dft.py recipes/dft/mistral_base_dft.yaml \
    --probs_dir=./ckpts/dft/logps.pt
```

## Generating
We use [gen.py](https://github.com/PenGuln/DFT/blob/main/generator/gen.py) to generate negative samples. To create a 1-to-m dataset (m negative samples per prompt), run the generator with different seeds, then merge the outputs with `generator/merge_and_upload.py`. The following example shows how to create an 1-to-8 dataset:

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
    --push_to_hub  # Optional: push to Hugging Face
```

To replicate our UF self-play datasets:

```bash
bash generator/gen_uf.sh
```

To replicate our MetaMath self-play datasets:

```bash
bash generator/gen_mmqa.sh
```

## Evaluation
We use lmeval-harness and alpacaEval to complete the evaluation tasks. Make sure you are using the same version to reproduce our results:

```
pip install lm_eval==0.4.5
pip install alpaca_eval==0.6.2
```

## Citation

```bibtex
@misc{guo2025discriminativefinetuninggenerativelarge,
      title={Discriminative Finetuning of Generative Large Language Models without Reward Models and Preference Data}, 
      author={Siqi Guo and Ilgee Hong and Vicente Balmaseda and Tuo Zhao and Tianbao Yang},
      year={2025},
      eprint={2502.18679},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.18679}, 
}
```
