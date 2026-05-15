#!/bin/bash
# ============================================================
# run_job.sh — Self-contained Python job for cloud environments
# Usage: bash run_job.sh
# ============================================================

set -euo pipefail

# Config
PYTHON=${PYTHON:-python3}
VENV_DIR="/tmp/job_venv"
SCRIPT_PATH="/tmp/job_script.py"

echo "==> Setting up environment..."

# Virtual Env
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Dependencies
echo "==> Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet \
    torch \
    transformers==5.8.0 \
    trl==1.4.0 \
    datasets==4.8.0 \
    huggingface_hub==1.12.0 \
    peft==0.19.0  

# Python Script
cat > "$SCRIPT_PATH" << 'EOF'
import torch
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward
from trl.experimental.kto import KTOTrainer, KTOConfig
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
from peft import LoraConfig, get_peft_model

def main():
    # ------------------------ Set up ------------------------
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected. This script requires a CUDA-enabled GPU.")
    print(f"==> Using GPU: {torch.cuda.get_device_name(0)}")

    hf_token = ''
    login(hf_token)
    grpo_ds = load_dataset("Kurosawama/GRPO_FOL", split='train')

    checkpoints = [
        'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
        'google/gemma-3-12b-it',
        'Qwen/Qwen3-14B-FP8',
        'openai/gpt-oss-20b'
    ]

    # ------------------------ GRPO ------------------------
    def train_and_push(checkpoint):
        model_id = checkpoint.split('/')[-1]
        config = GRPOConfig(
            loss_type='drgrpo',
            use_vllm=True,
            epsilon=0.15
        )

        trainer = GRPOTrainer(
            model=checkpoint,
            args=config,               
            reward_funcs=accuracy_reward,
            train_dataset=grpo_ds
        )
        trainer.train()
        trainer.push_to_hub('Kurosawama/DrGRPO_{}'.format(model_id), private=True)
        print('Acabamos con {}'.format(model_id))

        del trainer
        del config
        torch.cuda.empty_cache()

    for checkpoint in checkpoints:
        print(f"==> Training: {checkpoint}")
        train_and_push(checkpoint)

    print('GRPO Finalizado')

    # ------------------------ KTO con GPT ------------------------
    kto_gptoss = load_dataset('Kurosawama/KTO_gpt-oss-20b', split = 'train')

    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    quantization_config = Mxfp4Config(dequantize=True)
    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        use_cache=False,
        device_map="auto",
    )

    model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", **model_kwargs)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules="all-linear"
    )

    peft_model = get_peft_model(model, peft_config)

    training_args = KTOConfig(
        output_dir = '',
        learning_rate=5e-6,
        num_train_epochs = 2,
        per_device_train_batch_size = 8, 
        remove_unused_columns = False,
        use_cache= False
    )
    trainer = KTOTrainer(model=peft_model, args=training_args, processing_class=tokenizer, train_dataset=kto_gptoss)
    trainer.train()
    trainer.push_to_hub('Kurosawama/KTO_gpt-oss-20b')
    print('Finito.')

if __name__ == "__main__":
    main()
EOF

echo "==> Running Python script..."
$PYTHON "$SCRIPT_PATH"

# Cleanup
echo "==> Done. Cleaning up..."
rm -f "$SCRIPT_PATH"
deactivate

echo "==> Job complete."