from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel

import torch
dev = ('cuda' if torch.cuda.is_available() else 'cpu')

base_model = 'Kurosawama/Llama-3.1-8B-Full-align'
adapter_model_name = 'media/discoexterno/francisco/modelos/checkpoint-1128'

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(mode, adapter_model_name)