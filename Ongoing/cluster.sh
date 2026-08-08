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
    datasets==4.8.0 \
    huggingface_hub==1.12.0 

# Python Script
cat > "$SCRIPT_PATH" << 'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig, pipeline
from datasets import Dataset
import pandas as pd
import numpy as np
import torch

trans_prompt = """
    Given a set of premises, the task is to parse the problem and the question into first-order logic formulars. Answer only with the translated premises.
    The grammar of the first-order logic formular is defined as follows:
    1) logical conjunction of expr1 and expr2: expr1 ∧ expr2
    2) logical disjunction of expr1 and expr2: expr1 ∨ expr2
    3) logical exclusive disjunction of expr1 and expr2: expr1 ⊕ expr2
    4) logical negation of expr1: ¬expr1
    5) expr1 implies expr2: expr1 → expr2
    6) expr1 if and only if expr2: expr1 ↔ expr2
    7) logical universal quantification: ∀x
    8) logical existential quantification: ∃x
    --------------
    Natural Language Premises:
    All people who regularly drink coffee are dependent on caffeine. People either regularly drink coffee or joke about being addicted to caffeine. No one who jokes about being addicted to caffeine is unaware that caffeine is a drug. Rina is either a student and unaware that caffeine is a drug, or neither a student nor unaware that caffeine is a drug. If Rina is not a person dependent on caffeine and a student, then Rina is either a person dependent on caffeine and a student, or neither a person dependent on caffeine nor a student.
    Predicates:
    Dependent(x) ::: x is a person dependent on caffeine.
    Drinks(x) ::: x regularly drinks coffee.
    Jokes(x) ::: x jokes about being addicted to caffeine.
    Unaware(x) ::: x is unaware that caffeine is a drug.
    Student(x) ::: x is a student.
    FOL Premises:
    ∀x (Drinks(x) → Dependent(x)) ::: All people who regularly drink coffee are dependent on caffeine.
    ∀x (Drinks(x) ⊕ Jokes(x)) ::: People either regularly drink coffee or joke about being addicted to caffeine.
    ∀x (Jokes(x) → ¬Unaware(x)) ::: No one who jokes about being addicted to caffeine is unaware that caffeine is a drug. 
    (Student(rina) ∧ Unaware(rina)) ⊕ ¬(Student(rina) ∨ Unaware(rina)) ::: Rina is either a student and unaware that caffeine is a drug, or neither a student nor unaware that caffeine is a drug. 
    ¬(Dependent(rina) ∧ Student(rina)) → (Dependent(rina) ∧ Student(rina)) ⊕ ¬(Dependent(rina) ∨ Student(rina)) ::: If Rina is not a person dependent on caffeine and a student, then Rina is either a person dependent on caffeine and a student, or neither a person dependent on caffeine nor a student.
    --------------
    
    Natural Language Premises:
    {}
    FOL Premises:
"""

infer_prompt = """
    Given a set of premises and conclusion in first order logic, your task is to determine the logical validity of the conclusion: True, False, or Uncertain. Answer only with the logical value.
    A True conclusion is one that can be obtained via a valid inference procedure from the given premises.
    A False conclusion is one that contradicts one or more premises during the inference procedure. 
    An Uncertain conclusion is neither True nor False. Meaning that there is insufficient information in the premises to infer it, but the conclusion it self doesn't contradict any premise.
    --------------
    The following example shows a set of premises and conclusions where each conclusion represents a different logical validity. You should answer similarly.
    FOL-PREMISES:
    ∀x (WorkAt(x, meta) → HighIncome(x))
    ∀x (HighIncome(x) → ¬MeansToDestination(x, bus))
    ∀x (MeansToDestination(x, bus) ⊕ MeansToDestination(x, drive))
    ∀x (HaveCar(x) → MeansToDestination(x, drive))
    ∀x (Student(x) → ¬ MeansToDestination(x, drive))
    HaveCar(james) ∨ WorkAt(james, meta)
    --------------
    FOL-CONCLUSION:
    MeansToDestination(x, drive) ∨ Student(james)
    Student(james)
    ¬HighIncome(james)

    Analysis:
    The first conclusion is True. Premise 6 states that either James has a car (in which case premise 4 gives us the conclusion) or James works at Meta (in which case premise 4 implies premise 2, which combined with premise 3 gives us the conclusion)
    The second conclusion is False. Premise 5 states that students can't have a Car as a MeansToDestination, however the first condition tells us James has such means.
    The third conclusion is Uncertain. Premise 1 is the only guarantee to have a High Income, however we can't determine that James works at Meta (Premise 6).
    ----------------------------
    FOL-PREMISES:
    {}
    --------------
    FOL-CONCLUSION:
    {}
    --------------
    ANSWER:
"""

retrans_prompt = """
    Given a single premise in first order logic, your task is to translate the premise into natural language. Answer only with the translated premise. It should be a single sentence.
    The grammar of the first-order logic formular is defined as follows:
    1) logical conjunction of expr1 and expr2: expr1 ∧ expr2
    2) logical disjunction of expr1 and expr2: expr1 ∨ expr2
    3) logical exclusive disjunction of expr1 and expr2: expr1 ⊕ expr2
    4) logical negation of expr1: ¬expr1
    5) expr1 implies expr2: expr1 → expr2
    6) expr1 if and only if expr2: expr1 ↔ expr2
    7) logical universal quantification: ∀x
    8) logical existential quantification: ∃x
    --------------
    Below are examples of the translation:
    PREMISES:
    ¬(PartTime(jackie) ⊕ ForbesList(jackie)) → ∃y (LessThan(y, num2) ∧ TakesCourses(x,y)) ∧ ForbesList(jackie)
    ¬In(borjMasouda, tunisia)

    NATURAL LANGUAGE:
    If Jackie either enrolls as part-time in the current semester and is listed in the Forbes 30 Under 30, or neither enrolls as part-time in the current semester nor is listed in the Forbes 30 Under 30, then Jackie takes less than two courses in the current semester and listed in the Forbes 30 Under 30.
    Borj Masouda is not in Tunisia.
    --------------    
    PREMISE:
    {}

    NATURAL LANGUAGE:
"""

def main():
    login('')
    KTO_checkpoint_list = [
    'Kurosawama/KTO_DeepSeek-R1-0528-Qwen3-8B',
    'Kurosawama/KTO_Qwen3-14B',
    'Kurosawama/KTO_gemma-3-12b-it'
    ]

    def get_logits(message, modelo, tokenizador):
    text = tokenizador.apply_chat_template(
        message,
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = True
    )
    inputs = tokenizador([text], return_tensors = 'pt').to(modelo.device)
    outputs = modelo.generate(**inputs)
    return outputs

    def ppl_per_output(logits):
        current_ppl = 0 
        for elem in logits:
            raw = torch.max(elem[0])
            log = torch.log(raw)
            current_ppl += log
        avg = (-1/len(logits)) * current_ppl
        perplexity = torch.exp(avg)
        return round(perplexity.item(), 4)

    def get_ppl(message, modelo, tokenizador):
        logits = get_logits(message, modelo, tokenizador).logits
        ppl = ppl_per_output(logits)
        #print("Perplexity: ", ppl)
        del logits
        torch.cuda.empty_cache()
        return ppl

    def get_KTO_perp(model_name, validation):
        print('Empezando...')
        if validation:
            dataset = pd.read_json('/home/flopezp/Kurosagol/FOLIO/FOLIO/folio_validation.jsonl', lines=True)
            split = 'validation'
        else:
            dataset = pd.read_json('/home/flopezp/Kurosagol/FOLIO/FOLIO/folio_test.jsonl', lines = True)
            split = 'test'
        
        trans_prompts = [trans_prompt.format(dataset['premises'][i]) for i in range(len(dataset['premises']))]
        infer_prompts = [infer_prompt.format(dataset['premises-FOL'][i], dataset['conclusion-FOL'][i]) for i in range(len(dataset['premises-FOL']))]
        retrans_prompts = [retrans_prompt.format(dataset['conclusion-FOL'][i]) for i in range(len(dataset['conclusion-FOL']))]
        prompts = trans_prompts + infer_prompts + retrans_prompts

        checkpoint = model_name
        gen_config = GenerationConfig(
            return_dict_in_generate = True,
            do_sample = True,
            temperature = 0.2,
            max_new_tokens = 1000,
            output_logits = True,
            output_scores = True,
            return_tensors = 'pt'
        )

        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype= torch.bfloat16)
        tokenizador = AutoTokenizer.from_pretrained(checkpoint)
        modelo = AutoModelForCausalLM.from_pretrained(checkpoint, generation_config = gen_config, quantization_config = quant_config, device_map = 'cuda')

        perplexity_list = []
        counter = 0
        for elem in prompts:
            counter +=1
            if counter % 20 == 0:
                print('-' * 15, f'Iteración: {counter}', '-' *15)
            perplexity_list.append(get_ppl(elem, modelo, tokenizador))
        
        del quant_config
        del tokenizador
        del modelo
        torch.cuda.empty_cache()

        print('-' * 80)
        print('Modelo: {}'.format(model_name))
        print('Split: {}'.format(split))
        print('Total generations: {}'.format(len(prompts)))    
        print('Average Perplexity: {}'.format(sum(perplexity_list)/len(prompts)))
        print('-' * 80)

    for kto in KTO_checkpoint_list:
        get_KTO_perp(kto, True)
        get_KTO_perp(kto, False)
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