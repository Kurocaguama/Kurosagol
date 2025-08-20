from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig, AutoModel
import pandas as pd
import re, torch

os_translation = """
    Given a problem description and a question, the task is to parse the problem and the question into first-order logic formulars.
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
    Problem:
    All people who regularly drink coffee are dependent on caffeine. People either regularly drink coffee or joke about being addicted to caffeine. No one who jokes about being addicted to caffeine is unaware that caffeine is a drug. Rina is either a student and unaware that caffeine is a drug, or neither a student nor unaware that caffeine is a drug. If Rina is not a person dependent on caffeine and a student, then Rina is either a person dependent on caffeine and a student, or neither a person dependent on caffeine nor a student.
    Predicates:
    Dependent(x) ::: x is a person dependent on caffeine.
    Drinks(x) ::: x regularly drinks coffee.
    Jokes(x) ::: x jokes about being addicted to caffeine.
    Unaware(x) ::: x is unaware that caffeine is a drug.
    Student(x) ::: x is a student.
    Premises:
    ∀x (Drinks(x) → Dependent(x)) ::: All people who regularly drink coffee are dependent on caffeine.
    ∀x (Drinks(x) ⊕ Jokes(x)) ::: People either regularly drink coffee or joke about being addicted to caffeine.
    ∀x (Jokes(x) → ¬Unaware(x)) ::: No one who jokes about being addicted to caffeine is unaware that caffeine is a drug. 
    (Student(rina) ∧ Unaware(rina)) ⊕ ¬(Student(rina) ∨ Unaware(rina)) ::: Rina is either a student and unaware that caffeine is a drug, or neither a student nor unaware that caffeine is a drug. 
    ¬(Dependent(rina) ∧ Student(rina)) → (Dependent(rina) ∧ Student(rina)) ⊕ ¬(Dependent(rina) ∨ Student(rina)) ::: If Rina is not a person dependent on caffeine and a student, then Rina is either a person dependent on caffeine and a student, or neither a person dependent on caffeine nor a student.
    --------------
    
    Problem:
    {}
    Predicates:
"""

os_inference = """
    Given a problem in first order logic, the task is to perform a series of inference steps to deduct the logical solution of the problem.
    Valid inference rules are such as: 
    Modus Ponens (p, p → q, therefore q)
    Modus Tollens (p → q, ¬q, therefore ¬p)
    Hypotheticall Syllogism (p → q, q → r, therfore p → r)
    Destructive Dilemma (p → q, r → s, ¬q ∨ ¬s, therefore ¬p ∨ ¬r)
    Constructive Dilemma (p → q, r → s, p ∨ r, therefore q ∨ s)
    Bidirectional Dilemma (p → q, r → s, p ∨ ¬s, therefore q ∨ ¬r)
    -------------------
    Problem:
    ∀x (DrinkRegularly(x, coffee) → IsDependentOn(x, caffeine))
    ∀x (DrinkRegularly(x, coffee) ∨ (¬WantToBeAddictedTo(x, caffeine)))
    ∀x (¬WantToBeAddictedTo(x, caffeine) → ¬AwareThatDrug(x, caffeine))
    ¬(Student(rina) ⊕ ¬AwareThatDrug(rina, caffeine))
    ¬(IsDependentOn(rina, caffeine) ⊕ Student(rina))
    Conclusion:
    ¬WantToBeAddictedTo(rina, caffeine) ∨ (¬AwareThatDrug(rina, caffeine))
    -------------------

    Problem:
    {}
    Conclusion:
"""

os_retranslation = """
    Given a single premise in first order logic, the task is to retranslate it to natural language.
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
    Premise:
    ¬WantToBeAddictedTo(rina, caffeine) ∨ (¬AwareThatDrug(rina, caffeine))

    Retranslation:
    Rina doesn't want to be addicted to caffeine or is unaware that caffeine is a drug.
    --------------

    Premise:
    {}
    Retranslation:
"""

folio = load_dataset('yale-nlp/FOLIO', split = 'validation')
trans = folio['premises']
infer = folio['premises-FOL']
retrans = folio['conclusion-FOL']

full_checkpoint = [
    'Kurosawama/Llama-3.1-8B-Full-align',
    'Kurosawama/Llama-3.1-8B-Instruct-Full-align',
    'Kurosawama/Llama-3.2-3B-Full-align',
    'Kurosawama/Llama-3.2-3B-Instruct-Full-align',
    'Kurosawama/gemma-3-1b-it-Full-align'
]

dev = ('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model_id, tokenizer_id, stage):
    """
        stage = str ; trans, infer, retrans. El valor modifica los parámetros internos del modelo
    """
    if stage == 'trans':
        dataset = trans
        prompt = os_translation
    if stage == 'infer':
        dataset = infer
        prompt = os_inference
    if stage == 'retrans':
        dataset = retrans
        prompt = os_retranslation

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.chat_template = """
			<|im_start|>system
			{SYSTEM}<|im_end|>
			<|im_start|>user
			{INPUT}<|im_ed|>
			<|im_start|>assistant
			{OUTPUT}<|im_end|>
		"""
    quant_config = BitsAndBytesConfig(load_in_4bit = True, bnb_4bit_compute_dtype = torch.bfloat16)
    gen_config = GenerationConfig.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, quantization_config = quant_config, generation_config = gen_config).to(dev)

    answer_list = []
    for _ in dataset[:5]:
        aux_prompt = prompt.format(_)
        inputs = tokenizer.encode(aux_prompt, return_tensors = 'pt').to(dev)

        outputs = model.generate(**inputs, max_new_tokens = 100)
        answer = tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]
        answer = answer[len(prompt):]
        answer_list.append(answer)

    return evaluate

print(evaluate('Kurosawama/Llama-3.1-8B-Full-align', 'meta-llama/Llama-3.1-8B', 'trans'))