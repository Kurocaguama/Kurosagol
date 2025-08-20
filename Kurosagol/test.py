from transformers import pipeline
from datasets import load_dataset, Dataset
import pandas as pd
import re

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
final_trans = folio['conclusion']

checkpoint_list = [
    'Kurosawama/Llama-3.1-8B-Full-align',
    'Kurosawama/Llama-3.1-8B-Instruct-Full-align',
    'Kurosawama/Llama-3.2-3B-Full-align',
    'Kurosawama/Llama-3.2-3B-Instruct-Full-align',
    'Kurosawama/gemma-3-1b-it-Full-align'
]

# Las siguientes dos funciones son necesarias para evaluar los modelos ALINEADOS.
# respond() genera las respuestas de cada modelo dentro de cada etapa del pipeline.
# llm_gen_to_hf_dataset() ajusta estas respuestas y las sube al hf hub.

def respond(model_id, stage):
    pipe = pipeline("text-generation", model = model_id, device = 'cuda:0')
    if stage == 'trans':
        dataset = trans
        prompt = os_translation
    if stage == 'infer':
        dataset = infer
        prompt = os_inference
    if stage == 'retrans':
        dataset = retrans
        prompt = os_retranslation

    answer_list = []
    aux_iter = 0
    for _ in dataset:
        aux_prompt = prompt.format(_)
        #answer = pipe([{"role": "user", "content": aux_prompt}], max_new_tokens = 150)
        answer = pipe(aux_prompt, max_new_tokens = 275)
        cut_answer = answer[0]["generated_text"]
        cut_answer = cut_answer[len(aux_prompt):]
        if aux_iter % 5 == 0:
            print(cut_answer)
        answer_list.append(cut_answer)
        aux_iter += 1
    return answer_list


def llm_gen_to_hf_dataset(checkpoint, stage):
    name = re.split('\/', checkpoint)[1]
    if stage == 'trans':
        ds = infer
    if stage == 'infer':
        ds = retrans
    if stage == 'retrans':
        ds = final_trans
    eval_generation = respond(checkpoint, stage)
    dic1 = {'FOLIO': ds, f'{checkpoint}\'s Answer': eval_generation}
    eval_df = pd.DataFrame(data=dic1)
    hf_dataset = Dataset.from_pandas(eval_df)
    hf_dataset.push_to_hub(f'Kurosawama/EVALUATION_{stage}_{name}')

for check in checkpoint_list:
    llm_gen_to_hf_dataset(check, 'trans')
    llm_gen_to_hf_dataset(check, 'infer')
    llm_gen_to_hf_dataset(check, 'retrans')




#testing = respond(checkpoint_list[0], "trans")
#print(len(testing))

# Lo de arriba también venga.
# Esto de abajo funciona

#model_name_or_path = 'Kurosawama/gemma-3-1b-it-Full-align'
#pipe = pipeline("text-generation", model=model_name_or_path)
#print(pipe("Translate the following set of premises to First-Order Logic: All men are mortal. Plato is a man.")[0]["generated_text"])