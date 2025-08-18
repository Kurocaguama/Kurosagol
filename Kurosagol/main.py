import Kurosagol as k
import torch, re
from datasets import load_dataset

folio = load_dataset('yale-nlp/FOLIO', split = 'train')

prompt_nl_to_fol = """
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
    A detailed example is shown next. Only answer with the premises, omit the explanation.
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


prompt_inference = """
    Given a set of premises in first order logic, the task is to perform a series of inference steps to deduct the logical solution of the problem.
    Valid inference rules are such as: 
    Modus Ponens (p, p → q, therefore q)
    Modus Tollens (p → q, ¬q, therefore ¬p)
    Hypotheticall Syllogism (p → q, q → r, therfore p → r)
    Destructive Dilemma (p → q, r → s, ¬q ∨ ¬s, therefore ¬p ∨ ¬r)
    Constructive Dilemma (p → q, r → s, p ∨ r, therefore q ∨ s)
    Bidirectional Dilemma (p → q, r → s, p ∨ ¬s, therefore q ∨ ¬r)
    Express your result in first order logic.
    A detailed example is shown next. Only answer with the conclusion.
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

prompt_retranslation = """
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
    A detailed example is shown next. Only answer with the retranslation.
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


def full_pipeline(k_instance, dataset_name):
    """
    Realiza el proceso de limpiar y subir el dataset a HF.

    k_instance = k.DatasetManager(folio, checkpoint_name, 85, prompt)
    dataset_name = str ; Tiene que ser el nombre final para subirlo a HuggingFace. OJO.
    """
    torch.cuda.empty_cache()
    k_instance.clean_dataset()
    k_instance.gen_strats_list()
    dataset = k_instance.good_dataset(k_instance.greedy)
    dataset.push_to_hub(dataset_name)
    torch.cuda.empty_cache()
    print("{} subido a HuggingFace".format(dataset_name))


def full_pipe_final(model_id):
    model_regex = re.split('\/', model_id)[1]
    translation = k.DatasetManager(folio, model_id, 150, prompt_nl_to_fol, 'trans')
    full_pipeline(translation, f"Kurosawama/Translation_DPO_{model_regex}")
    inference = k.DatasetManager(folio, model_id, 150, prompt_inference, 'infer')
    full_pipeline(inference, f"Kurosawama/Inference_DPO_{model_regex}")
    retranslation = k.DatasetManager(folio, model_id, 150, prompt_retranslation, 'retrans')
    full_pipeline(retranslation, f"Kurosawama/Retranslation_DPO_{model_regex}")
    print("Fin. Favor de revisar en HuggingFace.")
    print("Viva Messi.")

# For a further iteration
checkpoint_list = [ 
    'meta-llama/Llama-3.1-8B', #Done 85 
    'meta-llama/Llama-3.1-8B-Instruct', #Done 100
    'meta-llama/Llama-3.2-3B', #Done 100
    'meta-llama/Llama-3.2-3B-Instruct', #Done 100
    'meta-llama/Llama-3.3-70B-Instruct',
    'openai/gpt-oss-20b',
    'deepseek-ai/DeepSeek-R1',
    'google/gemma-3-270m',
    'google/gemma-3-1b-it'
]

test_checkpoint = ['google/gemma-3-270m', 'google/gemma-3-1b-it']
for _ in test_checkpoint:    
    full_pipe_final(_)  