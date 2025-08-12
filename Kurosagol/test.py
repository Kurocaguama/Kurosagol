"""
Ejemplo de uso de la paquetería para generar los datasets de DPO.

1.- El prompt se debe de declarar antes de pasar la instancia del DatasetManager. Aquí es cuando se determina si es una estrategia con zero-shot, one-shot, o few-shot.
"""

import Kurosagol as k
import torch
from datasets import load_dataset

# Ahorita la paquetería está enfocada en su totalidad a este dataset. A largo plazo se busca generalizar para poder trabajar con múltiples datasets.
folio = load_dataset('yale-nlp/FOLIO', split = 'train')

# Prompt consistente con Logic-LLM.
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

# ¿Se tendría que explicar el procedimiento de inferencia o con las reglas estamos bien?
prompt_fol_inference = """
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


prompt_fol_to_nl = """
    Given a conclusion in first order logic, the task is to retranslate the conclusion to natural language.
    -------------------
    Conclusion:
    -------------------
    Translation:
    -------------------
    Conclusion:
    {}
    Translation:
"""


# Declaramos la instancia del DatasetManager con los parámetros necesarios (dataset['columna'], model_id, max_new_tokens)
#test = k.DatasetManager(folio, 'meta-llama/Llama-3.1-8B', 85, prompt_nl_to_fol)

# Usando .clean_dataset() obtenemos la versión limpia y sin repeticiones de FOLIO
#test.clean_dataset()

# La línea de abajo solo se ejecuta en caso de que se quiera hacer más corta la implementación. La segunda línea de este bloque es necesaria.
#test.clean_list = test.clean_list[:2]
#test.gen_strats_list()

# Se define la lista a partir de la cual se va a generar el dataset final. La función .good_dataset() regresa el dataset en formato de preferencias.
#test_dataset = test.good_dataset(test.greedy)

# Solo en caso de que se vaya a hacer un commit a HuggingFace
#test_dataset.push_to_hub('Kurosawama/greedy_DPO')


# ------------------------------------------------------------------------------------------------
# --------------------------------------ACTUAL USAGE----------------------------------------------
# ------------------------------------------------------------------------------------------------
# Puta madre se tiene que limpiar un poco.

torch.cuda.empty_cache()

translation = k.DatasetManager(folio, 'meta-llama/Llama-3.1-8B', 85, prompt_nl_to_fol)
print('Device: {}'.format(translation.dev))
#inference = k.DatasetManager(folio, 'meta-llama/Llama-3.1-8B', 85, prompt_fol_inference)
#retranslation = k.DatasetManager(folio, 'meta-llama/Llama-3.1-8B', 85, prompt_fol_to_nl)

translation.clean_dataset()
#inference.clean_dataset()
#retranslation.clean_dataset()

translation.gen_strats_list()
#inference.gen_strats_list()
#retranslation.gen_strats_list()

translation_ds = translation.good_dataset(translation.greedy)
#inference_ds = inference.good_dataset(translation.greedy)
#retranslation_ds = retranslation.good_dataset(retranslation.greedy) # NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO

translation_ds.push_to_hub('Kurosawama/Translation_DPO_greedy')

# Pruebas para ver los elementos del dataset
#print(test_dataset)
#print(test_dataset['chosen'])
#for _ in test_dataset['chosen']:
#    print(_)