"""
Ejemplo de uso de la paquetería para generar los datasets de DPO.

1.- El prompt se debe de declarar antes de pasar la instancia del DatasetManager. Aquí es cuando se determina si es una estrategia con zero-shot, one-shot, o few-shot.
"""

import Kurosagol as k
from datasets import load_dataset

# Ahorita la paquetería está enfocada en su totalidad a este dataset. A largo plazo se busca generalizar para poder trabajar con múltiples datasets.
aux = load_dataset('yale-nlp/FOLIO', split = 'train')

# Prompt consistente con LogicBenchmark.
prompt = """
    Given a problem description and a question. The task is to parse the problem and the question into first-order logic formulars.
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

# Declaramos la instancia del DatasetManager con los parámetros necesarios (dataset['columna'], model_id, max_new_tokens)
test = k.DatasetManager(aux, 'meta-llama/Llama-3.2-1B', 60, prompt)


# Usando .clean_dataset() obtenemos la versión limpia y sin repeticiones de FOLIO
test.clean_dataset()


# La línea de abajo solo se ejecuta en caso de que se quiera hacer más corta la implementación. La segunda línea de este bloque es necesaria.
test.clean_list = test.clean_list[:2]
test.gen_strats_list()


# Se define la lista a partir de la cual se va a generar el dataset final. La función .good_dataset() regresa el dataset en formato de preferencias.
test_dataset = test.good_dataset(test.greedy)

# Solo en caso de que se vaya a hacer un commit a HuggingFace
#test_dataset.push_to_hub('Kurosawama/greedy_DPO')


# Pruebas para ver los elementos del dataset
print(test_dataset)
print(test_dataset['chosen'])
for _ in test_dataset['chosen']:
    print(_)