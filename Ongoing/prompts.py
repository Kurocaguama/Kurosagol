# =========================================================================
# ============================== Translation ==============================
# =========================================================================
translation1 = """
    Given a set of premises, the task is to parse the problem and the question into first-order logic formulas. The following is the information you need to complete the task.
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
    The following is an example of the task you need to carry out. You should follow the output format of the FOL Premises section.
    Natural Language Premises:
    All people who regularly drink coffee are dependent on caffeine. People either regularly drink coffee or joke about being addicted to caffeine. No one who jokes about being addicted to caffeine is unaware that caffeine is a drug. Rina is either a student and unaware that caffeine is a drug, or neither a student nor unaware that caffeine is a drug. If Rina is not a person dependent on caffeine and a student, then Rina is either a person dependent on caffeine and a student, or neither a person dependent on caffeine nor a student.
    Predicates:
    Dependent(x) ::: x is a person dependent on caffeine.
    Drinks(x) ::: x regularly drinks coffee.
    Jokes(x) ::: x jokes about being addicted to caffeine.
    Unaware(x) ::: x is unaware that caffeine is a drug.
    Student(x) ::: x is a student.
    <text>
    FOL Premises:
    ∀x (Drinks(x) → Dependent(x)) 
    ∀x (Drinks(x) ⊕ Jokes(x)) 
    ∀x (Jokes(x) → ¬Unaware(x))
    (Student(rina) ∧ Unaware(rina)) ⊕ ¬(Student(rina) ∨ Unaware(rina)) 
    ¬(Dependent(rina) ∧ Student(rina)) → (Dependent(rina) ∧ Student(rina)) ⊕ ¬(Dependent(rina) ∨ Student(rina))
    </text>
    --------------
    Answer only with the FOL premises with the desired output.

    Natural Language Premises:
    {}
    FOL Premises:
"""

translation2 = """
    Given a set of premises, the task is to parse the problem and the question into first-order logic formulas. The following is the information you need to complete the task.
    The grammar of the first-order logic formular is defined as follows:
    1) logical conjunction of expr1 and expr2: expr1 ∧ expr2
    2) logical disjunction of expr1 and expr2: expr1 ∨ expr2
    3) logical exclusive disjunction of expr1 and expr2: expr1 ⊕ expr2
    4) logical negation of expr1: ¬expr1
    5) expr1 implies expr2: expr1 → expr2
    6) expr1 if and only if expr2: expr1 ↔ expr2
    7) logical universal quantification: ∀x
    8) logical existential quantification: ∃x
    The following is an example of the task you need to carry out. You should follow the output format of the FOL Premises section.
    Natural Language Premises:
    All people who regularly drink coffee are dependent on caffeine. People either regularly drink coffee or joke about being addicted to caffeine. No one who jokes about being addicted to caffeine is unaware that caffeine is a drug. Rina is either a student and unaware that caffeine is a drug, or neither a student nor unaware that caffeine is a drug. If Rina is not a person dependent on caffeine and a student, then Rina is either a person dependent on caffeine and a student, or neither a person dependent on caffeine nor a student.
    Predicates:
    Dependent(x) ::: x is a person dependent on caffeine.
    Drinks(x) ::: x regularly drinks coffee.
    Jokes(x) ::: x jokes about being addicted to caffeine.
    Unaware(x) ::: x is unaware that caffeine is a drug.
    Student(x) ::: x is a student.
    FOL Premises:
    ∀x (Drinks(x) → Dependent(x)) 
    ∀x (Drinks(x) ⊕ Jokes(x)) 
    ∀x (Jokes(x) → ¬Unaware(x))
    (Student(rina) ∧ Unaware(rina)) ⊕ ¬(Student(rina) ∨ Unaware(rina)) 
    ¬(Dependent(rina) ∧ Student(rina)) → (Dependent(rina) ∧ Student(rina)) ⊕ ¬(Dependent(rina) ∨ Student(rina))
    Answer only with the FOL premises with the desired output. Start your final output with <text>. End your final output with </text>

    Natural Language Premises:
    {}
    FOL Premises:
"""


# Este es el que mejor funciona la neta
translation3 = """
    Given a set of premises, your task is to parse the problem and the question into first-order logic formulas. The following is the information you need to complete the task.
    The grammar of the first-order logic formular is defined as follows:
    1) logical conjunction of expr1 and expr2: expr1 ∧ expr2
    2) logical disjunction of expr1 and expr2: expr1 ∨ expr2
    3) logical exclusive disjunction of expr1 and expr2: expr1 ⊕ expr2
    4) logical negation of expr1: ¬expr1
    5) expr1 implies expr2: expr1 → expr2
    6) expr1 if and only if expr2: expr1 ↔ expr2
    7) logical universal quantification: ∀x
    8) logical existential quantification: ∃x
    Predicates have to be AT LEAST two characters long.
    The following is an example of the task you need to carry out. You should follow the output format of the FOL Premises section.
    Natural Language Premises:
    All people who regularly drink coffee are dependent on caffeine. People either regularly drink coffee or joke about being addicted to caffeine. No one who jokes about being addicted to caffeine is unaware that caffeine is a drug. Rina is either a student and unaware that caffeine is a drug, or neither a student nor unaware that caffeine is a drug. If Rina is not a person dependent on caffeine and a student, then Rina is either a person dependent on caffeine and a student, or neither a person dependent on caffeine nor a student.
    Predicates:
    Dependent(x)
    Drinks(x)
    Jokes(x)
    Unaware(x)
    Student(x)
    FOL Premises:
    ∀x (Drinks(x) → Dependent(x)) 
    ∀x (Drinks(x) ⊕ Jokes(x)) 
    ∀x (Jokes(x) → ¬Unaware(x))
    (Student(rina) ∧ Unaware(rina)) ⊕ ¬(Student(rina) ∨ Unaware(rina)) 
    ¬(Dependent(rina) ∧ Student(rina)) → (Dependent(rina) ∧ Student(rina)) ⊕ ¬(Dependent(rina) ∨ Student(rina))
    Answer only with the FOL premises with the desired output. Start your final output with <text>. End your final output with </text>

    Natural Language Premises:
    {}
    FOL Premises:
"""

mod_translation3 = """
    Given a set of premises, your task is to parse the problem and the question into first-order logic formulas. The following is the information you need to complete the task.
    The grammar of the first-order logic formular is defined as follows:
    1) logical conjunction of expr1 and expr2: expr1 ∧ expr2
    2) logical disjunction of expr1 and expr2: expr1 ∨ expr2
    3) logical negation of expr1: ¬expr1
    4) logical exclusive disjunction of expr1 and expr2: (expr1 ∨ expr2) ∧ ¬(expr1 ∧ expr2)
    5) expr1 implies expr2: expr1 → expr2
    6) expr1 if and only if expr2: expr1 ↔ expr2
    7) logical universal quantification: ∀x
    8) logical existential quantification: ∃x
    Predicates have to be AT LEAST two characters long.
    The following is an example of the task you need to carry out. You should follow the output format of the FOL Premises section.
    Natural Language Premises:
    All people who regularly drink coffee are dependent on caffeine. People either regularly drink coffee or joke about being addicted to caffeine. No one who jokes about being addicted to caffeine is unaware that caffeine is a drug. Rina is either a student and unaware that caffeine is a drug, or neither a student nor unaware that caffeine is a drug. If Rina is not a person dependent on caffeine and a student, then Rina is either a person dependent on caffeine and a student, or neither a person dependent on caffeine nor a student.
    Predicates:
    Dependent(x)
    Drinks(x)
    Jokes(x)
    Unaware(x)
    Student(x)
    FOL Premises:
    ∀x (Drinks(x) → Dependent(x)) 
    ∀x (Drinks(x) ∨ Jokes(x)) ∧ ¬(Drinks(x) ∧ Jokes(x))
    ∀x (Jokes(x) → ¬Unaware(x))
    ((Student(rina) ∧ Unaware(rina)) ∨ ¬(Student(rina) ∨ Unaware(rina))) ∧ ¬((Student(rina) ∧ Unaware(rina)) ∧ ¬(Student(rina) ∨ Unaware(rina)))
    ¬(Dependent(rina) ∧ Student(rina)) → (Dependent(rina) ∧ Student(rina)) ⊕ ¬(Dependent(rina) ∨ Student(rina))
    Answer only with the FOL premises with the desired output. Start your final output with <text>. End your final output with </text>

    Natural Language Premises:
    {}
    FOL Premises:
"""


translation4 = """
    Your task is to translate the given argument into first-order logic (FOL) following the exact specifications below.
    Do not repeat or reuse the example provided. Only translate the new argument.

    1. Structure of the argument
    Each argument consists of:
    - Premises: statements providing evidence or assumptions.
    - Conclusion: the main claim derived from the premises.
    If the argument contains implicit premises (unstated but necessary assumptions), make them explicit in your translation.

    2. FOL Grammar
    Use the following formal grammar exactly:
    1) Conjunction of expr1 and expr2: expr1 ∧ expr2
    2) Disjunction of expr1 and expr2: expr1 ∨ expr2
    3) Exclusive disjunction of expr1 and expr2: expr1 ⊕ expr2
    4) Negation of expr1: ¬expr1
    5) expr1 implies expr2: expr1 → expr2
    6) expr1 if and only if expr2: expr1 ↔ expr2
    7) Universal quantification: ∀x
    8) Existential quantification: ∃x

    3. Translation Steps
    a) Identify the main entities (objects), their properties, and relationships.
    b) Translate the argument into FOL using the defined grammar.
    c) Include implicit premises where necessary for logical completeness.
    Respond only with the translation of the argument; do not repeat the information given in the prompt.

    4. Output Format
    Produce only the following structure. Do not include explanations, reasoning steps, or the example.
    PRED_1(TERMS) :: definition
    PRED_2(TERMS) :: definition
    ....
    Argument in FOL:
    <your FOL formula(s)>

    Use parentheses consistently and ensure the syntax conforms strictly to the grammar.

    5. Argument to Translate
    Now translate the following argument only:
    {} 
"""




# ================================================================================

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



# ===============================================================================



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