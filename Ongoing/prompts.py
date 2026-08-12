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