Propositional Logic

Propositional logic is a fundamental form of logic that deals with propositions (statements that are either true or false) 
and logical connectives. It's a building block for more complex logical systems and is crucial in AI for representing and 
reasoning about knowledge.

Key Components:

1. Propositional Symbols:
   - Represent atomic propositions (e.g., B for Burglary, E for Earthquake)
   - Can be either true or false

2. Logical Connectives:
   - NOT (¬): Negation
   - AND (∧): Conjunction
   - OR (∨): Disjunction
   - IMPLY (⇒): Implication
   - IF AND ONLY IF (⇔): Biconditional

3. Model:
   - A set of true/false assignments to all propositional symbols
   - Example: {B: True, E: False, A: True, M: False, J: True}

4. Sentences:
   - Formed by combining propositional symbols with connectives
   - Example: (E ∨ B) ⇒ A (If there's an Earthquake OR Burglary, then the Alarm will sound)

5. Truth Values:
   - Sentences are evaluated as either true or false with respect to a given model
   - Truth tables are used to define the meaning of connectives

Example Sentences:
- A ⇒ (J ∧ M): If the Alarm sounds, then John AND Mary will call
- J ⇔ M: John calls if and only if Mary calls
- J ⇔ ¬M: John calls if and only if Mary doesn't call

Key Points:
1. Propositional logic deals with absolute truth values, not probabilities.
2. It's limited to expressing facts about specific propositions and can't generalize.
3. Truth tables are used to evaluate complex sentences.
4. It forms the basis for more complex logics like First-Order Logic.


In AI and automated planning, propositional logic is used to represent states, actions, and goals in a simple, boolean manner. 
It allows for basic reasoning about the truth of statements in different scenarios, which is fundamental for planning and 
decision-making algorithms.


Truth Table


| P     | Q     | ¬P    | P ∧ Q | P ∨ Q | P ⇒ Q | P ⇔ Q |
|-------|-------|-------|-------|-------|-------|-------|
| False | False | True  | False | False | True  | True  |
| False | True  | True  | False | True  | True  | False |
| True  | False | False | False | True  | False | False |
| True  | True  | False | True  | True  | True  | True  |

Explanation of the truth table:

1. ¬P (NOT P):
   - True when P is False, False when P is True.

2. P ∧ Q (P AND Q):
   - True only when both P and Q are True.

3. P ∨ Q (P OR Q):
   - True when either P or Q (or both) are True.

4. P ⇒ Q (P IMPLIES Q):
   - False only when P is True and Q is False.
   - True in all other cases, including when P is False (vacuous truth).

5. P ⇔ Q (P IF AND ONLY IF Q):
   - True when P and Q have the same truth value (both True or both False).
   - False when P and Q have different truth values.


This truth table is fundamental in propositional logic as it defines the behavior of logical connectives for all possible 
combinations of truth values of P and Q. It allows us to evaluate complex logical statements by breaking them down into 
their constituent parts and applying these rules.



Q#1: (P is false, Q is false):
Correct answer: (P ∧ (P ⇒ Q)) ⇔ (¬ (¬P ∨ ¬Q))

Explanation:
- P ∧ (P ⇒ Q) is false because P is false.
- ¬ (¬P ∨ ¬Q) is false because ¬P ∨ ¬Q is true (both ¬P and ¬Q are true).
- The equivalence of two false statements is true.

Q#2(P is false, Q is true):
Correct answer: (P ∧ (P ⇒ Q)) ⇔ (¬ (¬P ∨ ¬Q))

Explanation:
- P ∧ (P ⇒ Q) is false because P is false.
- ¬ (¬P ∨ ¬Q) is false because ¬P ∨ ¬Q is true (¬P is true).
- The equivalence of two false statements is true.

Q#3(P is true, Q is false):
Correct answer: (P ∧ (P ⇒ Q)) ⇔ (¬ (¬P ∨ ¬Q))

Explanation:
- P ∧ (P ⇒ Q) is false because P ⇒ Q is false when P is true and Q is false.
- ¬ (¬P ∨ ¬Q) is false because ¬P ∨ ¬Q is true (¬Q is true).
- The equivalence of two false statements is true.

Q#4(P is true, Q is true):
Correct answers: All three statements

Explanation:
- P ∧ (P ⇒ Q) is true because both P and P ⇒ Q are true.
- ¬ (¬P ∨ ¬Q) is true because ¬P ∨ ¬Q is false (both ¬P and ¬Q are false).
- (P ∧ (P ⇒ Q)) ⇔ (¬ (¬P ∨ ¬Q)) is true because both sides are true.


In all cases, the equivalence statement is true because it represents a tautology - a statement that is always true regardless 
of the truth values of its components. This demonstrates an important principle in propositional logic about logical 
equivalences.


Q#1:
Given that (E v B) ⇔ A, A ⇔ (J ∧ M), and B are all true statements, what is the value of the variable E?

Correct Answer: Cannot be determined
Explanation: While we know B is true, E could be either true or false. If E is true, (E v B) is true. If E is false, (E v B) 
is still true because B is true. Therefore, we can't determine E's specific value.

Q#2:
Given that (E v B) ⇔ A, A ⇔ (J ∧ M), and B are all true statements, what is the value of the variable B?

Correct Answer: True
Explanation: The question directly states that B is a true statement.

Q#3:
Given that (E v B) ⇔ A, A ⇔ (J ∧ M), and B are all true statements, what is the value of the variable A?

Correct Answer: True
Explanation: Since (E v B) ⇔ A is true and B is true, (E v B) is true. Therefore, A must be true for the equivalence to hold.

Q#4:
Given that (E v B) ⇔ A, A ⇔ (J ∧ M), and B are all true statements, what is the value of the variable J?

Correct Answer: Cannot be determined
Explanation: We know A is true, and A ⇔ (J ∧ M) is true. This means (J ∧ M) must be true, but we can't determine if J is true 
or false without knowing M's value.

Q#5:
Given that (E v B) ⇔ A, A ⇔ (J ∧ M), and B are all true statements, what is the value of the variable M?

Correct Answer: Cannot be determined
Explanation: Similar to J, we know (J ∧ M) is true, but we can't determine M's specific value without knowing J's value.


1. Valid Sentence:
   - Definition: A sentence that is true in every possible model.
   - Explanation: No matter what truth values you assign to the individual propositional symbols, the sentence always evaluates to true.
   - Example: P ∨ ¬P (P or not P) is valid because it's always true regardless of whether P is true or false.
   - Also known as a tautology.

2. Satisfiable Sentence:
   - Definition: A sentence that is true in at least one model.
   - Explanation: There exists at least one assignment of truth values to the propositional symbols that makes the sentence true.
   - Example: P ∧ Q is satisfiable because it's true when both P and Q are true, even though it's not true in all cases.
   - Note: All valid sentences are satisfiable, but not all satisfiable sentences are valid.

3. Unsatisfiable Sentence:
   - Definition: A sentence that cannot be true in any possible model.
   - Explanation: No matter what truth values you assign to the propositional symbols, the sentence always evaluates to false.
   - Example: P ∧ ¬P (P and not P) is unsatisfiable because it's always false.
   - Also known as a contradiction.

Key Points:
- A valid sentence is true in all models.
- A satisfiable sentence is true in at least one model.
- An unsatisfiable sentence is false in all models.
- Every sentence is either satisfiable or unsatisfiable.
- Valid sentences are a subset of satisfiable sentences.


These concepts are crucial in logical reasoning and form the basis for many algorithms in artificial intelligence, including 
automated theorem proving and logical inference systems.


Q#1:
Is the following sentence valid, satisfiable, or unsatisfiable? P ∨ ¬P

Correct Answer: Valid
Explanation: This sentence is a tautology, which is always true regardless of the truth value of P. If P is true, P ∨ ¬P 
is true. If P is false, ¬P is true, so P ∨ ¬P is still true. Therefore, it's valid (true in all possible models).


Q#2:
Is the following sentence valid, satisfiable, or unsatisfiable? P ∧ ¬P

Correct Answer: Unsatisfiable
Explanation: This sentence is a contradiction. It can never be true because P cannot be simultaneously true and false. 
Therefore, it's unsatisfiable (false in all possible models).


Q#3:
Is the following sentence valid, satisfiable, or unsatisfiable? P ∨ (P ∧ Q)

Correct Answer: Not provided in the image, but the correct answer would be Satisfiable.
Explanation: This sentence is true whenever P is true, regardless of Q's value. It's also true when both P and Q are true. 
However, it's false when P is false and Q is true. Therefore, it's satisfiable (true in some models, but not all).


Q#4:
Is the following sentence valid, satisfiable, or unsatisfiable? (P ∧ Q) ∨ (P ∧ ¬Q)

Correct Answer: Not provided in the image, but the correct answer would be Satisfiable.
Explanation: This sentence is equivalent to P ∧ (Q ∨ ¬Q). It's true whenever P is true, regardless of Q's value. However, 
it's false when P is false. Therefore, it's satisfiable (true in some models, but not all).


Q#5:
Is the following sentence valid, satisfiable, or unsatisfiable? ((Food ∧ Party) ∨ (Drinks ∧ Party)) ⇔ ((Food ∨ Drinks) ∧ Party)


Correct Answer: Not provided in the image, but the correct answer would be Valid.
Explanation: This sentence is a logical equivalence. The left side is true when there's a party with food or drinks (or both). 
The right side expresses the same condition. This equivalence holds for all possible combinations of truth values for Food, Drinks, 
and Party. Therefore, it's valid (true in all possible models).