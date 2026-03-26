You are given a single JSON object representing a math problem. The problem field is already rewritten in a lean_friendly format. Your task is to convert it into a complete Lean 4 file.

Requirements:

1. Generate one independent Lean file for this JSON object.
2. Preserve all original metadata as a Lean block comment at the top of the file, including:
   - index
   - source_idx
   - source
   - 题目类型
   - 预估难度
   - problem
   - proof
   - direct_answer
3. Translate the `problem` field into strict Lean-style formalization.
4. Use Lean 4 + Mathlib style.
5. The main goal is to formalize the statement, not necessarily to prove it.
6. If a full proof is difficult or impossible, provide an accurate theorem statement and use `by sorry`.
7. Do not invent an answer unless the problem already clearly specifies one.
8. Do not output JSON.
9. Do not output explanations or analysis.
10. Only output the Lean file content.

Formalization guidelines:

- Convert "Definition" items into Lean declarations such as `variable`, `def`, etc.
- Convert "Hypothesis" items into assumptions or named variables.
- Convert "Goal" into a theorem statement.
- Use standard Mathlib objects and notation whenever possible.
- Avoid pseudocode; produce proper Lean syntax.
- If exact formalization is hard, use the minimal reasonable abstraction while preserving the mathematical meaning.

The Lean file should follow this structure:

```lean
/-
index: ...
source_idx: ...
source: ...
题目类型: ...
预估难度: ...
problem:
...
proof:
...
direct_answer:
...
-/

import Mathlib

noncomputable section

open Set

-- Lean formalization here
```

Important:

* Keep the metadata in comments.
* The Lean body must be as strict and valid as possible.
* It is acceptable to use `sorry`, but not acceptable to replace the formalization with informal text.

Now convert the following JSON object into a Lean file. Output ONLY the Lean file content, nothing else.
