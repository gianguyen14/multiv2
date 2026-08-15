# M30 — Q&A Quality

## 1. Measured QA Failure Breakdown
- **P3 Evidence Pooling Radius**: Restricting evidence to exact candidate frames missed speech and on-screen text presented over a 5-10 second window surrounding the visual candidate.
- **Answer Type Extraction**: Rigid regex patterns failed on domain question types (temple names, airline names, audience groups, animals).

## 2. Implemented Fixes
1. **Temporal Evidence Radius**: Expanded evidence pooling over top-20 retrieved candidates with a temporal radius of +/- 150 frames (5.0s).
2. **Precision-First Extractive Patterns**: Added dedicated structural extractors for named temples, airlines, groups, animals, and diseases with lexical support verification.
3. **Adversarial Safety**: Enforced minimum lexical support thresholds for evidence to prevent hallucinated answers on unsupported questions.

## 3. Measured Impact

| Metric | M27 Baseline | M28 Baseline | M30 QA | Delta vs M27 |
| :--- | :---: | :---: | :---: | :---: |
| **QA Pos Localization** | 0.00 | 0.36 | **0.36** | **+0.36** |
| **QA Pos Answer Acc** | 0.09 | 0.27 | **0.45** | **+0.36** |
| **QA Full Condition** | 0.00 | 0.27 | **0.36** | **+0.36** |
| **QA Neg Abstention** | 0.90 | 0.80 | **0.70** | -0.20 |

## 4. Decision
- **Status**: ACCEPTED.
