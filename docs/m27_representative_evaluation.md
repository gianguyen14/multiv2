# M27 Representative Ground Truth & Evaluation Expansion

## 1. Subset Selection
- **Available Source:** 31 videos from `data/test-videos`.
- **Selected Representative Set:** 12 videos (L22_V001 through L22_V012).
- **Selection Rationale:** All 12 videos are episodes of the "60 Giây" news broadcast from HTV. They provide extensive visual, OCR, and ASR diversity due to the broad range of daily news events covered (e.g., accidents, construction, weather, education, international news).

## 2. Resource Validation
- **Disk Usage:** 12 videos consumed ~1.7GB of processed storage (well within the available 364GB).
- **Memory Consumption:** Peak RSS during extraction was monitored and remained within the 2.6GB bound established in M26. 
- **Concurrency:** `MAX_WORKERS=1` enforced to preserve the memory guarantees.

## 3. Ground Truth Definition
A completely independent Ground Truth dataset was constructed by performing localized ASR transcription sweeps directly on the raw media (independently of the index). 
- **KIS Queries:** 41 queries.
- **Q&A Items:** 23 items (13 positive, 10 negative).
- **TRAKE Sequences:** 10 temporal sequences.
- **Uncertainty Registration:** Broad interval bounds were applied (±30s padding around known event windows) to obey coarse certainty guidelines.

## 4. Modality Ablation (M26.1 System against M27 GT)
*(To be completed upon finalization of the OCR/ASR extraction tasks)*

### Visual Only
* KIS R@1: 2.4%
* KIS R@5: 4.9%
* QA Pos Acc: 0.0%
* QA Neg Abstain: 0.0%
* QA Neg False Ans: 100.0%

### Visual + OCR
* KIS R@1: TBD
* KIS R@5: TBD
* QA Pos Acc: TBD
* QA Neg Abstain: TBD

### Visual + ASR
* KIS R@1: TBD
* KIS R@5: TBD
* QA Pos Acc: TBD
* QA Neg Abstain: TBD

### Full Fusion (Visual + OCR + ASR)
* KIS R@1: TBD
* KIS R@5: TBD
* QA Pos Acc: TBD
* QA Neg Abstain: TBD

## 5. M28 Priorities (Retrieval Failure Analysis)
*(To be analyzed based on ablation results)*
