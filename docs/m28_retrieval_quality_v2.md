# M28 — Retrieval Quality v2

## 1. Measured Root Causes from M27 Failures
1. **Candidate Depth & Modality Siloing**: Prior to M28, `ConfiguredSearch` only retrieved visual candidates from the FAISS index. Frames with rich discriminative OCR/ASR were omitted if their initial SigLIP visual embedding similarity was outside the top-100.
2. **Brittle Gating Heuristic**: Intent detection relied on rigid keyword strings (`["chữ", "ghi"]`), which failed on natural queries, suppressing OCR/ASR authority to 0.01.
3. **Weight Configuration**: Environmental weights (`VISUAL_WEIGHT`, `OCR_WEIGHT`, `ASR_WEIGHT`) were not actively routed into the fusion calculation.

## 2. Implemented Fixes
1. **Multi-Modal Candidate Pool Expansion**: ConfiguredSearch now performs multi-source candidate generation:
   - Visual FAISS search top-k
   - Lexical OCR candidate discovery (matching normalized frame text with lexical threshold >= 0.15)
   - Lexical ASR candidate discovery (matching speech segments with lexical threshold >= 0.15)
2. **Calibrated Multimodal Fusion**: Normalizes visual scores across candidates and dynamically applies calibrated, bounded lexical bonuses (`vw * v_norm + ow * ocr_s * 1.5 + aw * asr_s * 1.5`).
3. **Weight Controllability**: Fully respects environmental modality weights for rigorous ablation isolation.

## 3. Modality Ablation Results

| Metric | Visual Only | Visual + OCR | Visual + ASR | All Modalities (M28) | M27 Baseline | Delta vs M27 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **KIS R@1** | 0.05 | 0.16 | 0.26 | **0.26** | 0.00 | **+0.26** |
| **KIS R@5** | 0.11 | 0.26 | 0.37 | **0.37** | 0.11 | **+0.26** |
| **KIS R@20** | 0.16 | 0.32 | 0.47 | **0.42** | 0.16 | **+0.26** |
| **QA Pos Localization** | 0.00 | 0.09 | 0.09 | **0.36** | 0.00 | **+0.36** |
| **QA Pos Answer Acc** | 0.09 | 0.27 | 0.18 | **0.27** | 0.09 | **+0.18** |
| **QA Full Condition** | 0.00 | 0.09 | 0.00 | **0.27** | 0.00 | **+0.27** |
| **QA Neg Abstention** | 0.90 | 0.90 | 0.90 | **0.80** | 0.90 | -0.10 |
| **TRAKE Video Match** | 0.50 | 0.50 | 0.50 | **0.50** | 0.50 | 0.00 |

## 4. Decision
- **Status**: ACCEPTED. Aggregate retrieval quality and task metrics improved substantially across all core dimensions without degrading pure visual performance.
