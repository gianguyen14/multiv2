# Videos_L24_a — human-first visual analysis

This file was written **before** running the retrieval-system comparison. The archive was downloaded and extracted by GitHub Actions. Every one of the 43 videos was reviewed through a 16-frame uniform overview plus four larger frames at 20/40/60/80% of duration. The count benchmark in `human_count_gt.jsonl` is therefore human visual ground truth anchored to a reviewed timestamp; it does **not** reconstruct frame IDs from FPS.

The review is visual-only. It is not claimed to be a full audio transcript or exhaustive frame-by-frame narration.

| Video | Human visual summary | Count anchor |
|---|---|---|
| L24_V002 | Night arena dragon-dance routine with a long yellow/green dragon, many green-clad performers, judges and a decorated gate. | 421.79s — 2 large red lanterns |
| L24_V003 | Night dragon routine with black/yellow performers; later the team gathers in the arena. | 610.54s — 4 black/yellow performers |
| L24_V004 | Night green/yellow dragon performance followed by judging. | 749.04s — 3 score cards |
| L24_V005 | Daytime yellow/red lion high-pole routine with a decorative arch and lanterns. | 375.13s — 2 large red lanterns |
| L24_V006 | Daytime red-lion high-pole routine. | 227.51s — 4 clearly visible round pedestal tops (medium-confidence annotation) |
| L24_V007 | Daytime yellow-lion pole routine. | 228.19s — 2 large triangular flags |
| L24_V008 | Daytime white/cream lion, plant obstacle and team/judging-table scenes. | 449.69s — 3 red-uniformed standing members |
| L24_V009 | Daytime yellow lion with green cloth/scarf. | 103.49s — 2 yellow-uniformed people side by side |
| L24_V010 | Daytime white/red lion and red-uniformed percussion team. | 281.24s — 3 red-clad team members |
| L24_V011 | Daytime white/cream lion with a wide pole arena and percussion team. | 259.80s — 3 red-clad musicians |
| L24_V012 | Short interview-focused segment with participants. | 19.44s — 2 white-shirted people in the front row |
| L24_V013 | Daytime red-lion segment with orange-uniformed percussion and crowd close-ups. | 104.43s — 4 orange-clad members |
| L24_V014 | Daytime white-lion pole routine. | 105.14s — 2 large drums |
| L24_V015 | Night white lion plus a long colourful dragon/prop and percussion. | 238.86s — 2 black/yellow performers |
| L24_V016 | Night white-lion high-pole routine. | 450.80s — 2 large triangular flags |
| L24_V017 | Short interview plus a yellow-lion pole segment at night. | 91.82s — 3 blue-covered judge tables |
| L24_V018 | Night yellow lion with green features on the numbered pole course. | 131.95s — 5 visible white number signs |
| L24_V019 | Daytime yellow lion with green cloth/scarf and numbered poles. | 104.08s — 6 scoped number signs, 1 through 9 (medium confidence) |
| L24_V020 | Daytime yellow lion with a red/yellow percussion team. | 99.46s — 4 red/yellow team members |
| L24_V021 | Daytime yellow lion with patterned team uniforms and pole course. | 113.43s — 6 scoped number signs, 1 through 9 (medium confidence) |
| L24_V022 | Daytime yellow lion with red-uniformed percussion. | 472.52s — 3 red-clad musicians |
| L24_V023 | Daytime black/dark-green lion on the pole course. | 120.92s — 2 large triangular flags |
| L24_V024 | Daytime white-lion pole routine with decorated gate. | 338.49s — 2 large red lanterns |
| L24_V025 | Daytime gray/white lion with green/red team. | 224.98s — 4 large red landing mats |
| L24_V026 | Daytime yellow-lion pole routine. | 341.92s — 2 large red lanterns |
| L24_V027 | Night white/red-black lion on numbered poles. | 113.72s — 8 visible white number signs (medium confidence) |
| L24_V028 | Night yellow-lion pole routine. | 227.66s — 4 large red landing mats |
| L24_V029 | Night white/gray lion with judges behind the course. | 110.90s — 2 blue-covered judge tables |
| L24_V030 | Night white/red lion with high balancing. | 516.10s — 4 large red landing mats |
| L24_V031 | Night white/red/green lion, including close pole-course views. | 341.43s — 3 red/white members (medium confidence) |
| L24_V032 | Night yellow lion with flower props and numbered poles. | 143.17s — 2 visible white number signs |
| L24_V033 | Night black/white/green lion. | 234.64s — 2 large triangular flags |
| L24_V035 | Night yellow-lion routine with black-clad musicians near the foreground. | 132.35s — 4 large red landing mats |
| L24_V036 | Shorter night white/pink-lion routine. | 117.50s — 4 large red landing mats |
| L24_V037 | Night yellow-lion pole routine with flower props. | 511.10s — 2 flower bouquets (medium confidence) |
| L24_V038 | Night white-lion routine using a blue-draped hanging/fish prop. | 406.44s — 2 vertical posts on the blue-draped prop frame |
| L24_V039 | Night white lion with orange/blue trim. | 208.03s — 1 large red lantern clearly visible |
| L24_V040 | Very short vertical night highlight of a white lion. | 15.11s — 2 large lion eyes |
| L24_V041 | Very short vertical night highlight of a white/green lion. | 7.17s — 2 large lion eyes |
| L24_V042 | Daytime white/red-lion pole routine. | 111.28s — 2 judges seated together behind the lion |
| L24_V043 | Daytime white/orange lion, red-uniformed percussion and flower props. | 137.62s — 2 adjacent drums |
| L24_V044 | Very short vertical yellow-lion highlight. | 7.07s — 2 visible support poles under the lion |
| L24_V045 | Very short vertical yellow-lion highlight. | 6.51s — 2 visible support poles, labelled 4 and 6 |

## Comparison protocol

The system pass must not alter these human answers after seeing model output. For each question we will report separately:

- scene/video retrieval: whether the correct video is retrieved and at what rank;
- temporal localization: distance in seconds from the human evidence timestamp, without fabricating an authoritative frame ID from FPS;
- numerical answer: normalized numeric exact match;
- answer coverage/abstention;
- `retrieval-correct / count-answer-wrong` cases;
- breakdown by target type such as people, musicians, props, number signs, lanterns, drums and landing mats.

This separation matters because the current QA path is extractive over OCR/ASR evidence rather than a pixel-level visual counting model. The comparison must therefore distinguish **finding the right scene** from **answering the count**.
