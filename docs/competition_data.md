# Competition Data Contract

`data/competition/` is the internal import boundary for evaluation data. It is not an assertion about the organizer's eventual export format.

```text
data/competition/
├── videos/
└── ground_truth/
    ├── kis.jsonl
    ├── qa.jsonl
    └── trake.jsonl
```

Each JSONL file contains one JSON object per line.

## KIS

```json
{"query_id":"kis-001","query":"a cyclist crosses the road","video_id":"video_001","start_frame":500,"end_frame":510}
```

`start_frame` and `end_frame` are inclusive authoritative source-frame ordinals unless an imported dataset explicitly documents a different policy and converts it before writing this file.

## Q&A

```json
{"query_id":"qa-001","query":"What color is the vehicle?","video_id":"video_002","start_frame":800,"end_frame":900,"answers":["màu xanh","blue"]}
```

`answers` contains accepted normalized semantic variants. Raw organizer labels should be preserved in the adapter source data.

## TRAKE

```json
{"query_id":"trake-001","events":["person enters","person sits"],"video_id":"video_003","windows":[{"start_frame":95,"end_frame":105},{"start_frame":145,"end_frame":155}]}
```

Events and windows are positional and must have equal non-zero lengths. Windows are inclusive.

`backend.app.competition_data.load_jsonl` validates these contracts. Organizer-specific imports implement `OrganizerAdapter.convert(source_path, destination_root)` and write only this internal representation, keeping organizer parsing outside retrieval and scoring code.
