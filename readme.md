pixelation_detector/
├── pixelation_detector/
│   ├── __init__.py                  # package docstring, phase status tracker
│   ├── config.py                    # ALL tunables (sync used now; metrics/scoring/alarms defined but inert)
│   ├── io/
│   │   ├── __init__.py
│   │   ├── frame_source.py          # FrameSource ABC + FileFrameSource (OpenCV)
│   │   └── sync.py                  # FrameSynchronizer (pHash + cross-correlation)
│   ├── metrics/__init__.py          # placeholder — Phase 2
│   ├── detection/__init__.py        # placeholder — Phase 3
│   ├── scoring/__init__.py          # placeholder — Phase 6
│   ├── alarms/__init__.py           # placeholder — Phase 7
│   └── visualization/__init__.py    # placeholder — Phase 5/8
├── tests/
│   └── __init__.py                  # placeholder — Phase 2 onward
├── data/
│   ├── original.mp4                 # ⚠ my synthetic placeholder — replace with your real file
│   └── pixelated.mp4                # ⚠ my synthetic placeholder — replace with your real file
├── output/
│   └── sync_report.json             # generated diagnostics report (example output from this run)
├── main.py                          # Phase 0/1 CLI entry point
└── requirements.txt