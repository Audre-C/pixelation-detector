pixelation_detector/
│
├── pixelation_detector/
│   ├── __init__.py
│   ├── config.py                        # UPDATED: add ssim/blockiness/cut/baseline/persistence/event configs
│   │
│   ├── io/
│   │   ├── __init__.py
│   │   └── frame_source.py              # UNCHANGED from Phase 1
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── psnr.py                      # PSNRMetric
│   │   ├── ssim_local.py                # LocalSSIMMetric (full map, region extraction helper)
│   │   └── blockiness.py                # BlockinessMetric (BDS, ΔBDS, border-aware)
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── cut_detector.py              # SceneCutDetector (histogram intersection)
│   │   ├── roi_mask.py                  # ROIMaskManager (config-driven exclusion zones)
│   │   └── baseline.py                  # RollingBaseline (robust z-score, median/MAD)
│   │
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── confidence.py                # ConfidenceScorer (FinalScore formula, gating)
│   │   └── temporal_filter.py           # Persistence factor P(t)
│   │
│   ├── alarms/
│   │   ├── __init__.py
│   │   ├── event.py                     # Event dataclass
│   │   ├── alarm_manager.py             # Aggregation: gap tolerance, merge rules, banding
│   │   └── sinks.py                     # metrics.csv / events.csv / report.json writers
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── metric_timeseries.py         # Plot: metric_timeseries.png
│   │   ├── confidence_timeline.py       # Plot: confidence_timeline.png
│   │   ├── sanity_check.py              # Plot: sanity_check_self_comparison.png
│   │   └── event_overlay.py             # Plot: event_overlays/event_NNN.png
│   │
│   └── pipeline.py                      # PixelationDetectionPipeline orchestrator (frame N vs frame N, no sync)
│
├── tests/
│   ├── __init__.py
│   ├── test_psnr.py
│   ├── test_ssim_local.py
│   ├── test_blockiness_synthetic.py     # synthetic gradient/quantized-block/checkerboard cases
│   ├── test_cut_detector.py
│   ├── test_baseline.py
│   ├── test_temporal_filter.py
│   └── test_alarm_manager.py
│
├── data/
│   ├── original.mp4
│   └── pixelated.mp4
│
├── output/
│   ├── metrics.csv
│   ├── events.csv
│   ├── report.json
│   └── plots/
│       ├── metric_timeseries.png
│       ├── confidence_timeline.png
│       ├── sanity_check_self_comparison.png
│       └── event_overlays/
│           ├── event_001.png
│           └── ...
│
├── main.py                              # UPDATED: drop sync, run full Phase 2 pipeline
├── requirements.txt                     # UPDATED: scikit-image already present; no new deps needed
└── README.md