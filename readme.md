# Pixelation Detector

A Python-based video quality monitoring tool designed to detect pixelation and visual degradation in broadcast video streams by comparing a reference video against a test video.

## Overview

This project was developed to investigate automated detection of pixelation artifacts commonly observed in video transmission systems. The detector analyzes video streams frame-by-frame and computes objective quality metrics to identify periods where the test stream significantly diverges from the reference stream.

The system is intended as a proof-of-concept for broadcast and satellite communication environments where rapid identification of visual impairments can help operators diagnose transmission issues.

## Features

* Frame-by-frame video comparison
* PSNR (Peak Signal-to-Noise Ratio) analysis
* SSIM (Structural Similarity Index) analysis
* Blockiness Difference Score (BDS) calculation
* Scene cut detection
* Event-based pixelation detection
* Confidence scoring system
* CSV and JSON reporting
* Visualization generation:

  * Metric time-series plots
  * Detection confidence timelines
  * Event overlays
  * Sanity-check visualizations

## Project Structure

```text
pixelation_detector/
├── alarms/
├── detection/
├── io/
├── metrics/
├── visualization/
├── config.py
├── pipeline.py
└── main.py
```

## Detection Methodology

The detector compares a reference video stream with a test stream and evaluates:

### PSNR

Measures pixel-level differences between corresponding frames.

### SSIM

Measures structural similarity and perceptual quality differences.

### Blockiness Difference Score (BDS)

Estimates the presence of block-based compression artifacts commonly associated with pixelation.

### Event Generation

Potential pixelation events are generated when quality metrics exceed configured thresholds for a sustained period.

### Confidence Scoring

Multiple metrics are combined into a confidence score to reduce false positives and improve robustness.

## Output Artifacts

After execution, the detector generates:

### Metrics CSV

Contains frame-level measurements:

* Timestamp
* PSNR
* SSIM
* Blockiness score
* Confidence score

### Events CSV

Contains detected degradation events:

* Start time
* End time
* Duration
* Severity level
* Detection confidence

### Summary JSON

Contains overall analysis statistics and event summaries.

### Visualizations

Generated plots may include:

* Metric trends over time
* Detection confidence timeline
* Event overlay images
* Reference-vs-reference sanity checks

## Installation

### Requirements

* Python 3.10+
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Scikit-image

### Setup

```bash
git clone https://github.com/Audre-C/pixelation-detector.git
cd pixelation-detector

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

Example:

```bash
python main.py \
    --reference data/original.mp4 \
    --test data/pixelated.mp4
```

Outputs will be written to the configured output directory.

## Current Status

The project is currently under active development as part of an internship project focused on automated detection of video pixelation in broadcast and satellite communication systems.

Current testing has primarily been performed using synthetically generated pixelation artifacts. Future work includes validation and tuning using real-world transmission impairments captured from operational broadcast streams.

## Future Improvements

* Real-time transport stream integration
* Live monitoring support
* Machine-learning-based classification
* Automatic threshold optimization
* Additional video impairment detection (freezing, macroblocking, corruption, packet loss)

## Author

Audre Caraig

Electrical Engineering Student

Internship Project – Broadcast Operations 
