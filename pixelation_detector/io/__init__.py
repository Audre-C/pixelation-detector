"""
pixelation_detector.io
=======================

Frame input and frame-synchronization components.

Modules:
    frame_source — FrameSource abstraction + OpenCV-based FileFrameSource impl.
    sync         — FrameSynchronizer: block-correlation offset detection
                   (downsampled luma feature vectors + normalized
                   cross-correlation across a candidate offset sweep).
"""