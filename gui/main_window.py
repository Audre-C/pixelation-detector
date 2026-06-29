"""
gui/main_window.py
==================

MainWindow — the broadcast-operations demonstration UI.

Pure presentation layer. It SUBSCRIBES to a DetectionWorker's Qt signals (and a
PlaybackWorker's) and renders them; it contains no detection logic and never
calls the detector directly. All heavy work happens on the worker threads, so
the UI stays responsive.

Video grid (2x2):
  - Top row    : REFERENCE / TEST at native fps (real-time broadcast view),
                 driven by the decode-only PlaybackWorker.
  - Bottom row : REFERENCE / TEST as the detector sees them (paced to analysis
                 throughput), driven by the DetectionWorker.

Also: top status bar, live metrics panel, a small centered alarm pill (a free
child widget, so it never reflows the layout), and a newest-first event log.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.worker import (
    ActiveEvent,
    DetectionWorker,
    EventRecord,
    FramePayload,
    StatsPayload,
)
from gui.playback_worker import PlaybackFrame, PlaybackWorker

logger = logging.getLogger(__name__)


_SEVERITY_COLORS = {
    "NONE": "#3a3f44",
    "LOW": "#c9a227",
    "MEDIUM": "#e07b00",
    "HIGH": "#d62728",
}
_STATUS_COLORS = {
    "ONLINE": "#2e7d32",
    "MONITORING": "#1565c0",
    "ALARM ACTIVE": "#c62828",
}


def _fmt(value: Optional[float], fmt: str) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and value != value:  # NaN
        return "—"
    return format(value, fmt)


def _bgr_to_qimage(frame_bgr: np.ndarray) -> QImage:
    if frame_bgr.ndim == 2:
        rgb = np.ascontiguousarray(frame_bgr)
        h, w = rgb.shape
        return QImage(rgb.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class VideoPanel(QWidget):
    """Titled, aspect-ratio-preserving video display with an optional overlay."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)

        self._title = QLabel(title)
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setStyleSheet(
            "color: #e6e6e6; font-size: 14px; font-weight: 600; "
            "padding: 3px; background: #1b1f24;"
        )

        self._video = QLabel("Waiting for stream…")
        self._video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video.setMinimumSize(240, 135)
        self._video.setStyleSheet("background: #000000; color: #555;")
        self._video.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._title)
        layout.addWidget(self._video, stretch=1)

        self._source_pixmap: Optional[QPixmap] = None
        self._region_bbox: Optional[tuple] = None

    def set_frame(self, frame_bgr: np.ndarray, region_bbox: Optional[tuple]) -> None:
        self._source_pixmap = QPixmap.fromImage(_bgr_to_qimage(frame_bgr))
        self._region_bbox = region_bbox
        self._render()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._render()

    def _render(self) -> None:
        if self._source_pixmap is None:
            return
        scaled = self._source_pixmap.scaled(
            self._video.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if self._region_bbox is not None and self._source_pixmap.width() > 0:
            scale = scaled.width() / self._source_pixmap.width()
            x, y, w, h = self._region_bbox
            painter = QPainter(scaled)
            pen = QPen(QColor("#ff2d2d"))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawRect(
                int(x * scale), int(y * scale), int(w * scale), int(h * scale)
            )
            painter.end()
        self._video.setPixmap(scaled)


class _StatTile(QFrame):
    """A small labelled value tile used in the top bar and status panel."""

    def __init__(self, caption: str, parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("background: #1b1f24; border-radius: 6px;")
        self._caption = QLabel(caption)
        self._caption.setStyleSheet("color: #8b949e; font-size: 11px;")
        self._value = QLabel("—")
        self._value.setStyleSheet(
            "color: #e6e6e6; font-size: 18px; font-weight: 700;"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(2)
        layout.addWidget(self._caption)
        layout.addWidget(self._value)

    def set_value(self, text: str, color: Optional[str] = None) -> None:
        self._value.setText(text)
        self._value.setStyleSheet(
            f"color: {color or '#e6e6e6'}; font-size: 18px; font-weight: 700;"
        )


class MainWindow(QMainWindow):
    """The top-level demonstration window."""

    def __init__(
        self,
        detection_worker: DetectionWorker,
        playback_worker: PlaybackWorker,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._worker = detection_worker
        self._playback = playback_worker

        self.setWindowTitle("Broadcast Pixelation Monitor")
        self.resize(1480, 960)
        self.setStyleSheet("background: #0d1117;")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        root.addWidget(self._build_top_bar())
        root.addWidget(self._build_video_area(), stretch=4)
        root.addWidget(self._build_lower_area(), stretch=2)

        # Small floating alarm pill — free child of central, not in any layout,
        # so toggling it never reflows or resizes the window.
        self._notif = QLabel(central)
        self._notif.setVisible(False)
        self._notif.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._notif.setStyleSheet(self._notif_style("MEDIUM"))

        # Wire detection worker.
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.event_logged.connect(self._on_event)
        self._worker.stats_updated.connect(self._on_stats)
        self._worker.worker_error.connect(self._on_error)

        # Wire playback worker.
        self._playback.frames_ready.connect(self._on_playback)
        self._playback.worker_error.connect(self._on_error)

    # -- construction helpers ---------------------------------------------

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        title = QLabel("PIXELATION DETECTION  •  BROADCAST OPS")
        title.setStyleSheet(
            "color: #f0f6fc; font-size: 20px; font-weight: 800; "
            "letter-spacing: 1px;"
        )
        layout.addWidget(title)
        layout.addStretch(1)

        self._tile_status = _StatTile("SYSTEM STATUS")
        self._tile_fps = _StatTile("PROCESSING FPS")
        self._tile_elapsed = _StatTile("ELAPSED RUNTIME")
        self._tile_status.set_value("ONLINE", _STATUS_COLORS["ONLINE"])
        for tile in (self._tile_status, self._tile_fps, self._tile_elapsed):
            layout.addWidget(tile)
        return bar

    def _build_video_area(self) -> QWidget:
        area = QWidget()
        grid = QGridLayout(area)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10)

        # Top row: real-time broadcast view (PlaybackWorker).
        self._panel_ref_live = VideoPanel("REFERENCE — LIVE (real-time fps)")
        self._panel_test_live = VideoPanel("TEST — LIVE (real-time fps)")
        # Bottom row: detector view (DetectionWorker).
        self._panel_ref = VideoPanel("REFERENCE — DETECTOR (analyzed)")
        self._panel_test = VideoPanel("TEST — DETECTOR (analyzed)")

        grid.addWidget(self._panel_ref_live, 0, 0)
        grid.addWidget(self._panel_test_live, 0, 1)
        grid.addWidget(self._panel_ref, 1, 0)
        grid.addWidget(self._panel_test, 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)

        self._video_area = area
        return area

    def _build_lower_area(self) -> QWidget:
        area = QWidget()
        layout = QHBoxLayout(area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self._build_status_panel(), stretch=1)
        layout.addWidget(self._build_event_log(), stretch=2)
        return area

    def _build_status_panel(self) -> QWidget:
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        panel.setStyleSheet("background: #11161c; border-radius: 8px;")
        grid = QGridLayout(panel)
        grid.setContentsMargins(12, 12, 12, 12)
        grid.setSpacing(8)

        header = QLabel("LIVE METRICS")
        header.setStyleSheet("color: #8b949e; font-size: 12px; font-weight: 700;")
        grid.addWidget(header, 0, 0, 1, 2)

        self._tile_frame = _StatTile("FRAME #")
        self._tile_psnr = _StatTile("PSNR (dB)")
        self._tile_ssim = _StatTile("SSIM")
        self._tile_bds = _StatTile("ΔBDS")
        self._tile_score = _StatTile("FINAL SCORE")
        self._tile_sev = _StatTile("SEVERITY")

        grid.addWidget(self._tile_frame, 1, 0)
        grid.addWidget(self._tile_psnr, 1, 1)
        grid.addWidget(self._tile_ssim, 2, 0)
        grid.addWidget(self._tile_bds, 2, 1)
        grid.addWidget(self._tile_score, 3, 0)
        grid.addWidget(self._tile_sev, 3, 1)
        grid.setRowStretch(4, 1)
        return panel

    def _build_event_log(self) -> QWidget:
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        panel.setStyleSheet("background: #11161c; border-radius: 8px;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        header = QLabel("EVENT LOG  (newest first)")
        header.setStyleSheet("color: #8b949e; font-size: 12px; font-weight: 700;")
        layout.addWidget(header)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Timestamp (s)", "Frame", "Severity", "Peak score", "Duration"]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.setStyleSheet(
            "QTableWidget { background: #0d1117; color: #e6e6e6; "
            "gridline-color: #21262d; font-size: 13px; }"
            "QHeaderView::section { background: #1b1f24; color: #8b949e; "
            "padding: 4px; border: none; font-weight: 700; }"
        )
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self._table)
        return panel

    # -- alarm pill --------------------------------------------------------

    @staticmethod
    def _notif_style(severity: str) -> str:
        color = _SEVERITY_COLORS.get(severity, "#e07b00")
        return (
            f"background: {color}; color: #ffffff; font-size: 13px; "
            f"font-weight: 700; padding: 6px 12px; border-radius: 14px;"
        )

    def _reposition_notif(self) -> None:
        """Anchor the pill to the top-center of the video area."""
        if not self._notif.isVisible():
            return
        area_geo = self._video_area.geometry()
        self._notif.adjustSize()
        x = area_geo.x() + (area_geo.width() - self._notif.width()) // 2
        y = area_geo.y() + 16
        self._notif.move(x, y)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._reposition_notif()

    # -- slots -------------------------------------------------------------

    def _on_playback(self, payload: PlaybackFrame) -> None:
        self._panel_ref_live.set_frame(payload.reference_bgr, None)
        self._panel_test_live.set_frame(payload.test_bgr, None)

    def _on_frame(self, payload: FramePayload) -> None:
        self._panel_ref.set_frame(payload.reference_bgr, None)
        self._panel_test.set_frame(payload.test_bgr, payload.region_bbox)

        self._tile_frame.set_value(str(payload.frame_index))
        self._tile_psnr.set_value(_fmt(payload.psnr_db, ".2f"))
        self._tile_ssim.set_value(_fmt(payload.mean_ssim, ".4f"))
        self._tile_bds.set_value(_fmt(payload.delta_bds, ".3f"))
        self._tile_score.set_value(_fmt(payload.final_score, ".1f"))
        sev = payload.severity or "NONE"
        self._tile_sev.set_value(sev, _SEVERITY_COLORS.get(sev, "#e6e6e6"))

        self._update_notif(payload.active_event)

    def _update_notif(self, active: Optional[ActiveEvent]) -> None:
        if active is None:
            if self._notif.isVisible():
                self._notif.setVisible(False)
            return
        self._notif.setStyleSheet(self._notif_style(active.severity))
        self._notif.setText(
            f"●  Pixelation — event #{active.event_id} · "
            f"score {active.current_score:.0f} · {active.severity}"
        )
        if not self._notif.isVisible():
            self._notif.setVisible(True)
            self._notif.raise_()
        self._reposition_notif()

    def _on_event(self, record: EventRecord) -> None:
        ts = (
            f"{record.start_time_s:.2f}"
            if record.start_time_s == record.start_time_s  # not NaN
            else "—"
        )
        row_values = [
            ts,
            str(record.start_frame),
            record.severity,
            f"{record.peak_score:.1f}",
            f"{record.duration_frames}f",
        ]
        self._table.insertRow(0)
        for col, value in enumerate(row_values):
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if col == 2:
                item.setForeground(
                    QColor(_SEVERITY_COLORS.get(record.severity, "#e6e6e6"))
                )
            self._table.setItem(0, col, item)

    def _on_stats(self, stats: StatsPayload) -> None:
        self._tile_status.set_value(
            stats.status, _STATUS_COLORS.get(stats.status, "#e6e6e6")
        )
        self._tile_fps.set_value(f"{stats.processing_fps:.1f}")
        minutes, seconds = divmod(int(stats.elapsed_s), 60)
        self._tile_elapsed.set_value(f"{minutes:02d}:{seconds:02d}")

    def _on_error(self, message: str) -> None:
        self._tile_status.set_value("ERROR", "#c62828")
        self._panel_ref._video.setText(f"ERROR:\n{message}")
        self._panel_test._video.setText("stream unavailable")

    # -- lifecycle ---------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        logger.info("MainWindow closing; stopping workers.")
        self._worker.stop()
        self._playback.stop()
        self._worker.wait(3000)
        self._playback.wait(3000)
        super().closeEvent(event)