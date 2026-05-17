#!/usr/bin/env python3
"""
Reforma Traffic Counter  v2
─────────────────────────────────────────────────────────────────────────────
Detecta Ecobici, Bicycle, Person con modelo .eim (QNN/Hexagon NPU).
  • Tracking por centroides → conteo sin duplicados
  • ROI cuadrado central
  • Máscaras semitransparentes
  • Snapshot cada 10 detecciones nuevas (máx 20/sesión, elimina el más viejo)
  • OLED SSD1306 128×64 I2C
  • Historial acumulativo de sesiones → historial.json
"""

import cv2
import numpy as np
import os
import time
import logging
import threading
import json
from datetime import datetime
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from session_history import HistoryManager, SessionRecord

# ─── Edge Impulse Linux SDK ───────────────────────────────────────────────────
try:
    from edge_impulse_linux.image import ImageImpulseRunner
except ImportError:
    raise ImportError(
        "SDK no encontrado.\n"
        "  pip install edge_impulse_linux\n"
        "  (sin internet: instala desde wheel descargado previamente)"
    )

# ─── OLED SSD1306 (128×64, I2C) ──────────────────────────────────────────────
try:
    from PIL import Image as PILImage, ImageDraw, ImageFont
    import board
    import busio
    import adafruit_ssd1306
    OLED_AVAILABLE = True
except ImportError:
    OLED_AVAILABLE = False

# ─── Logging ──────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/detector.log"),
    ],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    # Modelo
    model_path: str            = "models/ecobicift-linux-aarch64-qnn-v2-impulse #4.eim"
    confidence_threshold: float = 0.55

    # Cámara
    camera_index: int          = 0
    camera_width: int          = 1920
    camera_height: int         = 1080
    camera_fps: int            = 30

    # ROI — fracción del lado corto de la imagen
    roi_fraction: float        = 0.55

    # Tracking
    max_disappeared: int       = 30    # frames sin detectarse antes de eliminar ID
    max_distance: float        = 80.0  # px máx para seguir siendo el mismo ID

    # Snapshots
    snapshot_interval: int     = 10    # nuevas detecciones entre cada snapshot

    # Etiquetas → categoría interna
    label_map: Dict[str, str]  = field(default_factory=lambda: {
        "ecobici": "ecobici",
        "bicycle": "bicycle",
        "person":  "person",
    })

    # OLED SSD1306 128×64
    oled_i2c_address: int      = 0x3C
    oled_update_interval: float = 1.5  # segundos entre actualizaciones

    # Visualización
    show_masks: bool           = True
    show_ids: bool             = True
    show_roi: bool             = True
    window_name: str           = "Reforma Counter"


CFG = Config()

COLORS = {
    "ecobici": (0,   200,  50),
    "bicycle": (255, 140,   0),
    "person":  (50,  150, 255),
}


# ══════════════════════════════════════════════════════════════════════════════
# CENTROID TRACKER
# ══════════════════════════════════════════════════════════════════════════════
class CentroidTracker:
    def __init__(self, max_disappeared: int = 30, max_distance: float = 80.0):
        self.next_id        = 0
        self.objects: OrderedDict[int, np.ndarray] = OrderedDict()
        self.labels:  Dict[int, str]               = {}
        self.disappeared: Dict[int, int]           = {}
        self.max_disappeared = max_disappeared
        self.max_distance    = max_distance

    def register(self, centroid: np.ndarray, label: str) -> int:
        oid = self.next_id
        self.objects[oid]     = centroid
        self.labels[oid]      = label
        self.disappeared[oid] = 0
        self.next_id         += 1
        return oid

    def deregister(self, oid: int):
        del self.objects[oid]
        del self.labels[oid]
        del self.disappeared[oid]

    def update(self, detections: List[Tuple[np.ndarray, str]]) -> Dict[int, Tuple[np.ndarray, str]]:
        if not detections:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return {oid: (c, self.labels[oid]) for oid, c in self.objects.items()}

        new_centroids = np.array([d[0] for d in detections])
        new_labels    = [d[1] for d in detections]

        if not self.objects:
            for c, l in zip(new_centroids, new_labels):
                self.register(c, l)
        else:
            obj_ids  = list(self.objects.keys())
            obj_cent = np.array(list(self.objects.values()))

            D    = np.linalg.norm(obj_cent[:, None] - new_centroids[None, :], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()

            for r, c in zip(rows, cols):
                if r in used_rows or c in used_cols:
                    continue
                if D[r, c] > self.max_distance:
                    continue
                oid = obj_ids[r]
                self.objects[oid]     = new_centroids[c]
                self.labels[oid]      = new_labels[c]
                self.disappeared[oid] = 0
                used_rows.add(r)
                used_cols.add(c)

            for r in set(range(len(obj_ids))) - used_rows:
                oid = obj_ids[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

            for c in set(range(len(new_centroids))) - used_cols:
                self.register(new_centroids[c], new_labels[c])

        return {oid: (c, self.labels[oid]) for oid, c in self.objects.items()}


# ══════════════════════════════════════════════════════════════════════════════
# COUNTERS
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Counters:
    persons:  int = 0
    bicycles: int = 0
    ecobicis: int = 0
    seen_ids: set = field(default_factory=set)

    def try_count(self, obj_id: int, label: str) -> bool:
        if obj_id in self.seen_ids:
            return False
        self.seen_ids.add(obj_id)
        if label == "person":
            self.persons += 1
        elif label in ("bicycle", "ecobici"):
            self.bicycles += 1
            if label == "ecobici":
                self.ecobicis += 1
        return True

    def as_dict(self):
        return {
            "personas":   self.persons,
            "bicicletas": self.bicycles,
            "ecobicis":   self.ecobicis,
        }


# ══════════════════════════════════════════════════════════════════════════════
# OLED SSD1306 128×64
# ══════════════════════════════════════════════════════════════════════════════
class OLEDDisplay:
    W, H = 128, 64

    def __init__(self, cfg: Config, session_id: int):
        self.enabled    = False
        self.session_id = session_id
        if not OLED_AVAILABLE:
            log.warning("Librerías OLED no disponibles — pantalla desactivada.")
            return
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            self.disp = adafruit_ssd1306.SSD1306_I2C(
                self.W, self.H, i2c, addr=cfg.oled_i2c_address
            )
            self.disp.fill(0)
            self.disp.show()
            self.enabled = True
            log.info("OLED SSD1306 128×64 OK en I2C 0x%02X", cfg.oled_i2c_address)
        except Exception as e:
            log.warning("OLED no disponible: %s", e)

    def _font(self, size: int):
        for p in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]:
            if Path(p).exists():
                try:
                    return ImageFont.truetype(p, size)
                except Exception:
                    pass
        return ImageFont.load_default()

    def update(self, counters: Counters, fps: float, elapsed_sec: float):
        if not self.enabled:
            return
        try:
            img  = PILImage.new("1", (self.W, self.H))
            draw = ImageDraw.Draw(img)
            fb   = self._font(11)
            fs   = self._font(10)

            draw.text((0,  0), f"REFORMA  S#{self.session_id:03d}", font=fb, fill=255)
            draw.line([(0, 12), (self.W, 12)], fill=255)
            draw.text((0, 14), f"Personas : {counters.persons}",  font=fs, fill=255)
            draw.text((0, 25), f"Bicis    : {counters.bicycles}", font=fs, fill=255)
            draw.text((0, 36), f"Ecobici  : {counters.ecobicis}", font=fs, fill=255)
            m, s = divmod(int(elapsed_sec), 60)
            draw.text((0, 52), f"{fps:.1f}fps  {m:02d}:{s:02d}",  font=fs, fill=255)

            self.disp.image(img)
            self.disp.show()
        except Exception as e:
            log.debug("OLED update error: %s", e)

    def clear(self):
        if self.enabled:
            try:
                self.disp.fill(0)
                self.disp.show()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# ROI
# ══════════════════════════════════════════════════════════════════════════════
def compute_roi(w: int, h: int, frac: float) -> Tuple[int, int, int, int]:
    side = int(min(w, h) * frac)
    cx, cy = w // 2, h // 2
    x1, y1 = cx - side // 2, cy - side // 2
    return x1, y1, x1 + side, y1 + side


def in_roi(cx: float, cy: float, roi: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = roi
    return x1 <= cx <= x2 and y1 <= cy <= y2


# ══════════════════════════════════════════════════════════════════════════════
# SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════════
def save_snapshot(frame: np.ndarray,
                  roi: Tuple[int, int, int, int],
                  bboxes: list,
                  counters: Counters,
                  session: SessionRecord):
    snap_path = session.register_snapshot()
    annotated = frame.copy()

    rx1, ry1, rx2, ry2 = roi
    cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)

    for (bx1, by1, bx2, by2, label, conf, oid) in bboxes:
        color = COLORS.get(label, (200, 200, 200))
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(annotated, f"#{oid} {label} {conf:.2f}",
                    (bx1, by1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

    for i, (txt, col) in enumerate([
        (f"Personas : {counters.persons}",  COLORS["person"]),
        (f"Bicis    : {counters.bicycles}", COLORS["bicycle"]),
        (f"Ecobici  : {counters.ecobicis}", COLORS["ecobici"]),
        (f"Sesion   : #{session.session_id:03d}", (220, 220, 220)),
    ]):
        y = 26 + i * 22
        cv2.putText(annotated, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
        cv2.putText(annotated, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col,     1)

    cv2.imwrite(str(snap_path), annotated)

    meta = {
        "sesion":      session.session_id,
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "contadores":  counters.as_dict(),
        "detecciones": len(bboxes),
    }
    with open(snap_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Snapshot → %s", snap_path)


# ══════════════════════════════════════════════════════════════════════════════
# MASK OVERLAY
# ══════════════════════════════════════════════════════════════════════════════
def draw_mask(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
              color: Tuple[int, int, int], alpha: float = 0.28):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run(cfg: Config = CFG):
    # ── Historial ─────────────────────────────────────────────────────────────
    history = HistoryManager()
    session = history.new_session(cfg.model_path)
    history.save_session(session)   # marca la sesión como "en curso"

    # ── Modelo ────────────────────────────────────────────────────────────────
    log.info("Cargando modelo: %s", cfg.model_path)
    runner = ImageImpulseRunner(cfg.model_path)
    try:
        model_info = runner.init()
        inp_w = model_info["model_parameters"]["image_input_width"]
        inp_h = model_info["model_parameters"]["image_input_height"]
        log.info("Modelo listo | input %dx%d | labels: %s",
                 inp_w, inp_h, model_info["model_parameters"].get("labels"))
    except Exception as e:
        log.error("Fallo al inicializar modelo: %s", e)
        runner.stop()
        return

    # ── Cámara ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(cfg.camera_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera_height)
    cap.set(cv2.CAP_PROP_FPS,          cfg.camera_fps)

    if not cap.isOpened():
        log.error("No se pudo abrir cámara %d", cfg.camera_index)
        runner.stop()
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("Cámara: %dx%d", actual_w, actual_h)

    roi      = compute_roi(actual_w, actual_h, cfg.roi_fraction)
    tracker  = CentroidTracker(cfg.max_disappeared, cfg.max_distance)
    counters = Counters()
    oled     = OLEDDisplay(cfg, session.session_id)

    t_start      = time.time()
    frame_count  = 0
    fps          = 0.0
    snap_trigger = 0

    # ── OLED thread ───────────────────────────────────────────────────────────
    _oled_stop = threading.Event()

    def _oled_loop():
        while not _oled_stop.is_set():
            oled.update(counters, fps, time.time() - t_start)
            _oled_stop.wait(cfg.oled_update_interval)

    threading.Thread(target=_oled_loop, daemon=True).start()

    log.info("Sesión #%03d activa. Q = salir  |  R = reset contadores",
             session.session_id)

    # ══════════════════════════════════════════════════════════════════════════
    # LOOP
    # ══════════════════════════════════════════════════════════════════════════
    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("Frame fallido, reintentando...")
            time.sleep(0.04)
            continue

        frame_count += 1
        fps = frame_count / (time.time() - t_start + 1e-9)

        # ── Inferencia ────────────────────────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            features, _ = runner.get_features_from_image(rgb)
            result      = runner.classify(features)
        except Exception as e:
            log.debug("Inferencia: %s", e)
            continue

        bb_list          = result.get("result", {}).get("bounding_boxes") or []
        detections_input = []
        bboxes_raw       = []

        for bb in bb_list:
            label_raw = bb.get("label", "").lower().strip()
            conf      = bb.get("value", 0.0)
            if conf < cfg.confidence_threshold:
                continue
            label = cfg.label_map.get(label_raw)
            if label is None:
                continue

            bx  = int(bb["x"]      / inp_w * actual_w)
            by  = int(bb["y"]      / inp_h * actual_h)
            bw  = int(bb["width"]  / inp_w * actual_w)
            bh  = int(bb["height"] / inp_h * actual_h)
            bx1 = max(0, bx);          by1 = max(0, by)
            bx2 = min(actual_w-1, bx+bw); by2 = min(actual_h-1, by+bh)
            cx, cy = (bx1+bx2)/2, (by1+by2)/2

            if not in_roi(cx, cy, roi):
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (80, 80, 80), 1)
                continue

            detections_input.append((np.array([cx, cy]), label))
            bboxes_raw.append((bx1, by1, bx2, by2, label, conf))

        # ── Tracker ───────────────────────────────────────────────────────────
        tracked = tracker.update(detections_input)

        final_bboxes = []
        for (bx1, by1, bx2, by2, label, conf) in bboxes_raw:
            bcx, bcy = (bx1+bx2)/2, (by1+by2)/2
            best_id, best_dist = -1, float("inf")
            for oid, (cent, olabel) in tracked.items():
                if olabel != label:
                    continue
                d = float(np.linalg.norm(cent - np.array([bcx, bcy])))
                if d < best_dist:
                    best_dist, best_id = d, oid
            final_bboxes.append((bx1, by1, bx2, by2, label, conf, best_id))

        # ── Contar ────────────────────────────────────────────────────────────
        newly_detected = 0
        for (_, _, _, _, label, _, oid) in final_bboxes:
            if oid != -1 and counters.try_count(oid, label):
                newly_detected += 1

        # ── Dibujar ROI ───────────────────────────────────────────────────────
        if cfg.show_roi:
            rx1, ry1, rx2, ry2 = roi
            roi_ovl = frame.copy()
            cv2.rectangle(roi_ovl, (rx1, ry1), (rx2, ry2), (255, 255, 0), -1)
            cv2.addWeighted(roi_ovl, 0.07, frame, 0.93, 0, frame)
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)

        # ── Dibujar bboxes ────────────────────────────────────────────────────
        for (bx1, by1, bx2, by2, label, conf, oid) in final_bboxes:
            color = COLORS.get(label, (200, 200, 200))
            if cfg.show_masks:
                draw_mask(frame, bx1, by1, bx2, by2, color)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
            if cfg.show_ids and oid != -1:
                tag = f"#{oid} {label}"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
                cv2.rectangle(frame, (bx1, by1-th-6), (bx1+tw+4, by1), color, -1)
                cv2.putText(frame, tag, (bx1+2, by1-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,0,0), 1)
            cv2.putText(frame, f"{conf:.2f}", (bx2-44, by2-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        # ── HUD ───────────────────────────────────────────────────────────────
        elapsed = time.time() - t_start
        m, s    = divmod(int(elapsed), 60)
        hud = [
            (f"Personas : {counters.persons}",  COLORS["person"]),
            (f"Bicis    : {counters.bicycles}", COLORS["bicycle"]),
            (f"Ecobici  : {counters.ecobicis}", COLORS["ecobici"]),
            (f"FPS:{fps:.1f}  {m:02d}:{s:02d}  S#{session.session_id:03d}",
             (220, 220, 220)),
        ]
        for i, (txt, col) in enumerate(hud):
            y = 26 + i * 24
            cv2.putText(frame, txt, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,0,0), 4)
            cv2.putText(frame, txt, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col,     2)

        # ── Snapshot ──────────────────────────────────────────────────────────
        snap_trigger += newly_detected
        if snap_trigger >= cfg.snapshot_interval:
            snap_trigger -= cfg.snapshot_interval
            threading.Thread(
                target=save_snapshot,
                args=(frame.copy(), roi, list(final_bboxes), counters, session),
                daemon=True,
            ).start()

        cv2.imshow(cfg.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            log.info("Saliendo...")
            break
        elif key == ord("r"):
            log.info("Reset de contadores e IDs.")
            counters = Counters()
            tracker  = CentroidTracker(cfg.max_disappeared, cfg.max_distance)

    # ══════════════════════════════════════════════════════════════════════════
    # CIERRE
    # ══════════════════════════════════════════════════════════════════════════
    _oled_stop.set()
    oled.clear()

    session.close(
        persons=counters.persons, bicycles=counters.bicycles,
        ecobicis=counters.ecobicis, total_frames=frame_count, avg_fps=fps,
    )
    history.save_session(session)

    cap.release()
    cv2.destroyAllWindows()
    runner.stop()

    log.info("Sesión #%03d cerrada y guardada. Contadores: %s",
             session.session_id, counters.as_dict())


if __name__ == "__main__":
    run()