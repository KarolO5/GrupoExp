"""
session_history.py
Manejo del historial acumulativo de sesiones (historial.json)
y snapshots por sesión (máx 20 por carpeta).
"""

import json
import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

HISTORY_FILE = "historial.json"
MAX_SNAPS_PER_SESSION = 20


# ══════════════════════════════════════════════════════════════════════════════
# SESSION RECORD
# ══════════════════════════════════════════════════════════════════════════════
class SessionRecord:
    """
    Representa una sesión de ejecución del programa.
    Se crea al inicio y se cierra al salir.
    """

    def __init__(self, session_id: int, model_path: str):
        self.session_id   = session_id
        self.model_path   = Path(model_path).name
        self.start_dt     = datetime.now()
        self.end_dt: Optional[datetime] = None

        # Contadores finales (se llenan al cerrar)
        self.persons      = 0
        self.bicycles     = 0
        self.ecobicis     = 0
        self.total_frames = 0
        self.avg_fps      = 0.0
        self.snap_count   = 0

        # Carpeta de snapshots de esta sesión
        self.snap_dir = Path("snapshots") / f"sesion_{session_id:03d}"
        self.snap_dir.mkdir(parents=True, exist_ok=True)

        log.info("Sesión %03d iniciada: %s", session_id,
                 self.start_dt.strftime("%Y-%m-%d %H:%M:%S"))

    # ── Snapshot management ───────────────────────────────────────────────────
    def snap_path(self, index: int) -> Path:
        """Devuelve la ruta para el snapshot número `index` de esta sesión."""
        return self.snap_dir / f"snap_{index:04d}.jpg"

    def register_snapshot(self) -> Path:
        """
        Registra un nuevo snapshot.
        Si hay más de MAX_SNAPS_PER_SESSION, elimina el más antiguo.
        Retorna la ruta donde guardar la imagen.
        """
        existing = sorted(self.snap_dir.glob("snap_*.jpg"))

        # Eliminar excedentes (deja espacio para el nuevo)
        while len(existing) >= MAX_SNAPS_PER_SESSION:
            oldest = existing.pop(0)
            oldest.unlink(missing_ok=True)
            # Elimina también el JSON sidecar si existe
            oldest.with_suffix(".json").unlink(missing_ok=True)
            log.debug("Snapshot antiguo eliminado: %s", oldest.name)

        self.snap_count = len(existing) + 1
        return self.snap_dir / f"snap_{datetime.now().strftime('%H%M%S_%f')}.jpg"

    # ── Close ─────────────────────────────────────────────────────────────────
    def close(self, persons: int, bicycles: int, ecobicis: int,
               total_frames: int, avg_fps: float):
        self.end_dt       = datetime.now()
        self.persons      = persons
        self.bicycles     = bicycles
        self.ecobicis     = ecobicis
        self.total_frames = total_frames
        self.avg_fps      = round(avg_fps, 2)
        self.snap_count   = len(list(self.snap_dir.glob("snap_*.jpg")))

        duration = self.end_dt - self.start_dt
        log.info("Sesión %03d cerrada | duración %s | P:%d B:%d E:%d",
                 self.session_id,
                 _fmt_duration(duration),
                 persons, bicycles, ecobicis)

    # ── Serialise ─────────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        end_str = self.end_dt.strftime("%Y-%m-%d %H:%M:%S") if self.end_dt else "en curso"
        duration = (self.end_dt - self.start_dt) if self.end_dt else timedelta(0)
        return {
            "sesion":          self.session_id,
            "modelo":          self.model_path,
            "inicio":          self.start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "fin":             end_str,
            "duracion":        _fmt_duration(duration),
            "duracion_seg":    round(duration.total_seconds(), 1),
            "conteo": {
                "personas":    self.persons,
                "bicicletas":  self.bicycles,   # ecobici + generic
                "ecobicis":    self.ecobicis,
            },
            "frames_totales":  self.total_frames,
            "fps_promedio":    self.avg_fps,
            "snapshots":       self.snap_count,
            "carpeta_snaps":   str(self.snap_dir),
        }


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY MANAGER
# ══════════════════════════════════════════════════════════════════════════════
class HistoryManager:
    """
    Lee y escribe `historial.json`.
    Determina el próximo session_id automáticamente.
    """

    def __init__(self, path: str = HISTORY_FILE):
        self.path    = Path(path)
        self.records = self._load()

    def _load(self) -> list:
        if not self.path.exists():
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError) as e:
            log.warning("historial.json corrupto, iniciando nuevo: %s", e)
        return []

    def next_session_id(self) -> int:
        if not self.records:
            return 1
        return max(r.get("sesion", 0) for r in self.records) + 1

    def new_session(self, model_path: str) -> SessionRecord:
        sid = self.next_session_id()
        return SessionRecord(sid, model_path)

    def save_session(self, session: SessionRecord):
        """Agrega o actualiza la sesión en historial.json."""
        entry = session.to_dict()

        # Actualiza si ya existe (por si se llama más de una vez)
        for i, r in enumerate(self.records):
            if r.get("sesion") == session.session_id:
                self.records[i] = entry
                break
        else:
            self.records.append(entry)

        self._write()

    def _write(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.records, f, ensure_ascii=False, indent=2)
            log.info("Historial guardado → %s", self.path)
        except OSError as e:
            log.error("No se pudo escribir historial.json: %s", e)

    # ── Quick summary (for OLED / terminal) ───────────────────────────────────
    def lifetime_totals(self) -> dict:
        """Suma acumulada de todas las sesiones."""
        return {
            "sesiones":   len(self.records),
            "personas":   sum(r["conteo"]["personas"]   for r in self.records),
            "bicicletas": sum(r["conteo"]["bicicletas"] for r in self.records),
            "ecobicis":   sum(r["conteo"]["ecobicis"]   for r in self.records),
        }


# ══════════════════════════════════════════════════════════════════════════════
# QUERY HELPERS  (para acceder al historial desde terminal)
# ══════════════════════════════════════════════════════════════════════════════
def print_history(path: str = HISTORY_FILE):
    """Imprime un resumen legible de todas las sesiones."""
    hm = HistoryManager(path)
    if not hm.records:
        print("Sin sesiones registradas aún.")
        return

    sep = "─" * 62
    print(f"\n{'═'*62}")
    print(f"  HISTORIAL REFORMA COUNTER — {len(hm.records)} sesión(es)")
    print(f"{'═'*62}")

    for r in hm.records:
        c = r["conteo"]
        print(f"\n  Sesión #{r['sesion']:03d}  |  {r['inicio']}  →  {r['fin']}")
        print(f"  Duración   : {r['duracion']}")
        print(f"  Personas   : {c['personas']}")
        print(f"  Bicicletas : {c['bicicletas']}  (Ecobici: {c['ecobicis']})")
        print(f"  Snapshots  : {r['snapshots']}  en  {r['carpeta_snaps']}")
        print(f"  FPS prom.  : {r['fps_promedio']}")
        print(f"  {sep}")

    totals = hm.lifetime_totals()
    print(f"\n  TOTALES ACUMULADOS:")
    print(f"  Personas   : {totals['personas']}")
    print(f"  Bicicletas : {totals['bicicletas']}  (Ecobici: {totals['ecobicis']})")
    print(f"{'═'*62}\n")


def print_last_session(path: str = HISTORY_FILE):
    hm = HistoryManager(path)
    if not hm.records:
        print("Sin sesiones registradas.")
        return
    last = hm.records[-1]
    print(json.dumps(last, ensure_ascii=False, indent=2))


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _fmt_duration(td: timedelta) -> str:
    total = int(td.total_seconds())
    h, rem = divmod(total, 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ══════════════════════════════════════════════════════════════════════════════
# CLI directo: python session_history.py
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "last":
        print_last_session()
    else:
        print_history()
