#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# setup.sh — Configura el entorno en la Rubik Pi 3 y ejecuta el detector
# Uso:
#   chmod +x setup.sh
#   ./setup.sh setup     # primera vez
#   ./setup.sh run       # iniciar detector
#   ./setup.sh run-bg    # correr en background (nohup)
#   ./setup.sh status    # ver si está corriendo
#   ./setup.sh logs      # tail del log en vivo
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

VENV_DIR="venv"
PYTHON="python3"
PID_FILE=".detector.pid"
LOG_FILE="logs/detector.log"

banner() { echo -e "\n\033[1;36m▶ $1\033[0m"; }
ok()     { echo -e "\033[1;32m✔ $1\033[0m"; }
warn()   { echo -e "\033[1;33m⚠ $1\033[0m"; }
err()    { echo -e "\033[1;31m✘ $1\033[0m"; exit 1; }

# ─── setup ────────────────────────────────────────────────────────────────────
cmd_setup() {
    banner "Verificando dependencias del sistema"

    # I2C para OLED
    if ! lsmod | grep -q i2c_dev; then
        warn "Módulo i2c_dev no cargado. Intentando cargar..."
        sudo modprobe i2c-dev || warn "No se pudo cargar i2c-dev (¿habilitado en device tree?)"
    fi

    # Grupos de usuario
    for grp in video i2c dialout; do
        if ! groups | grep -qw "$grp"; then
            warn "Usuario no está en grupo '$grp'. Agregando..."
            sudo usermod -aG "$grp" "$USER" || true
        fi
    done

    banner "Creando virtualenv con acceso a paquetes del sistema"
    $PYTHON -m venv "$VENV_DIR" --system-site-packages
    # system-site-packages permite usar OpenCV instalado con apt (compilado para aarch64)

    banner "Instalando dependencias Python"
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -r requirements.txt

    banner "Verificando Edge Impulse Linux SDK"
    "$VENV_DIR/bin/python" -c "from edge_impulse_linux.image import ImageImpulseRunner; print('EI SDK OK')" \
        || err "SDK no encontrado. Revisa requirements.txt"

    banner "Verificando modelo .eim"
    MODEL=$(ls models/*.eim 2>/dev/null | head -1)
    if [[ -z "$MODEL" ]]; then
        warn "No hay archivo .eim en models/. Coloca el modelo antes de correr."
    else
        ok "Modelo encontrado: $MODEL"
    fi

    banner "Verificando cámara"
    if ls /dev/video* &>/dev/null; then
        ok "Dispositivos de video: $(ls /dev/video*)"
    else
        warn "No se encontraron /dev/video*"
    fi

    banner "Verificando I2C (OLED)"
    if command -v i2cdetect &>/dev/null; then
        i2cdetect -y 1 2>/dev/null || warn "i2cdetect falló (¿bus I2C correcto?)"
    else
        warn "i2cdetect no instalado. sudo apt install i2c-tools"
    fi

    mkdir -p logs snapshots models
    ok "Setup completo. Corre: ./setup.sh run"
}

# ─── run ──────────────────────────────────────────────────────────────────────
cmd_run() {
    [[ -d "$VENV_DIR" ]] || err "Virtualenv no encontrado. Corre './setup.sh setup' primero."
    mkdir -p logs
    banner "Iniciando detector..."
    "$VENV_DIR/bin/python" detector.py
}

cmd_run_bg() {
    [[ -d "$VENV_DIR" ]] || err "Virtualenv no encontrado. Corre './setup.sh setup' primero."
    mkdir -p logs
    banner "Iniciando en background..."
    nohup "$VENV_DIR/bin/python" detector.py >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    ok "PID: $(cat $PID_FILE). Logs: $LOG_FILE"
}

cmd_status() {
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
        ok "Detector corriendo con PID $(cat $PID_FILE)"
    else
        warn "Detector no está corriendo"
    fi
}

cmd_stop() {
    if [[ -f "$PID_FILE" ]]; then
        kill "$(cat $PID_FILE)" && rm "$PID_FILE" && ok "Detector detenido"
    else
        warn "No hay PID guardado"
    fi
}

cmd_logs() {
    tail -f "$LOG_FILE"
}

# ─── router ───────────────────────────────────────────────────────────────────
case "${1:-help}" in
    setup)   cmd_setup   ;;
    run)     cmd_run     ;;
    run-bg)  cmd_run_bg  ;;
    status)  cmd_status  ;;
    stop)    cmd_stop    ;;
    logs)    cmd_logs    ;;
    *)
        echo "Uso: $0 {setup|run|run-bg|status|stop|logs}"
        exit 1
    ;;
esac
