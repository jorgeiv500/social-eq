"""Ejecuta el pipeline oficial con protección frente a cuelgues de NEOS.

Motivo: las llamadas RPC del gestor NEOS de Pyomo no llevan tiempo de espera, así que una
llamada que no responde bloquea la corrida de forma indefinida (ocurrió el 2026-09-02: 90
minutos en una sola especificación de robustez, con el proceso sin consumir CPU). Este
envoltorio no cambia el pipeline: fija un tiempo de espera de socket y reintenta la llamada
al solver, dejando intacto el camino de código de `run_pipeline`.

Uso: python scripts/run_pipeline_resiliente.py [--timeout 420] [--intentos 4]
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import social_eq.followup as followup  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=420, help="segundos de espera por llamada RPC a NEOS")
    ap.add_argument("--intentos", type=int, default=4, help="reintentos por modelo antes de abandonar")
    args = ap.parse_args()

    socket.setdefaulttimeout(args.timeout)
    original = followup.solve_model

    def solve_con_reintento(model, config, **kwargs):
        ultimo = None
        for intento in range(1, args.intentos + 1):
            try:
                return original(model, config, **kwargs)
            except Exception as exc:  # noqa: BLE001 - queremos reintentar ante cualquier fallo de red
                ultimo = exc
                print(f"  NEOS falló (intento {intento}/{args.intentos}): {type(exc).__name__}: {exc}", flush=True)
                if intento < args.intentos:
                    espera = 30 * intento
                    print(f"  reintentando en {espera}s...", flush=True)
                    time.sleep(espera)
        raise RuntimeError(f"NEOS no respondió tras {args.intentos} intentos") from ultimo

    followup.solve_model = solve_con_reintento
    print(f"Pipeline con tiempo de espera {args.timeout}s por llamada y {args.intentos} intentos por modelo.", flush=True)
    return followup.cli(["--root", str(ROOT)])


if __name__ == "__main__":
    raise SystemExit(main())
