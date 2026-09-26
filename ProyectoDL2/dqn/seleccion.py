"""Seleccion del agente final para la competencia (mejor de 5 partidas).

La evaluacion periodica usa 16 partidas: suficiente para seguir la curva, pero
una sola partida excepcional puede inflar E[mejor de 5]. Aqui se re-evaluan los
mejores checkpoints con muchas mas partidas (y semillas nuevas, distintas de las
usadas para elegirlos) y se copia el ganador a agente_final/.

    python -m dqn.seleccion --run ddqn-dueling-10M --top 5 --episodios 64
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dqn.config import ConfigDQN, ConfigEntorno
from dqn.metricas import resumen
from dqn.red import RedQ
from dqn.train import RAIZ_RUNS, evaluar

RAIZ = RAIZ_RUNS.parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--top", type=int, default=5, help="checkpoints a re-evaluar (segun la evaluacion periodica)")
    p.add_argument("--episodios", type=int, default=64)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=777_000, help="distinta de las semillas de entrenamiento/evaluacion")
    p.add_argument("--destino", default="agente_final")
    args = p.parse_args()

    carpeta = RAIZ_RUNS / args.run
    env_cfg = ConfigEntorno.cargar(carpeta / "config_entorno.json")
    cfg = ConfigDQN.cargar(carpeta / "hiperparametros.json")
    evals = pd.read_csv(carpeta / "evaluaciones.csv").drop_duplicates("paso", keep="last")
    candidatos = evals.nlargest(args.top, "E_mejor_de_5")["paso"].tolist()
    ultimo = int(evals["paso"].max())
    if ultimo not in candidatos:
        candidatos.append(ultimo)  # el final siempre compite

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_acciones = 18 if env_cfg.full_action_space else 6
    red = RedQ(n_acciones, env_cfg.frames_apilados, cfg.dueling).to(device)
    filas = []
    for paso in sorted(candidatos):
        ruta = carpeta / "checkpoints" / f"paso_{paso:08d}.pt"
        red.load_state_dict(torch.load(ruta, map_location=device, weights_only=True))
        puntajes = evaluar(red, env_cfg, args.episodios, args.n_envs, device, seed=args.seed)
        r = resumen(puntajes)
        filas.append({"paso": paso, "pesos": str(ruta.relative_to(RAIZ)), **{k: round(v, 1) for k, v in r.items()}})
        print(f"paso {paso:>10,}: media {r['media']:6.0f} +- {r['std']:4.0f} | max {r['max']:5.0f} | "
              f"E[mejor de 5] {r['E_mejor_de_5']:6.0f} | p20 {r['p20_mejor_de_5']:5.0f}", flush=True)

    with open(carpeta / "seleccion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    ganador = max(filas, key=lambda f: f["E_mejor_de_5"])
    destino = RAIZ / args.destino
    destino.mkdir(exist_ok=True)
    shutil.copy(RAIZ / ganador["pesos"], destino / "modelo.pt")
    shutil.copy(carpeta / "config_entorno.json", destino / "config_entorno.json")
    shutil.copy(carpeta / "hiperparametros.json", destino / "hiperparametros.json")
    (destino / "origen.txt").write_text(
        f"run: {args.run}\npaso: {ganador['paso']}\n"
        f"seleccion: {args.episodios} partidas greedy, semilla {args.seed}\n"
        f"media {ganador['media']} +- {ganador['std']}, max {ganador['max']}, "
        f"E[mejor de 5] {ganador['E_mejor_de_5']}, p20 {ganador['p20_mejor_de_5']}\n",
        encoding="utf-8",
    )
    print(f"\nGanador: paso {ganador['paso']:,} (E[mejor de 5] = {ganador['E_mejor_de_5']:.0f}) -> {destino}")


if __name__ == "__main__":
    main()
