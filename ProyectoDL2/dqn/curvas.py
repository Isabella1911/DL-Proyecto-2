"""Curvas de entrenamiento de una corrida (seccion 2.3 del informe).

    python -m dqn.curvas --run ddqn-dueling-10M

Genera informe/figuras/<run>.png con cuatro paneles:
    1. Puntaje por partida de entrenamiento (epsilon-greedy) + media movil.
    2. Evaluacion greedy periodica: media, maximo y E[mejor de 5].
    3. Perdida de Huber.
    4. Q medio de las acciones tomadas (divergencia = crecimiento sin freno).
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from dqn.train import RAIZ_RUNS

FIGURAS = RAIZ_RUNS.parent / "informe" / "figuras"
MILLONES = FuncFormatter(lambda x, _: f"{x / 1e6:g}M")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--ventana", type=int, default=100, help="partidas en la media movil")
    args = p.parse_args()

    carpeta = RAIZ_RUNS / args.run
    eps = pd.read_csv(carpeta / "episodios.csv")
    evals = pd.read_csv(carpeta / "evaluaciones.csv").drop_duplicates("paso", keep="last")
    prog = pd.read_csv(carpeta / "progreso.csv").drop_duplicates("paso", keep="last")
    prog = prog[prog["perdida"] > 0]  # antes de aprender no hay perdida

    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    a = ax[0, 0]
    a.scatter(eps["paso"], eps["puntaje"], s=2, alpha=0.15, color="#8da0cb", label="partida")
    a.plot(eps["paso"], eps["puntaje"].rolling(args.ventana, min_periods=10).mean(), color="#1b4f9c", lw=2,
           label=f"media movil ({args.ventana})")
    a.set_title("Entrenamiento (epsilon-greedy)")
    a.set_ylabel("Puntaje por partida")

    a = ax[0, 1]
    a.fill_between(evals["paso"], evals["media"] - evals["std"], evals["media"] + evals["std"], color="#fc8d62", alpha=0.2)
    a.plot(evals["paso"], evals["media"], "o-", color="#e1581c", ms=3, label="media ± std")
    a.plot(evals["paso"], evals["max"], "^", color="#66a61e", ms=4, label="maximo")
    a.plot(evals["paso"], evals["E_mejor_de_5"], "s-", color="#7570b3", ms=3, label="E[mejor de 5]")
    a.set_title(f"Evaluacion greedy ({int(evals['n'].iloc[-1])} partidas por punto)")
    a.set_ylabel("Puntaje")

    ax[1, 0].plot(prog["paso"], prog["perdida"], color="#444")
    ax[1, 0].set_title("Perdida de Huber (media cada 10k pasos)")
    ax[1, 1].plot(prog["paso"], prog["q_medio"], color="#444")
    ax[1, 1].set_title("Q medio de las acciones del lote")

    for a in ax.flat:
        a.xaxis.set_major_formatter(MILLONES)
        a.set_xlabel("Pasos del agente")
        a.grid(alpha=0.3)
    ax[0, 0].legend(loc="upper left")
    ax[0, 1].legend(loc="upper left")
    fig.suptitle(args.run)
    fig.tight_layout()
    FIGURAS.mkdir(parents=True, exist_ok=True)
    salida = FIGURAS / f"{args.run}.png"
    fig.savefig(salida, dpi=150)
    print(f"Figura: {salida}")


if __name__ == "__main__":
    main()
