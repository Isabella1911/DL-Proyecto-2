"""Evaluacion con el protocolo de la competencia: 5 partidas greedy, cuenta la mejor.

Uso:
    python -m dqn.evaluate --run agente_final --episodios 5 --video
    python -m dqn.evaluate --run runs/ddqn-dueling-10M --pesos checkpoints/paso_05000000.pt

`--run` es una carpeta con config_entorno.json, hiperparametros.json y los pesos
(por defecto modelo.pt, o mejor.pt si no hay modelo.pt). El preprocesamiento se
reconstruye desde config_entorno.json: es el mismo del entrenamiento.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from dqn.config import ConfigDQN, ConfigEntorno
from dqn.entorno import crear_entorno, generar_video_agente
from dqn.metricas import resumen
from dqn.red import RedQ

RAIZ = Path(__file__).resolve().parent.parent


def cargar_agente(carpeta: Path, pesos: str | None = None, device: str = "cpu"):
    """Devuelve (funcion_agente, config_entorno, ruta_pesos) listos para jugar."""
    env_cfg = ConfigEntorno.cargar(carpeta / "config_entorno.json")
    cfg = ConfigDQN.cargar(carpeta / "hiperparametros.json")
    if pesos is None:
        pesos = "modelo.pt" if (carpeta / "modelo.pt").exists() else "mejor.pt"
    ruta = carpeta / pesos

    env = crear_entorno(env_cfg)
    n_acciones = int(env.action_space.n)
    env.close()
    red = RedQ(n_acciones, env_cfg.frames_apilados, cfg.dueling).to(device)
    red.load_state_dict(torch.load(ruta, map_location=device, weights_only=True))
    red.eval()

    def politica_greedy(observation: np.ndarray, env) -> int:
        with torch.no_grad():
            q = red(torch.from_numpy(np.asarray(observation))[None].to(device))
        return int(q.argmax(dim=1).item())

    return politica_greedy, env_cfg, ruta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="agente_final")
    p.add_argument("--pesos", default=None, help="archivo dentro de --run (por defecto modelo.pt / mejor.pt)")
    p.add_argument("--episodios", type=int, default=5)
    p.add_argument("--video", action="store_true", help="graba cada partida en <run>/videos/")
    p.add_argument("--seed", type=int, default=None, help="por defecto sin semilla, como en vivo")
    args = p.parse_args()

    carpeta = Path(args.run)
    if not carpeta.is_absolute():
        carpeta = RAIZ / carpeta
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agente, env_cfg, ruta = cargar_agente(carpeta, args.pesos, device)

    print("=" * 62)
    print(f"  Pesos    : {ruta.relative_to(RAIZ) if ruta.is_relative_to(RAIZ) else ruta}")
    print(f"  Device   : {device}")
    print(f"  Entorno  : {env_cfg.env_id} | sticky={env_cfg.repeat_action_probability} | "
          f"skip={env_cfg.frame_skip} | {env_cfg.screen_size}x{env_cfg.screen_size} gris x{env_cfg.frames_apilados}")
    print(f"  Politica : greedy (argmax Q)")
    print(f"  Episodios: {args.episodios}")
    print("=" * 62)

    inicio = time.time()
    if args.video:
        videos, metricas = generar_video_agente(
            agente, carpeta / "videos", name_prefix=carpeta.name, n_episodios=args.episodios, config=env_cfg, seed=args.seed
        )
    else:
        from dqn.entorno import ejecutar_episodio

        env = crear_entorno(env_cfg)
        metricas = []
        for i in range(args.episodios):
            semilla = None if args.seed is None else args.seed + i
            metricas.append({"episodio": i, **ejecutar_episodio(env, agente, seed=semilla)})
            print(f"  episodio {i:>3}: {metricas[-1]['puntaje']:>6.0f}", flush=True)
        env.close()
        videos = []

    print("\nEpisodio |   Pasos |  Puntaje")
    print("---------+---------+---------")
    for m in metricas:
        print(f"{m['episodio']:>8} | {m['pasos']:>7} | {m['puntaje']:>8.0f}")
    puntajes = [m["puntaje"] for m in metricas]
    mejor = int(np.argmax(puntajes))
    print(f"\n  MEJOR (metrica de la competencia): {puntajes[mejor]:.0f}  (episodio {mejor})")
    print(f"  Promedio: {np.mean(puntajes):.1f} +- {np.std(puntajes):.1f}")
    if len(puntajes) > 5:
        r = resumen(puntajes)
        print(f"  E[mejor de 5] estimado con {len(puntajes)} partidas: {r['E_mejor_de_5']:.0f} "
              f"(p20: {r['p20_mejor_de_5']:.0f})")
    if videos:
        print(f"  Video de la mejor partida: {videos[mejor]}")
    print(f"  Tiempo: {time.time() - inicio:.0f} s")


if __name__ == "__main__":
    main()
