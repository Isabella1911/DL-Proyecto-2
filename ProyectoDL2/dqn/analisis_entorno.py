"""Analisis del entorno antes de entrenar (seccion 2.1 del informe).

    python -m dqn.analisis_entorno

Genera:
    runs/analisis_entorno.json      -> espacios, acciones, recompensas, lineas base
    informe/figuras/preprocesamiento.png
    informe/figuras/recompensas.png
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import ale_py
import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dqn.config import ConfigEntorno
from dqn.entorno import crear_entorno
from dqn.metricas import resumen

gym.register_envs(ale_py)
RAIZ = Path(__file__).resolve().parent.parent
FIGURAS = RAIZ / "informe" / "figuras"


def entorno_crudo() -> dict:
    env = gym.make("ALE/SpaceInvaders-v5")
    obs, info = env.reset(seed=0)
    ale = env.unwrapped.ale
    datos = {
        "observacion_cruda": {"forma": list(obs.shape), "dtype": str(obs.dtype)},
        "acciones_minimas": env.unwrapped.get_action_meanings(),
        "n_acciones_full_action_space": 18,
        "vidas_iniciales": int(info["lives"]),
        "kwargs_v5": {k: v for k, v in env.spec.kwargs.items() if k != "game"},
        "max_frames_por_episodio": int(ale.getInt("max_num_frames_per_episode")),
    }
    env.close()
    return datos


def jugar(politica, n: int, cfg: ConfigEntorno, seed: int) -> dict:
    """Juega n partidas registrando todo lo que el agente veria."""
    env = crear_entorno(cfg)
    puntajes, largos, valores, pasos_con_r, pasos_tot = [], [], Counter(), 0, 0
    terminados = truncados = 0
    pasos_entre_recompensas, recompensa_por_vida = [], []
    for i in range(n):
        env.reset(seed=seed + i)
        total, pasos, desde_ultima, vidas, en_vida = 0.0, 0, 0, 3, 0.0
        while True:
            _, r, term, trunc, info = env.step(politica(env))
            total += r; pasos += 1; desde_ultima += 1; en_vida += r
            if r != 0:
                valores[int(r)] += 1
                pasos_con_r += 1
                pasos_entre_recompensas.append(desde_ultima)
                desde_ultima = 0
            if info["lives"] < vidas or term or trunc:
                recompensa_por_vida.append(en_vida)
                vidas, en_vida = info["lives"], 0.0
            if term or trunc:
                terminados += term; truncados += trunc
                break
        puntajes.append(total); largos.append(pasos); pasos_tot += pasos
    env.close()
    return {
        "resumen": resumen(puntajes),
        "largo_medio": float(np.mean(largos)),
        "puntajes": puntajes,
        "valores_recompensa": dict(sorted(valores.items())),
        "fraccion_pasos_con_recompensa": pasos_con_r / pasos_tot,
        "pasos_medios_entre_recompensas": float(np.mean(pasos_entre_recompensas)),
        "recompensa_media_por_vida": float(np.mean(recompensa_por_vida)),
        "terminated": int(terminados),
        "truncated": int(truncados),
    }


def figura_preprocesamiento(cfg: ConfigEntorno) -> None:
    crudo = gym.make("ALE/SpaceInvaders-v5")
    crudo.reset(seed=1)
    for _ in range(60):
        rgb, *_ = crudo.step(1)
    crudo.close()
    env = crear_entorno(cfg)
    obs, _ = env.reset(seed=1)
    for _ in range(60):
        obs, *_ = env.step(1)
    env.close()

    fig, ax = plt.subplots(1, 5, figsize=(13, 3.2))
    ax[0].imshow(rgb)
    ax[0].set_title(f"Original {rgb.shape[0]}x{rgb.shape[1]}x3")
    for k in range(4):
        ax[k + 1].imshow(obs[k], cmap="gray")
        ax[k + 1].set_title(f"Stack t-{3 - k}  (84x84)")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURAS / "preprocesamiento.png", dpi=150)
    plt.close(fig)


def figura_recompensas(res: dict) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    val = res["aleatorio"]["valores_recompensa"]
    ax[0].bar([str(k) for k in val], list(val.values()), color="#4c72b0")
    ax[0].set_xlabel("Recompensa por paso")
    ax[0].set_ylabel("Frecuencia")
    ax[0].set_title("Valores de recompensa (agente aleatorio)")
    datos = [res[n]["puntajes"] for n in ("aleatorio", "siempre_fire")]
    ax[1].boxplot(datos, tick_labels=["Aleatorio", "Siempre FIRE"])
    ax[1].set_ylabel("Puntaje por partida")
    ax[1].set_title("Lineas base (3 vidas)")
    fig.tight_layout()
    fig.savefig(FIGURAS / "recompensas.png", dpi=150)
    plt.close(fig)


def main(n: int = 30) -> None:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    cfg = ConfigEntorno()
    rng = np.random.default_rng(0)
    res = {"entorno": entorno_crudo(), "config_preprocesamiento": cfg.__dict__}
    res["aleatorio"] = jugar(lambda env: int(rng.integers(env.action_space.n)), n, cfg, seed=0)
    res["siempre_fire"] = jugar(lambda env: 1, n, cfg, seed=0)
    figura_preprocesamiento(cfg)
    figura_recompensas(res)
    (RAIZ / "runs").mkdir(exist_ok=True)
    (RAIZ / "runs" / "analisis_entorno.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    e = res["entorno"]
    print(f"Observacion cruda: {e['observacion_cruda']} | vidas: {e['vidas_iniciales']} | max frames: {e['max_frames_por_episodio']}")
    print(f"Acciones ({len(e['acciones_minimas'])}): {e['acciones_minimas']}")
    print(f"kwargs v5: {e['kwargs_v5']}")
    for nombre in ("aleatorio", "siempre_fire"):
        r = res[nombre]
        s = r["resumen"]
        print(f"\n[{nombre}] {n} partidas: media {s['media']:.1f} +- {s['std']:.1f}, max {s['max']:.0f}, "
              f"E[mejor de 5] {s['E_mejor_de_5']:.0f}, largo medio {r['largo_medio']:.0f} pasos")
        print(f"  valores de recompensa: {r['valores_recompensa']}")
        print(f"  pasos con recompensa: {100 * r['fraccion_pasos_con_recompensa']:.1f}% | "
              f"cada {r['pasos_medios_entre_recompensas']:.1f} pasos | por vida {r['recompensa_media_por_vida']:.0f} | "
              f"terminated={r['terminated']} truncated={r['truncated']}")


if __name__ == "__main__":
    main()
