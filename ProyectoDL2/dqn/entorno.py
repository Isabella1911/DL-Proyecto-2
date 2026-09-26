"""Construccion del entorno y funciones de ejecucion (estilo Laboratorio 5).

Cadena de wrappers, de adentro hacia afuera:

    ALE/SpaceInvaders-v5 (frameskip=1, sticky 0.25, 6 acciones)
      -> RecordVideo            (opcional; graba a color, 60 fps, todos los frames)
      -> AtariPreprocessing     (noops, skip 4 con max-pool, gris, 84x84)
      -> FrameStackObservation (4 frames -> observacion (4, 84, 84) uint8)

El entorno NO recorta recompensas ni corta el episodio al perder una vida: eso
lo hace el bucle de entrenamiento. Asi el mismo entorno sirve para entrenar y
para evaluar, y las recompensas que reporta son siempre el puntaje real.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import ale_py
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordVideo

from dqn.config import ConfigEntorno

gym.register_envs(ale_py)

#: Firma de un agente: recibe la observacion y el entorno, devuelve una accion.
FuncionAgente = Callable[[np.ndarray, gym.Env], int]


def crear_entorno(
    config: ConfigEntorno = ConfigEntorno(),
    video_folder: str | Path | None = None,
    name_prefix: str = "agente",
) -> gym.Env:
    """Crea el entorno preprocesado. Con `video_folder` graba cada episodio."""
    env = gym.make(
        config.env_id,
        frameskip=1,  # el salto de frames lo hace AtariPreprocessing
        repeat_action_probability=config.repeat_action_probability,
        full_action_space=config.full_action_space,
        render_mode="rgb_array" if video_folder is not None else None,
    )
    if video_folder is not None:
        env = RecordVideo(
            env,
            video_folder=str(video_folder),
            name_prefix=name_prefix,
            episode_trigger=lambda _: True,
            fps=60,
        )
    env = AtariPreprocessing(
        env,
        noop_max=config.noop_max,
        frame_skip=config.frame_skip,
        screen_size=config.screen_size,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )
    env = FrameStackObservation(env, stack_size=config.frames_apilados)
    return env


def crear_entornos_vectorizados(config: ConfigEntorno, n_envs: int) -> gym.vector.VectorEnv:
    """`n_envs` copias en procesos separados, para recolectar experiencia en paralelo.

    SAME_STEP: cuando un episodio termina, el mismo `step` devuelve ya la
    primera observacion del episodio siguiente (no hay paso "muerto").
    """
    return gym.vector.AsyncVectorEnv(
        [lambda: crear_entorno(config) for _ in range(n_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


def ejecutar_episodio(
    env: gym.Env,
    funcion_agente: FuncionAgente,
    max_steps: int = 50_000,
    seed: int | None = None,
) -> dict:
    """Juega un episodio completo (las 3 vidas). Devuelve pasos y puntaje."""
    obs, _ = env.reset(seed=seed)
    puntaje, pasos = 0.0, 0
    for pasos in range(1, max_steps + 1):
        obs, recompensa, terminated, truncated, _ = env.step(funcion_agente(obs, env))
        puntaje += float(recompensa)
        if terminated or truncated:
            break
    return {"pasos": pasos, "puntaje": puntaje, "truncado": bool(truncated)}


def generar_video_agente(
    funcion_agente: FuncionAgente,
    video_folder: str | Path,
    name_prefix: str,
    n_episodios: int = 1,
    config: ConfigEntorno = ConfigEntorno(),
    seed: int | None = None,
) -> tuple[list[Path], list[dict]]:
    """Juega `n_episodios` grabando cada uno. Devuelve rutas de video y metricas."""
    env = crear_entorno(config, video_folder=video_folder, name_prefix=name_prefix)
    metricas = []
    try:
        for i in range(n_episodios):
            semilla = None if seed is None else seed + i
            metricas.append({"episodio": i, **ejecutar_episodio(env, funcion_agente, seed=semilla)})
    finally:
        env.close()  # cierra y escribe el ultimo video
    # RecordVideo numera los episodios desde 0 en el orden en que se jugaron.
    videos = [Path(video_folder) / f"{name_prefix}-episode-{i}.mp4" for i in range(n_episodios)]
    return videos, metricas


def agente_aleatorio(observation: np.ndarray, env: gym.Env) -> int:
    """Linea base: acciones uniformes."""
    return int(env.action_space.sample())
