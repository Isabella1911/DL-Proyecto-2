"""Configuracion del entorno y del algoritmo.

Todo lo que define el preprocesamiento vive en `ConfigEntorno`. El entrenamiento
lo guarda como JSON junto a los pesos y la evaluacion lo lee de ahi, asi que es
imposible evaluar con un preprocesamiento distinto al del entrenamiento.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class ConfigEntorno:
    env_id: str = "ALE/SpaceInvaders-v5"
    #: Sticky actions: con probabilidad 0.25 el emulador repite la accion
    #: anterior. Es el valor por defecto de v5, el de la competencia.
    repeat_action_probability: float = 0.25
    #: Solo las 6 acciones minimas de Space Invaders (no las 18 del joystick).
    full_action_space: bool = False
    #: Cada accion del agente se repite 4 frames; se toma el max de los 2
    #: ultimos para eliminar el parpadeo de los sprites.
    frame_skip: int = 4
    screen_size: int = 84
    frames_apilados: int = 4
    #: Al reiniciar se ejecutan entre 0 y 30 NOOPs aleatorios: cada partida
    #: arranca en un estado distinto.
    noop_max: int = 30

    def guardar(self, ruta: str | Path) -> None:
        Path(ruta).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def cargar(cls, ruta: str | Path) -> "ConfigEntorno":
        datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
        return cls(**datos)


@dataclass(frozen=True)
class ConfigDQN:
    """Hiperparametros de Double + Dueling DQN.

    Los valores por defecto siguen la configuracion de Atari de Dopamine /
    Rainbow (Adam en vez del RMSProp del DQN de Nature), escalados a los
    pasos disponibles. Un "paso" es una decision del agente = 4 frames.
    """

    pasos_totales: int = 10_000_000
    n_envs: int = 8
    seed: int = 0

    # --- red ---
    dueling: bool = True
    double: bool = True

    # --- replay buffer ---
    #: Transiciones guardadas. Cada una ocupa ~7 KB (un solo frame de 84x84);
    #: 500k = 3.5 GB de RAM.
    capacidad_buffer: int = 500_000
    pasos_antes_de_aprender: int = 50_000

    # --- optimizacion ---
    lr: float = 1e-4
    adam_eps: float = 1.5e-4
    batch_size: int = 32
    gamma: float = 0.99
    #: Un paso de gradiente cada `pasos_por_update` transiciones (replay ratio
    #: de 32/4 = 8 muestras por transicion, como el DQN original).
    pasos_por_update: int = 4
    #: Pasos del agente entre copias de la red online a la red objetivo.
    periodo_red_objetivo: int = 8_000
    max_norma_gradiente: float = 10.0

    # --- exploracion epsilon-greedy ---
    epsilon_inicial: float = 1.0
    epsilon_final: float = 0.01
    #: Pasos en los que epsilon decae linealmente de inicial a final.
    pasos_decaimiento_epsilon: int = 1_000_000

    # --- evaluacion periodica ---
    periodo_evaluacion: int = 250_000
    episodios_evaluacion: int = 16

    def guardar(self, ruta: str | Path) -> None:
        Path(ruta).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def cargar(cls, ruta: str | Path) -> "ConfigDQN":
        datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
        validos = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in datos.items() if k in validos})
