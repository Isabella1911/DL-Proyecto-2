"""Replay buffer que guarda cada frame una sola vez.

Guardar las observaciones apiladas (4, 84, 84) de s y s' ocuparia 8 frames por
transicion (~56 KB): 500k transiciones = 28 GB. Pero stacks consecutivos
comparten 3 de sus 4 frames, asi que basta guardar el frame mas reciente de
cada paso (~7 KB) y reconstruir el stack al muestrear.

Hay un sub-buffer circular por entorno paralelo, para que los frames de un
mismo entorno queden contiguos. La reconstruccion replica exactamente a
`FrameStackObservation`: al inicio de un episodio el stack se rellena
repitiendo el primer frame. (Perder una vida NO reinicia el stack: el juego
sigue, solo el objetivo de Bellman se corta.)
"""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacidad: int, n_envs: int, forma_frame=(84, 84), frames_apilados: int = 4, seed: int = 0):
        self.n_envs = n_envs
        self.cap = capacidad // n_envs  # capacidad por entorno
        self.k = frames_apilados
        self.frames = np.zeros((n_envs, self.cap, *forma_frame), dtype=np.uint8)
        self.acciones = np.zeros((n_envs, self.cap), dtype=np.int64)
        self.recompensas = np.zeros((n_envs, self.cap), dtype=np.float32)
        #: done[t]: el objetivo de Bellman no se propaga desde s_{t+1}
        #: (fin de partida o vida perdida).
        self.dones = np.zeros((n_envs, self.cap), dtype=np.bool_)
        #: primero[t]: el frame t es el primero de una partida (tras reset).
        self.primero = np.zeros((n_envs, self.cap), dtype=np.bool_)
        self.pos = 0
        self.tamano = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.tamano * self.n_envs

    def agregar(self, obs: np.ndarray, primero: np.ndarray, acciones, recompensas, dones) -> None:
        """Guarda un paso de todos los entornos.

        `obs` es la observacion apilada (n_envs, 4, 84, 84) de s_t, sobre la
        que se eligio `acciones`; solo se guarda su ultimo frame.
        """
        p = self.pos
        self.frames[:, p] = obs[:, -1]
        self.primero[:, p] = primero
        self.acciones[:, p] = acciones
        self.recompensas[:, p] = recompensas
        self.dones[:, p] = dones
        self.pos = (p + 1) % self.cap
        self.tamano = min(self.tamano + 1, self.cap)

    def _apilar(self, e: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Stacks (B, 4, 84, 84) terminados en el indice t de cada entorno e."""
        idx = np.empty((len(t), self.k), dtype=np.int64)
        idx[:, -1] = t
        for j in range(self.k - 2, -1, -1):
            siguiente = idx[:, j + 1]
            # Si el frame siguiente abre la partida, se repite (relleno "reset").
            idx[:, j] = np.where(self.primero[e, siguiente], siguiente, (siguiente - 1) % self.cap)
        return self.frames[e[:, None], idx]

    def muestrear(self, batch_size: int) -> dict[str, np.ndarray]:
        # d = antiguedad del indice t: d >= 1 garantiza que s_{t+1} ya existe y
        # d <= tamano - k que los k-1 frames anteriores no fueron sobrescritos.
        if self.tamano < self.k + 2:
            raise ValueError("buffer demasiado vacio para muestrear")
        e = self.rng.integers(0, self.n_envs, size=batch_size)
        d = self.rng.integers(1, self.tamano - self.k + 1, size=batch_size)
        t = (self.pos - 1 - d) % self.cap
        t1 = (t + 1) % self.cap
        return {
            "obs": self._apilar(e, t),
            "acciones": self.acciones[e, t],
            "recompensas": self.recompensas[e, t],
            "dones": self.dones[e, t],
            "obs_sig": self._apilar(e, t1),
        }
