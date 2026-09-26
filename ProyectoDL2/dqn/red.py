"""Red Q: torso convolucional de Nature DQN + cabeza Dueling opcional.

Dimensiones con la observacion (4, 84, 84):

    conv 32 8x8 s4 -> (32, 20, 20)
    conv 64 4x4 s2 -> (64, 9, 9)
    conv 64 3x3 s1 -> (64, 7, 7) = 3136 features

Dueling (Wang et al., 2016) separa la cabeza en dos corrientes:

    V(s)    : 3136 -> 512 -> 1
    A(s, a) : 3136 -> 512 -> n_acciones
    Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)

Restar la media hace identificable la descomposicion (si no, V y A podrian
desplazarse en una constante sin cambiar Q). La ventaja: V se aprende en cada
transicion aunque solo se haya tomado una accion, lo que ayuda en juegos como
Space Invaders donde muchas acciones tienen consecuencias casi iguales.
"""

from __future__ import annotations

import torch
from torch import nn


class RedQ(nn.Module):
    def __init__(self, n_acciones: int, canales: int = 4, dueling: bool = True):
        super().__init__()
        self.dueling = dueling
        self.torso = nn.Sequential(
            nn.Conv2d(canales, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        n_features = 64 * 7 * 7
        if dueling:
            self.valor = nn.Sequential(nn.Linear(n_features, 512), nn.ReLU(), nn.Linear(512, 1))
            self.ventaja = nn.Sequential(nn.Linear(n_features, 512), nn.ReLU(), nn.Linear(512, n_acciones))
        else:
            self.q = nn.Sequential(nn.Linear(n_features, 512), nn.ReLU(), nn.Linear(512, n_acciones))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """`obs`: uint8 (B, 4, 84, 84). Devuelve Q (B, n_acciones)."""
        x = self.torso(obs.float() / 255.0)
        if not self.dueling:
            return self.q(x)
        a = self.ventaja(x)
        return self.valor(x) + a - a.mean(dim=1, keepdim=True)
