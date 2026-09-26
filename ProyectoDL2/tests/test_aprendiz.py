"""El update con CUDA Graph debe dar los mismos pesos que el camino normal.

    python -m tests.test_aprendiz
"""

import copy

import numpy as np
import torch

from dqn.aprendiz import Aprendiz
from dqn.config import ConfigDQN
from dqn.red import RedQ


def lote_aleatorio(rng, b=32):
    return {
        "obs": rng.integers(0, 256, (b, 4, 84, 84), dtype=np.uint8),
        "obs_sig": rng.integers(0, 256, (b, 4, 84, 84), dtype=np.uint8),
        "acciones": rng.integers(0, 6, b).astype(np.int64),
        "recompensas": rng.choice([-1.0, 0.0, 1.0], b).astype(np.float32),
        "dones": rng.random(b) < 0.1,
    }


def crear(cfg, dev, grafo, red0):
    red, obj = copy.deepcopy(red0).to(dev), copy.deepcopy(red0).to(dev)
    return Aprendiz(red, obj, cfg, dev, (4, 84, 84), usar_grafo=grafo)


def diferencia(a, b):
    return max((p - q).abs().max().item() for p, q in zip(a.red.parameters(), b.red.parameters()))


def main():
    dev = torch.device("cuda")
    # Determinista: si no, cuDNN elige algoritmos distintos en cada camino y
    # Adam amplifica ese ruido de redondeo (normaliza gradientes casi nulos).
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg = ConfigDQN()
    torch.manual_seed(0)
    red0 = RedQ(6)
    rng = np.random.default_rng(0)
    lotes = [lote_aleatorio(rng) for _ in range(60)]

    normal, grafo = crear(cfg, dev, False, red0), crear(cfg, dev, True, red0)
    for i, lote in enumerate(lotes[:40]):
        normal.actualizar(lote)
        grafo.actualizar(lote)
        if i == 20:
            normal.sincronizar_objetivo()
            grafo.sincronizar_objetivo()
    torch.cuda.synchronize()
    d = diferencia(normal, grafo)
    print(f"40 updates: dif. max de pesos = {d:.2e}; metricas normal {normal.leer_metricas()} grafo {grafo.leer_metricas()}")
    assert d == 0.0, d

    # Reanudacion: grafo nuevo que parte de pesos + Adam ya avanzados.
    reanudado = crear(cfg, dev, True, normal.red)
    reanudado.objetivo.load_state_dict(normal.objetivo.state_dict())
    # deepcopy: load_state_dict NO copia tensores que ya estan en la GPU; sin
    # esto ambos aprendices compartirian el estado de Adam (como tras torch.load).
    reanudado.optim.load_state_dict(copy.deepcopy(normal.optim.state_dict()))
    for lote in lotes[40:]:
        normal.actualizar(lote)
        reanudado.actualizar(lote)
    torch.cuda.synchronize()
    d = diferencia(normal, reanudado)
    print(f"reanudado +20 updates: dif. max de pesos = {d:.2e}")
    assert d == 0.0, d
    print("OK")


if __name__ == "__main__":
    main()
