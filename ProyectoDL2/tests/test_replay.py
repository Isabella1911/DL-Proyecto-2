"""El buffer debe reconstruir exactamente los stacks que produce el entorno.

    python -m tests.test_replay
"""

import numpy as np

from dqn.config import ConfigEntorno
from dqn.entorno import crear_entorno
from dqn.replay import ReplayBuffer


def main(pasos: int = 3000, capacidad: int = 1000) -> None:
    env = crear_entorno(ConfigEntorno())
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(capacidad, n_envs=1)
    obs, _ = env.reset(seed=0)
    primero, historial, n_partidas = True, [], 0
    for _ in range(pasos):
        a = int(rng.integers(env.action_space.n))
        sig, r, term, trunc, _ = env.step(a)
        buf.agregar(obs[None], np.array([primero]), [a], [np.sign(r)], [term or trunc])
        historial.append(obs.copy())
        primero = term or trunc
        if primero:
            n_partidas += 1
            sig, _ = env.reset()
        obs = sig
    env.close()

    # Cada indice valido del buffer contra el stack real que vio el agente.
    cap = buf.cap
    for d in range(1, buf.tamano - buf.k + 1):
        t = (buf.pos - 1 - d) % cap
        real = historial[len(historial) - 1 - d]
        reconstruido = buf._apilar(np.array([0]), np.array([t]))[0]
        assert np.array_equal(real, reconstruido), f"stack distinto en d={d}"
        if not buf.dones[0, t]:
            real_sig = historial[len(historial) - d]
            assert np.array_equal(real_sig, buf._apilar(np.array([0]), np.array([(t + 1) % cap]))[0]), f"s' distinto en d={d}"
    lote = buf.muestrear(32)
    assert lote["obs"].shape == (32, 4, 84, 84) and lote["obs"].dtype == np.uint8
    print(f"OK: {buf.tamano - buf.k} stacks verificados, {n_partidas} partidas completas, buffer con vuelta completa")


if __name__ == "__main__":
    main()
