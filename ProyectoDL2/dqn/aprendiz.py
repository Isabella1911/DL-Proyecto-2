"""El paso de aprendizaje de Double (+ Dueling) DQN, capturado en un CUDA Graph.

Por que un grafo: la red es chica y el batch es de 32, asi que en Windows el
update lo domina el costo de *lanzar* ~200 kernels uno por uno (medido: ~7 ms
por update, casi igual con batch 32 que con 64). Un CUDA Graph graba esos
lanzamientos una vez y los repite como una sola operacion: ~3 ms, con pesos
identicos a la version normal (ver tests/test_aprendiz.py).

Restriccion de los grafos: todas las entradas viven en tensores fijos en la GPU
(`_o`, `_a`, ...) y cada update copia el lote nuevo dentro de ellos. La red
objetivo se sincroniza con copias in-place, asi que el grafo sigue valido.
En CPU se usa el camino normal (sin grafo).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from dqn.config import ConfigDQN
from dqn.red import RedQ


class Aprendiz:
    def __init__(self, red: RedQ, objetivo: RedQ, cfg: ConfigDQN, device: torch.device, forma_obs, usar_grafo: bool = True):
        self.red, self.objetivo, self.cfg, self.device = red, objetivo, cfg, device
        self.usar_grafo = usar_grafo and device.type == "cuda"
        # capturable=True: el estado de Adam (incluido el contador de pasos)
        # vive en la GPU, requisito para capturarlo en el grafo. Se usa en GPU
        # aunque no haya grafo: su redondeo difiere del Adam normal (~1e-9) y
        # asi ambos caminos son bit a bit identicos.
        self.optim = torch.optim.Adam(red.parameters(), lr=cfg.lr, eps=cfg.adam_eps, capturable=device.type == "cuda")
        b = cfg.batch_size
        self._o = torch.zeros((b, *forma_obs), dtype=torch.uint8, device=device)
        self._o2 = torch.zeros_like(self._o)
        self._a = torch.zeros(b, dtype=torch.int64, device=device)
        self._r = torch.zeros(b, dtype=torch.float32, device=device)
        self._d = torch.zeros(b, dtype=torch.float32, device=device)
        #: Acumuladores de perdida y Q medio desde el ultimo `leer_metricas`.
        self._acum = torch.zeros(2, device=device)
        self.n_updates = 0
        self._grafo = None
        if self.usar_grafo:
            # Memoria fijada (pinned): permite copias a la GPU asincronas.
            self._pin = {
                n: torch.empty(t.shape, dtype=t.dtype).pin_memory()
                for n, t in (("obs", self._o), ("obs_sig", self._o2), ("acciones", self._a), ("recompensas", self._r), ("dones", self._d))
            }
            self._copia_lista = torch.cuda.Event()
            self._copia_lista.record()

    # ------------------------------------------------------------------ #
    def _paso(self) -> None:
        q = self.red(self._o).gather(1, self._a[:, None]).squeeze(1)
        with torch.no_grad():
            if self.cfg.double:
                # Double DQN: la red online elige a', la objetivo la evalua.
                a2 = self.red(self._o2).argmax(dim=1, keepdim=True)
                q2 = self.objetivo(self._o2).gather(1, a2).squeeze(1)
            else:
                q2 = self.objetivo(self._o2).max(dim=1).values
            y = self._r + self.cfg.gamma * (1.0 - self._d) * q2
        perdida = F.smooth_l1_loss(q, y)  # Huber
        self.optim.zero_grad(set_to_none=False)
        perdida.backward()
        torch.nn.utils.clip_grad_norm_(self.red.parameters(), self.cfg.max_norma_gradiente)
        self.optim.step()
        self._acum[0] += perdida.detach()
        self._acum[1] += q.detach().mean()

    def _capturar(self) -> None:
        """Calienta y graba el grafo sin alterar pesos ni estado de Adam."""
        pesos = [p.detach().clone() for p in self.red.parameters()]
        estado = {p: {k: v.clone() for k, v in s.items()} for p, s in self.optim.state.items()}
        lateral = torch.cuda.Stream()
        lateral.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(lateral):
            for _ in range(3):
                self._paso()
        torch.cuda.current_stream().wait_stream(lateral)
        self._grafo = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._grafo):
            self._paso()
        # Los pasos de calentamiento y captura si se ejecutaron: se deshacen
        # copiando in-place (reemplazar tensores invalidaria el grafo).
        with torch.no_grad():
            for p, w in zip(self.red.parameters(), pesos):
                p.copy_(w)
            for p, s in self.optim.state.items():
                for k, v in s.items():
                    if p in estado:
                        v.copy_(estado[p][k])
                    else:
                        v.zero_()
            self._acum.zero_()

    # ------------------------------------------------------------------ #
    def actualizar(self, lote: dict[str, np.ndarray]) -> None:
        if not self.usar_grafo:
            self._o.copy_(torch.from_numpy(lote["obs"]))
            self._o2.copy_(torch.from_numpy(lote["obs_sig"]))
            self._a.copy_(torch.from_numpy(lote["acciones"]))
            self._r.copy_(torch.from_numpy(lote["recompensas"]))
            self._d.copy_(torch.from_numpy(lote["dones"]).float())
            self._paso()
        else:
            # No escribir la memoria fijada mientras la copia anterior siga pendiente.
            self._copia_lista.synchronize()
            for nombre, destino in (("obs", self._o), ("obs_sig", self._o2), ("acciones", self._a), ("recompensas", self._r), ("dones", self._d)):
                self._pin[nombre].numpy()[:] = lote[nombre]
                destino.copy_(self._pin[nombre], non_blocking=True)
            self._copia_lista.record()
            if self._grafo is None:
                self._capturar()
            self._grafo.replay()
        self.n_updates += 1

    def sincronizar_objetivo(self) -> None:
        with torch.no_grad():
            for p_obj, p in zip(self.objetivo.parameters(), self.red.parameters()):
                p_obj.copy_(p)

    def leer_metricas(self) -> tuple[float, float]:
        """(perdida media, Q medio) desde la ultima lectura; reinicia los acumuladores."""
        n = max(self.n_updates, 1)
        perdida, q = (self._acum / n).tolist()
        self._acum.zero_()
        self.n_updates = 0
        return perdida, q
