"""Metrica de la competencia: el mejor de 5 episodios.

Con 5 episodios, el maximo es muy ruidoso: dos agentes con el mismo promedio
pueden tener maximos muy distintos segun cuanto varien sus partidas. Para
comparar checkpoints se juegan N >> 5 episodios y se estima la distribucion
del "mejor de 5" a partir de esa muestra.

Si x_(1) <= ... <= x_(n) son los puntajes ordenados, el maximo de k partidas
(muestreadas con reemplazo de la distribucion empirica) es x_(i) con
probabilidad (i/n)^k - ((i-1)/n)^k. De ahi salen su esperanza y sus cuantiles
sin simulacion.
"""

from __future__ import annotations

import numpy as np


def _pesos_maximo(n: int, k: int) -> np.ndarray:
    i = np.arange(1, n + 1)
    return (i / n) ** k - ((i - 1) / n) ** k


def esperanza_mejor_de_k(puntajes, k: int = 5) -> float:
    x = np.sort(np.asarray(puntajes, dtype=float))
    return float(np.dot(_pesos_maximo(len(x), k), x))


def cuantil_mejor_de_k(puntajes, q: float, k: int = 5) -> float:
    """Puntaje que el mejor de k supera con probabilidad 1 - q."""
    x = np.sort(np.asarray(puntajes, dtype=float))
    acumulada = np.cumsum(_pesos_maximo(len(x), k))
    return float(x[np.searchsorted(acumulada, q)])


def resumen(puntajes, k: int = 5) -> dict[str, float]:
    x = np.asarray(puntajes, dtype=float)
    return {
        "n": len(x),
        "media": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        f"E_mejor_de_{k}": esperanza_mejor_de_k(x, k),
        # Escenario pesimista: 1 de cada 5 veces el mejor de 5 queda por debajo.
        f"p20_mejor_de_{k}": cuantil_mejor_de_k(x, 0.2, k),
    }
