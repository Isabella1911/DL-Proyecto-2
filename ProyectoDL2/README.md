# Proyecto 2 CC3092: Double + Dueling DQN en Space Invaders

Agente de aprendizaje por refuerzo para `ALE/SpaceInvaders-v5`, implementado
desde cero en PyTorch: **Double DQN + arquitectura Dueling**, entrenado 10M pasos
(40M frames).

## Resultado

La competencia toma **la mejor de 5 partidas**. El agente final (`agente_final/`,
checkpoint de 9.25M pasos) obtuvo:

| Evaluación | Partidas | Media | Máx | E[mejor de 5] | Mejor de 5 real |
|---|---:|---:|---:|---:|---:|
| Selección (semillas nuevas) | 64 | 1417 ± 601 | 2865 | 2127 | — |
| Protocolo de competencia | 5 | 1813 ± 755 | 2595 | — | **2595** |

Como referencia, en el mismo análisis la política aleatoria da 147 de media
(E[mejor de 5] = 275) y "siempre disparar" da 285 en todas las partidas.

## Instalación

Requiere **Python 3.11** (las versiones de `requirements.txt` están fijadas).
**No hace falta GPU** para evaluar: en CPU el agente juega igual.

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Después, torch (uno de los dos):

```bash
# con GPU NVIDIA
.venv\Scripts\python.exe -m pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
# sin GPU
.venv\Scripts\python.exe -m pip install torch==2.5.1
```

En Linux/macOS la ruta es `.venv/bin/python`. Se llama al Python del entorno
directamente porque en PowerShell `Activate.ps1` suele estar bloqueado.

## Evaluar el agente (día de la presentación)

```bash
.venv\Scripts\python.exe -m dqn.evaluate --run agente_final --episodios 5 --video
```

Carga los pesos, reconstruye el entorno con **el mismo preprocesamiento del
entrenamiento** (leído de `agente_final/config_entorno.json`), juega 5 partidas
completas (3 vidas) con política greedy, imprime el puntaje de cada una, la
mejor y el promedio, y graba los videos en `agente_final/videos/`. Tarda ~2 min.

### Cargar los pesos desde código

```python
from pathlib import Path
from dqn.evaluate import cargar_agente
from dqn.entorno import crear_entorno, ejecutar_episodio

agente, config, _ = cargar_agente(Path("agente_final"))
env = crear_entorno(config)
print(ejecutar_episodio(env, agente))   # {'pasos': ..., 'puntaje': ..., 'truncado': False}
```

`modelo.pt` es un `state_dict` de `dqn.red.RedQ(n_acciones=6, canales=4, dueling=True)`.

## Reproducir el entrenamiento

```bash
.venv\Scripts\python.exe -m dqn.analisis_entorno                       # análisis del entorno (sección 2.1)
.venv\Scripts\python.exe -m dqn.train --run ddqn-dueling-10M           # ~4.2 h en una RTX 3070 Laptop
.venv\Scripts\python.exe -m dqn.seleccion --run ddqn-dueling-10M --top 5 --episodios 64
.venv\Scripts\python.exe -m dqn.curvas --run ddqn-dueling-10M          # figuras
```

Si el entrenamiento se corta, se retoma con `--reanudar`. Pruebas de las dos
piezas delicadas: `python -m tests.test_replay` y `python -m tests.test_aprendiz`
(esta última requiere GPU).

## Estructura

| Ruta | Contenido |
|---|---|
| `dqn/config.py` | `ConfigEntorno` (preprocesamiento) y `ConfigDQN` (hiperparámetros); se guardan como JSON junto a los pesos |
| `dqn/entorno.py` | `crear_entorno`, `ejecutar_episodio`, `generar_video_agente` (estilo Lab 5) |
| `dqn/red.py` | CNN de Nature DQN + cabeza Dueling |
| `dqn/replay.py` | Replay buffer que guarda un frame por transición (500k = 3.5 GB) |
| `dqn/aprendiz.py` | Paso de Double DQN capturado en un CUDA Graph |
| `dqn/train.py` | Bucle de entrenamiento, evaluación periódica, checkpoints |
| `dqn/metricas.py` | Estimación de E[mejor de 5] a partir de N partidas |
| `dqn/seleccion.py` | Re-evalúa los mejores checkpoints y copia el ganador a `agente_final/` |
| `dqn/evaluate.py` | Protocolo de la competencia |
| `dqn/analisis_entorno.py`, `dqn/curvas.py` | Análisis del entorno y figuras |
| `agente_final/` | Pesos, configuración, origen y videos del agente entregado |
| `runs/registro_iteraciones.csv` | Tabla de iteraciones (sección 2.3) |
| `runs/ddqn-dueling-10M/` | CSV de entrenamiento y evaluación, checkpoints candidatos, `mejor.pt`, `final.pt` |
| `informe/figuras/` | Figuras para el informe |

## Configuración

**Entorno**: `ALE/SpaceInvaders-v5`, sticky actions 0.25, 6 acciones mínimas,
frame skip 4 con max-pool, 84×84 en gris, 4 frames apilados, hasta 30 NOOPs
al inicio. En entrenamiento, recompensa recortada a su signo y pérdida de vida
como terminal para el objetivo de Bellman (sin reiniciar la partida); la
evaluación usa el puntaje real y la partida completa.

**Algoritmo**: Adam (lr 1e-4, eps 1.5e-4), pérdida de Huber, γ = 0.99, batch 32,
un update cada 4 pasos, red objetivo cada 8000 pasos, ε lineal de 1.0 a 0.01
en 1M pasos, buffer de 500k, 50k pasos antes de aprender, 8 entornos en paralelo.
