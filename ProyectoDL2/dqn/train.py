"""Entrenamiento de Double + Dueling DQN.

Uso:
    python -m dqn.train --run piloto-1M --pasos 1000000
    python -m dqn.train --run ddqn-dueling-10M
    python -m dqn.train --run ddqn-dueling-10M --reanudar     # tras un corte

Cada corrida escribe en runs/<run>/:
    config_entorno.json, hiperparametros.json  -> para reconstruir el agente
    progreso.csv            -> perdida, Q medio, epsilon, fps (cada 10k pasos)
    episodios.csv           -> puntaje de cada partida de entrenamiento
    evaluaciones.csv        -> evaluacion greedy periodica
    checkpoints/paso_*.pt   -> pesos de cada evaluacion (para seleccionar despues)
    mejor.pt, ultimo.pt     -> mejor segun E[mejor de 5] / estado completo para reanudar

Dos detalles del bucle que NO estan en el entorno (ver dqn/entorno.py):
  - Recompensa recortada a {-1, 0, +1} (sign) para aprender; el puntaje que se
    registra es siempre el real.
  - Perder una vida corta el objetivo de Bellman (done=True en el buffer) pero
    la partida sigue: el agente aprende que morir es terminal sin que se
    reinicie el juego.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from dqn.aprendiz import Aprendiz
from dqn.config import ConfigDQN, ConfigEntorno
from dqn.entorno import crear_entornos_vectorizados
from dqn.metricas import resumen
from dqn.red import RedQ
from dqn.replay import ReplayBuffer

RAIZ_RUNS = Path(__file__).resolve().parent.parent / "runs"
PERIODO_LOG = 10_000


class RegistroCSV:
    """CSV en modo append: sobrevive a reanudaciones sin perder filas."""

    def __init__(self, ruta: Path, columnas: list[str]):
        nuevo = not ruta.exists()
        self.f = open(ruta, "a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=columnas)
        if nuevo:
            self.w.writeheader()

    def escribir(self, fila: dict) -> None:
        self.w.writerow(fila)
        self.f.flush()

    def cerrar(self) -> None:
        self.f.close()


def mantener_despierto() -> None:
    """Pide a Windows no suspender el equipo mientras este proceso viva.

    No cambia la configuracion de energia: es la misma solicitud temporal que
    hace un reproductor de video, y se libera sola al terminar el proceso.
    """
    import sys

    if sys.platform == "win32":
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def epsilon(paso: int, cfg: ConfigDQN) -> float:
    frac = min(paso / cfg.pasos_decaimiento_epsilon, 1.0)
    return cfg.epsilon_inicial + frac * (cfg.epsilon_final - cfg.epsilon_inicial)


def evaluar(red: RedQ, env_cfg: ConfigEntorno, n_episodios: int, n_envs: int, device, seed: int) -> list[float]:
    """Juega `n_episodios` completos (3 vidas) con politica greedy.

    Cada entorno juega exactamente n_episodios / n_envs partidas. Si se
    tomaran "las primeras n que terminen", se sesgaria la muestra hacia las
    partidas cortas, que son las peores.
    """
    por_env = int(np.ceil(n_episodios / n_envs))
    envs = crear_entornos_vectorizados(env_cfg, n_envs)
    obs, _ = envs.reset(seed=seed)
    acumulado = np.zeros(n_envs)
    puntajes: list[list[float]] = [[] for _ in range(n_envs)]
    red.eval()
    try:
        while any(len(p) < por_env for p in puntajes):
            with torch.no_grad():
                q = red(torch.from_numpy(obs).to(device))
            obs, r, term, trunc, _ = envs.step(q.argmax(dim=1).cpu().numpy())
            acumulado += r
            for i in np.flatnonzero(term | trunc):
                if len(puntajes[i]) < por_env:
                    puntajes[i].append(float(acumulado[i]))
                acumulado[i] = 0.0
    finally:
        envs.close()
        red.train()
    return [p for lista in puntajes for p in lista][:n_episodios]


def entrenar(nombre: str, cfg: ConfigDQN, env_cfg: ConfigEntorno, reanudar: bool, nota: str) -> None:
    carpeta = RAIZ_RUNS / nombre
    (carpeta / "checkpoints").mkdir(parents=True, exist_ok=True)
    mantener_despierto()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    envs = crear_entornos_vectorizados(env_cfg, cfg.n_envs)
    n_acciones = int(envs.single_action_space.n)
    red = RedQ(n_acciones, env_cfg.frames_apilados, cfg.dueling).to(device)
    objetivo = RedQ(n_acciones, env_cfg.frames_apilados, cfg.dueling).to(device)
    objetivo.eval()
    obs_forma = (env_cfg.frames_apilados, env_cfg.screen_size, env_cfg.screen_size)
    aprendiz = Aprendiz(red, objetivo, cfg, device, obs_forma)

    paso, mejor_e5 = 0, -np.inf
    if reanudar:
        estado = torch.load(carpeta / "ultimo.pt", map_location=device, weights_only=False)
        red.load_state_dict(estado["red"])
        aprendiz.optim.load_state_dict(estado["optim"])
        paso, mejor_e5 = estado["paso"], estado["mejor_e5"]
        print(f"Reanudando desde el paso {paso:,}. El buffer se vuelve a llenar desde cero.")
    else:
        if (carpeta / "ultimo.pt").exists():
            raise SystemExit(f"{carpeta} ya tiene una corrida. Usa --reanudar o otro --run.")
        env_cfg.guardar(carpeta / "config_entorno.json")
        cfg.guardar(carpeta / "hiperparametros.json")
        if nota:
            (carpeta / "nota.txt").write_text(nota + "\n", encoding="utf-8")
    aprendiz.sincronizar_objetivo()

    buffer = ReplayBuffer(cfg.capacidad_buffer, cfg.n_envs, (env_cfg.screen_size,) * 2, env_cfg.frames_apilados, cfg.seed)
    # Tras reanudar, se recolecta de nuevo antes de aprender (con el epsilon actual).
    aprender_desde = paso + cfg.pasos_antes_de_aprender

    log_prog = RegistroCSV(carpeta / "progreso.csv", ["paso", "minutos", "fps", "epsilon", "perdida", "q_medio", "puntaje_medio_20", "episodios"])
    log_eps = RegistroCSV(carpeta / "episodios.csv", ["paso", "puntaje", "largo"])
    log_eval = RegistroCSV(carpeta / "evaluaciones.csv", ["paso", "n", "media", "std", "min", "max", "E_mejor_de_5", "p20_mejor_de_5"])

    obs, info = envs.reset(seed=cfg.seed + paso)
    vidas = info["lives"].copy()
    primero = np.ones(cfg.n_envs, dtype=bool)
    ret_ep = np.zeros(cfg.n_envs)
    largo_ep = np.zeros(cfg.n_envs, dtype=int)
    ultimos: list[float] = []
    n_episodios = 0

    updates_pendientes = 0.0
    ultimo_objetivo = paso
    proxima_eval = (paso // cfg.periodo_evaluacion + 1) * cfg.periodo_evaluacion
    proximo_log = (paso // PERIODO_LOG + 1) * PERIODO_LOG
    t0, paso_t0, inicio = time.time(), paso, time.time()

    def guardar_ultimo():
        torch.save({"red": red.state_dict(), "optim": aprendiz.optim.state_dict(), "paso": paso, "mejor_e5": mejor_e5}, carpeta / "ultimo.pt")

    print(f"Corrida {nombre} | device={device} | acciones={n_acciones} | {cfg.pasos_totales:,} pasos")
    try:
        while paso < cfg.pasos_totales:
            # --- actuar (epsilon-greedy) ---
            eps = epsilon(paso, cfg)
            with torch.no_grad():
                greedy = red(torch.from_numpy(obs).to(device)).argmax(dim=1).cpu().numpy()
            azar = rng.random(cfg.n_envs) < eps
            acciones = np.where(azar, rng.integers(0, n_acciones, cfg.n_envs), greedy)

            # Los 8 entornos avanzan en sus procesos MIENTRAS la GPU aprende; el
            # lote sale del buffer tal como estaba antes de este paso.
            envs.step_async(acciones)

            # --- aprender ---
            if paso >= aprender_desde:
                updates_pendientes += cfg.n_envs / cfg.pasos_por_update
                while updates_pendientes >= 1:
                    updates_pendientes -= 1
                    aprendiz.actualizar(buffer.muestrear(cfg.batch_size))
                if paso - ultimo_objetivo >= cfg.periodo_red_objetivo:
                    aprendiz.sincronizar_objetivo()
                    ultimo_objetivo = paso

            obs_sig, r, term, trunc, info = envs.step_wait()
            fin = term | trunc
            vida_perdida = (info["lives"] < vidas) & ~fin
            # truncated tambien corta el bootstrap: sesgo minimo (el limite es
            # de 27k pasos, casi nunca se alcanza).
            buffer.agregar(obs, primero, acciones, np.sign(r), term | trunc | vida_perdida)
            vidas = info["lives"].copy()
            primero = fin
            obs = obs_sig

            ret_ep += r
            largo_ep += 1
            for i in np.flatnonzero(fin):
                log_eps.escribir({"paso": paso, "puntaje": ret_ep[i], "largo": largo_ep[i]})
                ultimos = (ultimos + [float(ret_ep[i])])[-20:]
                n_episodios += 1
                ret_ep[i], largo_ep[i] = 0.0, 0
            paso += cfg.n_envs

            # --- registro ---
            if paso >= proximo_log:
                proximo_log += PERIODO_LOG
                ahora = time.time()
                fps = (paso - paso_t0) / (ahora - t0)
                t0, paso_t0 = ahora, paso
                perdida, q_medio = aprendiz.leer_metricas()
                fila = {
                    "paso": paso,
                    "minutos": round((ahora - inicio) / 60, 1),
                    "fps": round(fps),
                    "epsilon": round(eps, 4),
                    "perdida": round(perdida, 5),
                    "q_medio": round(q_medio, 4),
                    "puntaje_medio_20": round(float(np.mean(ultimos)), 1) if ultimos else "",
                    "episodios": n_episodios,
                }
                log_prog.escribir(fila)
                print(f"paso {paso:>10,} | {fila['fps']:>5} fps | eps {eps:.3f} | perdida {fila['perdida']:.4f} | Q {fila['q_medio']:.3f} | ultimos 20: {fila['puntaje_medio_20']}")

            # --- evaluacion ---
            if paso >= proxima_eval:
                proxima_eval += cfg.periodo_evaluacion
                puntajes = evaluar(red, env_cfg, cfg.episodios_evaluacion, cfg.n_envs, device, seed=10_000 + paso)
                res = resumen(puntajes)
                log_eval.escribir({"paso": paso, **{k: round(v, 1) for k, v in res.items()}})
                torch.save(red.state_dict(), carpeta / "checkpoints" / f"paso_{paso:08d}.pt")
                marca = ""
                if res["E_mejor_de_5"] > mejor_e5:
                    mejor_e5 = res["E_mejor_de_5"]
                    torch.save(red.state_dict(), carpeta / "mejor.pt")
                    marca = "  <- mejor"
                guardar_ultimo()
                print(f"EVAL paso {paso:,}: media {res['media']:.0f} +- {res['std']:.0f} | max {res['max']:.0f} | E[mejor de 5] {res['E_mejor_de_5']:.0f}{marca}")
    except KeyboardInterrupt:
        print("Interrumpido: guardando estado para --reanudar.")
    finally:
        guardar_ultimo()
        torch.save(red.state_dict(), carpeta / "final.pt")
        envs.close()
        for log in (log_prog, log_eps, log_eval):
            log.cerrar()
    print(f"Listo: {paso:,} pasos en {(time.time() - inicio) / 60:.1f} min. Resultados en {carpeta}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="nombre de la corrida (carpeta en runs/)")
    p.add_argument("--reanudar", action="store_true")
    p.add_argument("--nota", default="", help="que cambia esta iteracion respecto a la anterior")
    base = ConfigDQN()
    for campo, valor in asdict(base).items():
        flag = "--" + campo.replace("_", "-")
        if isinstance(valor, bool):
            p.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "si"), default=None)
        else:
            p.add_argument(flag, type=type(valor), default=None)
    p.add_argument("--pasos", type=int, default=None, help="alias de --pasos-totales")
    args = p.parse_args()

    carpeta = RAIZ_RUNS / args.run
    if args.reanudar:
        # Al reanudar manda lo guardado; solo se permite extender --pasos.
        cfg = ConfigDQN.cargar(carpeta / "hiperparametros.json")
        env_cfg = ConfigEntorno.cargar(carpeta / "config_entorno.json")
        if args.pasos:
            cfg = replace(cfg, pasos_totales=args.pasos)
            cfg.guardar(carpeta / "hiperparametros.json")
    else:
        cambios = {k: v for k, v in vars(args).items() if k in asdict(base) and v is not None}
        if args.pasos:
            cambios["pasos_totales"] = args.pasos
        cfg = replace(base, **cambios)
        env_cfg = ConfigEntorno()
    entrenar(args.run, cfg, env_cfg, args.reanudar, args.nota)


if __name__ == "__main__":
    main()
