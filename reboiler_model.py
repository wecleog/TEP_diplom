"""
Куб-испаритель (Reboiler/Evaporator) — базовая модель процесса
================================================================
Структура:
  - ReboilerProcess  : внутренний блок (ОДУ процесса)
  - PIDController    : внешний блок управления
  - FaultInjector    : инжекция неисправностей для бенчмарка
  - Simulator        : оркестратор симуляции

Уравнения (баланс массы и энергии):
  dM/dt  = F_in - F_out - F_vap               [кг/с]  — без самовыравнивания
  dMx/dt = F_in*x_in - F_out*x_out - F_vap*y  [кг·мол.д./с]
  dU/dt  = Q_steam - Q_loss - F_vap*H_vap      [Вт]

Управляющие переменные (MV):
  u[0] = v_in    — положение клапана питания     [0..1]
  u[1] = v_out   — положение клапана выхода жидкости [0..1]
  u[2] = F_steam — расход греющего пара           [кг/с]

Возмущения (DV):
  d[0] = x_in    — мольная доля лёгкого компонента в питании [0..1]
  d[1] = T_in    — температура питания            [K]

Выходные переменные (PV):
  y[0] = L       — уровень жидкости в кубе        [м]
  y[1] = T       — температура жидкости            [K]
  y[2] = x       — мольная доля лёгкого компонента [0..1]
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Параметры процесса
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ProcessParams:
    """Физические параметры куба-испарителя."""

    # Геометрия
    A_cross: float = 2.0        # площадь сечения куба, м²

    # Расходные характеристики клапанов (линейные)
    Cv_in:  float = 5.0         # пропускная способность клапана питания, кг/с
    Cv_out: float = 4.0         # пропускная способность клапана выхода, кг/с

    # Тепловые параметры
    H_vap:   float = 2260e3     # теплота испарения воды, Дж/кг
    Cp_liq:  float = 4180.0     # теплоёмкость жидкости, Дж/(кг·К)
    Q_steam_max: float = 500e3  # максимальный тепловой поток греющего пара, Вт
    UA_loss: float = 200.0      # коэффициент тепловых потерь, Вт/К
    T_amb:   float = 298.15     # температура окружающей среды, К

    # Равновесие жидкость-пар (упрощённый закон Рауля)
    alpha:   float = 2.5        # относительная летучесть лёгкого компонента

    # Плотность жидкости
    rho:     float = 850.0      # кг/м³

    # Точка кипения (линейная апроксимация от состава)
    T_bp0:   float = 373.15     # температура кипения чистого тяжёлого компонента, К
    dT_bp:   float = -25.0      # понижение точки кипения при x=1, К

    # Номинальные начальные условия
    L0:     float = 1.0         # начальный уровень, м
    T0:     float = 360.0       # начальная температура, К
    x0:     float = 0.4         # начальная мольная доля


# ──────────────────────────────────────────────────────────────────────────────
# Внутренний блок: процесс
# ──────────────────────────────────────────────────────────────────────────────

class ReboilerProcess:
    """
    Внутренний блок — ОДУ куба-испарителя.

    Вектор состояния: z = [M, Mx, U]
      M  — масса жидкости в кубе, кг
      Mx — масса × мольная доля лёгкого компонента, кг
      U  — внутренняя энергия, Дж
    """

    def __init__(self, params: Optional[ProcessParams] = None):
        self.p = params or ProcessParams()
        self.z0 = self._initial_state()

    # ── Начальные условия ──────────────────────────────────────────────────

    def _initial_state(self) -> np.ndarray:
        p = self.p
        M0  = p.rho * p.A_cross * p.L0
        Mx0 = M0 * p.x0
        U0  = M0 * p.Cp_liq * p.T0
        return np.array([M0, Mx0, U0])

    # ── Вспомогательные вычисления ─────────────────────────────────────────

    def _state_to_outputs(self, z: np.ndarray) -> tuple:
        """Преобразование вектора состояния в физические величины."""
        p = self.p
        M, Mx, U = z
        M  = max(M, 1e-3)       # защита от отрицательных масс
        x  = np.clip(Mx / M, 0.0, 1.0)
        T  = U / (M * p.Cp_liq)
        L  = M / (p.rho * p.A_cross)
        return L, T, x

    def _equilibrium_y(self, x: float) -> float:
        """Мольная доля лёгкого компонента в паре (закон Рауля)."""
        p = self.p
        return p.alpha * x / (1.0 + (p.alpha - 1.0) * x)

    def _boiling_point(self, x: float) -> float:
        """Линейная апроксимация точки кипения от состава."""
        p = self.p
        return p.T_bp0 + p.dT_bp * x

    def _vaporization_rate(self, T: float, x: float, Q_net: float, M: float) -> float:
        """
        Скорость испарения.
        Предполагаем, что жидкость находится на линии кипения:
        избыток тепла идёт на испарение.
        """
        p = self.p
        T_bp = self._boiling_point(x)
        # Испарение пропорционально перегреву + тепловому потоку
        F_vap = max(Q_net / p.H_vap + 0.001 * M * (T - T_bp), 0.0)
        return F_vap

    # ── Правая часть ОДУ ───────────────────────────────────────────────────

    def ode(self, t: float, z: np.ndarray,
            u: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        dz/dt = f(z, u, d)

        u = [v_in, v_out, F_steam]  — управление
        d = [x_in, T_in]            — возмущения
        """
        p = self.p
        v_in, v_out, F_steam = np.clip(u, 0.0, 1.0)
        F_steam = F_steam * 1.0   # уже в кг/с (0..1 нормировка снаружи)
        x_in, T_in = d

        L, T, x = self._state_to_outputs(z)
        M = z[0]

        # Расходы через клапаны
        F_in  = p.Cv_in  * v_in
        F_out = p.Cv_out * v_out * np.sqrt(max(L, 0.0))  # зависит от напора

        # Тепловой баланс
        Q_steam = F_steam * p.Q_steam_max          # тепло от пара
        Q_loss  = p.UA_loss * (T - p.T_amb)        # потери в окружающую среду
        Q_net   = Q_steam - Q_loss

        # Равновесный состав пара
        y_vap = self._equilibrium_y(x)

        # Скорость испарения
        F_vap = self._vaporization_rate(T, x, Q_net, M)

        # Энтальпия питания
        H_in = p.Cp_liq * T_in

        # ── Уравнения баланса ────────────────────────────────────────────
        # Баланс массы: dM/dt = F_in - F_out - F_vap
        # ВАЖНО: нет самовыравнивания — при постоянных клапанах уровень дрейфует
        dM_dt  = F_in - F_out - F_vap

        # Баланс компонента: d(Mx)/dt
        dMx_dt = F_in * x_in - F_out * x - F_vap * y_vap

        # Баланс энергии: dU/dt
        dU_dt  = F_in * H_in - F_out * p.Cp_liq * T + Q_net - F_vap * p.H_vap

        return np.array([dM_dt, dMx_dt, dU_dt])

    def outputs(self, z: np.ndarray) -> np.ndarray:
        """Вернуть вектор выходных переменных y = [L, T, x]."""
        L, T, x = self._state_to_outputs(z)
        return np.array([L, T, x])


# ──────────────────────────────────────────────────────────────────────────────
# Внешний блок: ПИД-регулятор
# ──────────────────────────────────────────────────────────────────────────────

class PIDController:
    """
    Простой ПИД с насыщением и anti-windup.

    Управляет одним каналом: измеренный выход → управляющий сигнал.
    """

    def __init__(self, Kp: float, Ki: float, Kd: float,
                 u_min: float = 0.0, u_max: float = 1.0,
                 setpoint: float = 0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.u_min = u_min
        self.u_max = u_max
        self.setpoint = setpoint

        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_t = None

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_t = None

    def step(self, t: float, measurement: float) -> float:
        """Один шаг ПИД, возвращает управляющий сигнал."""
        error = self.setpoint - measurement
        dt = (t - self._prev_t) if self._prev_t is not None else 1e-3
        dt = max(dt, 1e-6)

        # Интегральная составляющая с anti-windup
        self._integral += error * dt

        # Дифференциальная составляющая
        d_term = (error - self._prev_error) / dt if self._prev_t is not None else 0.0

        u = self.Kp * error + self.Ki * self._integral + self.Kd * d_term

        # Насыщение + anti-windup
        u_sat = np.clip(u, self.u_min, self.u_max)
        if u != u_sat:
            self._integral -= error * dt  # отмотать интеграл

        self._prev_error = error
        self._prev_t = t
        return u_sat


# ──────────────────────────────────────────────────────────────────────────────
# Инжектор неисправностей (для бенчмарка)
# ──────────────────────────────────────────────────────────────────────────────

# Каталог неисправностей — соответствует структуре TEP benchmark
FAULT_CATALOG = {
    0:  {"name": "Норма",                    "type": None},
    1:  {"name": "Скачок состава питания",   "type": "step_disturbance",  "channel": "x_in",    "magnitude": +0.15},
    2:  {"name": "Медленный дрейф состава",  "type": "ramp_disturbance",  "channel": "x_in",    "rate": 0.002},
    3:  {"name": "Засорение клапана питания","type": "valve_stiction",    "channel": "v_in",     "factor": 0.6},
    4:  {"name": "Утечка теплоносителя",     "type": "gain_loss",         "channel": "F_steam",  "factor": 0.7},
    5:  {"name": "Залипание клапана выхода", "type": "valve_stuck",       "channel": "v_out",    "value": 0.3},
    6:  {"name": "Шум датчика уровня",       "type": "sensor_noise",      "channel": "L",        "std": 0.05},
    7:  {"name": "Смещение датчика темп.",   "type": "sensor_bias",       "channel": "T",        "bias": +5.0},
}


class FaultInjector:
    """
    Инжектирует неисправности в управляющие сигналы,
    возмущения и измерения.
    """

    def __init__(self, fault_id: int = 0, fault_start: float = 100.0):
        self.fault_id = fault_id
        self.fault_start = fault_start
        self.fault = FAULT_CATALOG.get(fault_id, FAULT_CATALOG[0])
        self._ramp_base = None

    def apply_to_inputs(self, t: float,
                        u: np.ndarray, d: np.ndarray) -> tuple:
        """Модифицирует u и d согласно типу неисправности."""
        u = u.copy()
        d = d.copy()
        if t < self.fault_start or self.fault["type"] is None:
            return u, d

        ftype   = self.fault["type"]
        channel = self.fault.get("channel", "")

        if ftype == "step_disturbance" and channel == "x_in":
            d[0] = np.clip(d[0] + self.fault["magnitude"], 0.0, 1.0)

        elif ftype == "ramp_disturbance" and channel == "x_in":
            if self._ramp_base is None:
                self._ramp_base = d[0]
            d[0] = np.clip(self._ramp_base + self.fault["rate"] * (t - self.fault_start), 0.0, 1.0)

        elif ftype == "valve_stiction" and channel == "v_in":
            u[0] *= self.fault["factor"]

        elif ftype == "gain_loss" and channel == "F_steam":
            u[2] *= self.fault["factor"]

        elif ftype == "valve_stuck" and channel == "v_out":
            u[1] = self.fault["value"]

        return u, d

    def apply_to_outputs(self, t: float, y: np.ndarray) -> np.ndarray:
        """Добавляет шум/смещение к измерениям."""
        y = y.copy()
        if t < self.fault_start or self.fault["type"] is None:
            return y

        ftype   = self.fault["type"]
        channel = self.fault.get("channel", "")

        if ftype == "sensor_noise" and channel == "L":
            y[0] += np.random.normal(0, self.fault["std"])

        elif ftype == "sensor_bias" and channel == "T":
            y[1] += self.fault["bias"]

        return y


# ──────────────────────────────────────────────────────────────────────────────
# Симулятор
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    t:      np.ndarray          # время, с
    y:      np.ndarray          # выходы процесса [N x 3]: L, T, x
    u:      np.ndarray          # управление     [N x 3]: v_in, v_out, F_steam
    d:      np.ndarray          # возмущения     [N x 2]: x_in, T_in
    z:      np.ndarray          # состояние      [N x 3]
    fault_id: int = 0
    fault_start: float = 100.0


class Simulator:
    """
    Оркестратор: связывает процесс, регулятор и инжектор неисправностей.

    Схема управления:
      Уровень L  → ПИД_L  → клапан выхода v_out
      Температура T → ПИД_T → расход пара F_steam
      Клапан питания v_in = const (или задаётся вручную)
    """

    def __init__(self,
                 process:    Optional[ReboilerProcess] = None,
                 pid_level:  Optional[PIDController]  = None,
                 pid_temp:   Optional[PIDController]  = None,
                 fault_id:   int   = 0,
                 fault_start: float = 100.0):

        self.process = process or ReboilerProcess()

        # ПИД по уровню: управляет клапаном выхода
        self.pid_level = pid_level or PIDController(
            Kp=0.8, Ki=0.05, Kd=0.1,
            u_min=0.0, u_max=1.0,
            setpoint=self.process.p.L0
        )

        # ПИД по температуре: управляет расходом пара
        self.pid_temp = pid_temp or PIDController(
            Kp=0.005, Ki=0.0005, Kd=0.001,
            u_min=0.0, u_max=1.0,
            setpoint=self.process.p.T0
        )

        self.fault_injector = FaultInjector(fault_id, fault_start)

    def run(self,
            t_end:    float = 300.0,
            dt:       float = 1.0,
            v_in_nom: float = 0.5,
            x_in_nom: float = 0.4,
            T_in_nom: float = 340.0) -> SimulationResult:
        """
        Запустить симуляцию.

        Параметры:
            t_end     : длительность симуляции, с
            dt        : шаг дискретизации регулятора, с
            v_in_nom  : номинальное открытие клапана питания
            x_in_nom  : номинальный состав питания
            T_in_nom  : номинальная температура питания, К
        """
        t_span_all = np.arange(0.0, t_end + dt, dt)
        n_steps = len(t_span_all) - 1

        # Хранение результатов
        t_log = np.zeros(n_steps)
        y_log = np.zeros((n_steps, 3))   # L, T, x
        u_log = np.zeros((n_steps, 3))   # v_in, v_out, F_steam
        d_log = np.zeros((n_steps, 2))   # x_in, T_in
        z_log = np.zeros((n_steps, 3))   # состояние

        z = self.process.z0.copy()
        self.pid_level.reset()
        self.pid_temp.reset()

        for i in range(n_steps):
            t0 = t_span_all[i]
            t1 = t_span_all[i + 1]

            # Текущие выходы процесса
            y_true = self.process.outputs(z)

            # Базовые управление и возмущения
            d_base = np.array([x_in_nom, T_in_nom])
            u_base = np.array([
                v_in_nom,
                self.pid_level.step(t0, y_true[0]),  # уровень → v_out
                self.pid_temp.step(t0, y_true[1]),    # температура → F_steam
            ])

            # Применить неисправности
            u_eff, d_eff = self.fault_injector.apply_to_inputs(t0, u_base, d_base)
            y_meas = self.fault_injector.apply_to_outputs(t0, y_true)

            # Интегрирование ОДУ на шаг [t0, t1]
            sol = solve_ivp(
                fun=lambda t, z_: self.process.ode(t, z_, u_eff, d_eff),
                t_span=(t0, t1),
                y0=z,
                method='Radau',     # жёсткий решатель
                rtol=1e-4,
                atol=1e-6,
            )
            z = sol.y[:, -1]

            # Логирование
            t_log[i]   = t0
            y_log[i]   = y_meas          # то, что "видит" система (с возможным шумом)
            u_log[i]   = u_eff
            d_log[i]   = d_eff
            z_log[i]   = z

        return SimulationResult(
            t=t_log, y=y_log, u=u_log, d=d_log, z=z_log,
            fault_id=self.fault_injector.fault_id,
            fault_start=self.fault_injector.fault_start,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Визуализация
# ──────────────────────────────────────────────────────────────────────────────

def plot_simulation(res: SimulationResult, title: str = ""):
    fault_name = FAULT_CATALOG[res.fault_id]["name"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f"{title}\nНеисправность {res.fault_id}: «{fault_name}»  "
                 f"(начало в t={res.fault_start} с)", fontsize=12)

    # Выходные переменные
    axes[0, 0].plot(res.t, res.y[:, 0], 'b')
    axes[0, 0].axvline(res.fault_start, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_ylabel("Уровень L, м")
    axes[0, 0].set_title("Уровень в кубе")
    axes[0, 0].grid(True)

    axes[1, 0].plot(res.t, res.y[:, 1], 'r')
    axes[1, 0].axvline(res.fault_start, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_ylabel("Температура T, К")
    axes[1, 0].set_title("Температура жидкости")
    axes[1, 0].grid(True)

    axes[2, 0].plot(res.t, res.y[:, 2], 'g')
    axes[2, 0].axvline(res.fault_start, color='r', linestyle='--', alpha=0.5)
    axes[2, 0].set_ylabel("Мольная доля x")
    axes[2, 0].set_title("Состав жидкости")
    axes[2, 0].set_xlabel("Время, с")
    axes[2, 0].grid(True)

    # Управляющие сигналы
    axes[0, 1].plot(res.t, res.u[:, 0], 'm')
    axes[0, 1].axvline(res.fault_start, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_ylabel("v_in [0..1]")
    axes[0, 1].set_title("Клапан питания")
    axes[0, 1].grid(True)

    axes[1, 1].plot(res.t, res.u[:, 1], 'c')
    axes[1, 1].axvline(res.fault_start, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_ylabel("v_out [0..1]")
    axes[1, 1].set_title("Клапан выхода (ПИД по уровню)")
    axes[1, 1].grid(True)

    axes[2, 1].plot(res.t, res.u[:, 2], 'orange')
    axes[2, 1].axvline(res.fault_start, color='r', linestyle='--', alpha=0.5)
    axes[2, 1].set_ylabel("F_steam [0..1]")
    axes[2, 1].set_title("Расход греющего пара (ПИД по температуре)")
    axes[2, 1].set_xlabel("Время, с")
    axes[2, 1].grid(True)

    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Точка входа — быстрая демонстрация
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Куб-испаритель — симуляция всех неисправностей")
    print("=" * 60)

    results = {}

    for fault_id in FAULT_CATALOG:
        print(f"\n[Fault {fault_id}] {FAULT_CATALOG[fault_id]['name']}")
        sim = Simulator(fault_id=fault_id, fault_start=100.0)
        res = sim.run(t_end=300.0, dt=1.0)
        results[fault_id] = res
        print(f"  L итог: {res.y[-1, 0]:.3f} м | "
              f"T итог: {res.y[-1, 1]:.1f} К | "
              f"x итог: {res.y[-1, 2]:.3f}")

    # Показать графики для нормы и двух неисправностей
    for fid in [0, 1, 4]:
        fig = plot_simulation(results[fid], title="Куб-испаритель TEP")
        fig.savefig(f"fault_{fid}.png", dpi=100, bbox_inches='tight')
        print(f"График сохранён: fault_{fid}.png")

    plt.show()
    print("\nГотово.")
