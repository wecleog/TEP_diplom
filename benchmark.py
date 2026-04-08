"""
Бенчмарк обнаружения неисправностей
=====================================
Методы:
  1. PCA + T² Хотеллинга + SPE (Q-статистика)  — классический TEP-бенчмарк
  2. CUSUM (накопительная сумма)                 — чувствителен к малым сдвигам
  3. Пороговый детектор по z-score               — простейший базовый метод

Метрики:
  - FAR  (False Alarm Rate)    — доля ложных тревог на нормальных данных
  - DR   (Detection Rate)      — доля обнаружений на данных с неисправностью
  - ADD  (Average Detection Delay) — среднее время задержки обнаружения
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from reboiler_model import (
    Simulator, ReboilerProcess, FAULT_CATALOG, plot_simulation
)


# ──────────────────────────────────────────────────────────────────────────────
# Детекторы
# ──────────────────────────────────────────────────────────────────────────────

class PCADetector:
    """
    PCA-монитор процесса: T²-статистика Хотеллинга и Q (SPE).
    Обучается на нормальных данных, затем применяется онлайн.
    """

    def __init__(self, n_components: int = 2, alpha: float = 0.99):
        self.n_components = n_components
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self._T2_lim = None
        self._Q_lim  = None

    def fit(self, X: np.ndarray):
        """Обучение на нормальных данных X [N x n_features]."""
        Xs = self.scaler.fit_transform(X)
        self.pca.fit(Xs)
        T2, Q = self._compute_stats(Xs)

        n, p = X.shape
        a = self.n_components
        # Порог T² по F-распределению (приближение через хи-квадрат)
        self._T2_lim = chi2.ppf(self.alpha, df=a)
        # Порог Q: метод Джексона
        self._Q_lim  = np.percentile(Q, self.alpha * 100)

    def _compute_stats(self, Xs: np.ndarray):
        T = self.pca.transform(Xs)
        T2 = np.sum((T / np.sqrt(self.pca.explained_variance_)) ** 2, axis=1)
        Xs_hat = self.pca.inverse_transform(T)
        e = Xs - Xs_hat
        Q = np.sum(e ** 2, axis=1)
        return T2, Q

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Вернуть булев массив: True = тревога."""
        Xs = self.scaler.transform(X)
        T2, Q = self._compute_stats(Xs)
        return (T2 > self._T2_lim) | (Q > self._Q_lim), T2, Q


class CUSUMDetector:
    """CUSUM по каждой переменной. Тревога при превышении порога h."""

    def __init__(self, k: float = 0.5, h: float = 5.0):
        self.k = k     # допустимое отклонение в сигмах
        self.h = h     # порог срабатывания

    def fit(self, X: np.ndarray):
        self.mu    = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-9

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = (X - self.mu) / self.sigma
        S_pos = np.zeros(len(X))
        S_neg = np.zeros(len(X))
        alarm = np.zeros(len(X), dtype=bool)
        for i in range(len(X)):
            S_pos[i] = max(0, (S_pos[i-1] if i > 0 else 0) + z[i].mean() - self.k)
            S_neg[i] = max(0, (S_neg[i-1] if i > 0 else 0) - z[i].mean() - self.k)
            alarm[i]  = (S_pos[i] > self.h) or (S_neg[i] > self.h)
        return alarm, S_pos, S_neg


class ZScoreDetector:
    """Простейший детектор: тревога если |z| > threshold."""

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold

    def fit(self, X: np.ndarray):
        self.mu    = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-9

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = np.abs((X - self.mu) / self.sigma)
        alarm = z.max(axis=1) > self.threshold
        return alarm, z


# ──────────────────────────────────────────────────────────────────────────────
# Метрики бенчмарка
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(alarm: np.ndarray,
                    t: np.ndarray,
                    fault_start: float) -> dict:
    """
    alarm      : булев массив тревог
    t          : временная ось
    fault_start: момент начала неисправности
    """
    idx_fault = np.searchsorted(t, fault_start)

    # Нормальный период — до неисправности
    alarm_normal = alarm[:idx_fault]
    # Период неисправности — после
    alarm_faulty = alarm[idx_fault:]

    FAR = alarm_normal.mean() if len(alarm_normal) > 0 else 0.0
    DR  = alarm_faulty.mean() if len(alarm_faulty) > 0 else 0.0

    # Задержка обнаружения — первый момент непрерывной тревоги после fault_start
    ADD = np.nan
    if DR > 0:
        for i, a in enumerate(alarm_faulty):
            if a:
                ADD = t[idx_fault + i] - fault_start
                break

    return {"FAR": FAR, "DR": DR, "ADD": ADD}


# ──────────────────────────────────────────────────────────────────────────────
# Запуск бенчмарка
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmark(t_end: float = 400.0,
                  fault_start: float = 150.0,
                  dt: float = 1.0):

    print("=" * 65)
    print("БЕНЧМАРК ОБНАРУЖЕНИЯ НЕИСПРАВНОСТЕЙ — Куб-испаритель")
    print("=" * 65)

    # ── 1. Генерация обучающих данных (нормальный режим) ──────────────────
    print("\n[1/3] Генерация нормальных данных для обучения детекторов...")
    sim_train = Simulator(fault_id=0, fault_start=9999)
    res_train = sim_train.run(t_end=300.0, dt=dt)

    # Признаки: выходы + управление (как в TEP benchmark)
    X_train = np.hstack([res_train.y, res_train.u])
    print(f"      Обучающая выборка: {X_train.shape[0]} точек × {X_train.shape[1]} признаков")

    # ── 2. Обучение детекторов ─────────────────────────────────────────────
    print("\n[2/3] Обучение детекторов...")
    pca_det    = PCADetector(n_components=3, alpha=0.99)
    cusum_det  = CUSUMDetector(k=0.5, h=4.0)
    zscore_det = ZScoreDetector(threshold=3.5)

    pca_det.fit(X_train)
    cusum_det.fit(X_train)
    zscore_det.fit(X_train)
    print("      PCA, CUSUM, Z-score — обучены.")

    # ── 3. Тестирование на всех неисправностях ────────────────────────────
    print(f"\n[3/3] Тестирование (t_end={t_end}с, fault_start={fault_start}с)...\n")

    results_table = []

    for fault_id, fault_info in FAULT_CATALOG.items():
        sim = Simulator(fault_id=fault_id, fault_start=fault_start)
        res = sim.run(t_end=t_end, dt=dt)
        X_test = np.hstack([res.y, res.u])

        # Тревоги от каждого детектора
        alarm_pca,    T2, Q  = pca_det.predict(X_test)
        alarm_cusum,  Sp, Sn = cusum_det.predict(X_test)
        alarm_zscore, Zsc    = zscore_det.predict(X_test)

        # Метрики
        m_pca    = compute_metrics(alarm_pca,    res.t, fault_start)
        m_cusum  = compute_metrics(alarm_cusum,  res.t, fault_start)
        m_zscore = compute_metrics(alarm_zscore, res.t, fault_start)

        row = {
            "fault_id": fault_id,
            "name":     fault_info["name"],
            "pca":      m_pca,
            "cusum":    m_cusum,
            "zscore":   m_zscore,
        }
        results_table.append(row)

        add_pca    = f"{m_pca['ADD']:.0f}с"    if not np.isnan(m_pca['ADD'])    else "—"
        add_cusum  = f"{m_cusum['ADD']:.0f}с"  if not np.isnan(m_cusum['ADD'])  else "—"
        add_zscore = f"{m_zscore['ADD']:.0f}с" if not np.isnan(m_zscore['ADD']) else "—"

        print(f"  F{fault_id:02d} «{fault_info['name']:<32}»")
        print(f"       PCA:    FAR={m_pca['FAR']:.2f}  DR={m_pca['DR']:.2f}  ADD={add_pca}")
        print(f"       CUSUM:  FAR={m_cusum['FAR']:.2f}  DR={m_cusum['DR']:.2f}  ADD={add_cusum}")
        print(f"       ZScore: FAR={m_zscore['FAR']:.2f}  DR={m_zscore['DR']:.2f}  ADD={add_zscore}")

    # ── Итоговая таблица ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("ИТОГОВАЯ ТАБЛИЦА (Detection Rate)")
    print(f"{'Неисправность':<35} {'PCA':>6} {'CUSUM':>7} {'ZScore':>8}")
    print("-" * 65)
    for row in results_table:
        fid = row["fault_id"]
        if fid == 0:
            continue  # норма — нет смысла в DR
        print(f"  F{fid} {row['name']:<33}"
              f" {row['pca']['DR']:>5.2f}"
              f"  {row['cusum']['DR']:>5.2f}"
              f"  {row['zscore']['DR']:>6.2f}")

    # ── Графики бенчмарка ─────────────────────────────────────────────────
    _plot_benchmark_summary(results_table)

    return results_table


def _plot_benchmark_summary(results_table: list):
    """Сводный график DR по всем неисправностям."""
    fault_ids   = [r["fault_id"]     for r in results_table if r["fault_id"] > 0]
    names       = [f"F{r['fault_id']}\n{r['name'][:18]}" for r in results_table if r["fault_id"] > 0]
    dr_pca      = [r["pca"]["DR"]    for r in results_table if r["fault_id"] > 0]
    dr_cusum    = [r["cusum"]["DR"]  for r in results_table if r["fault_id"] > 0]
    dr_zscore   = [r["zscore"]["DR"] for r in results_table if r["fault_id"] > 0]

    x = np.arange(len(fault_ids))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - width, dr_pca,    width, label="PCA (T²+SPE)", color='steelblue')
    ax.bar(x,         dr_cusum,  width, label="CUSUM",         color='tomato')
    ax.bar(x + width, dr_zscore, width, label="Z-score",        color='seagreen')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Detection Rate (доля обнаружений)")
    ax.set_title("Бенчмарк обнаружения неисправностей — куб-испаритель\n"
                 "Сравнение методов: PCA vs CUSUM vs Z-score")
    ax.axhline(1.0, color='black', linestyle='--', alpha=0.3)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig("benchmark_summary.png", dpi=120, bbox_inches='tight')
    print("\nГрафик бенчмарка сохранён: benchmark_summary.png")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_benchmark(t_end=400.0, fault_start=150.0, dt=1.0)
