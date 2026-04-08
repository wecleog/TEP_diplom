"""
Интерфейс Streamlit — Куб-испаритель + Бенчмарк
================================================
Запуск:
    pip install streamlit numpy scipy matplotlib scikit-learn
    streamlit run app.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from io import BytesIO
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2
import warnings
warnings.filterwarnings("ignore")

# Импорт модели и бенчмарка
from reboiler_model import (
    ReboilerProcess, ProcessParams,
    PIDController, Simulator,
    FaultInjector, FAULT_CATALOG,
)
from benchmark import (
    PCADetector, CUSUMDetector, ZScoreDetector,
    compute_metrics,
)

# ─────────────────────────────────────────────────────────────────────────────
# Настройка страницы
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Куб-испаритель TEP",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .main { background: #0f1117; }

    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; color: #e8f4fd; }

    .metric-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-left: 3px solid #4fc3f7;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 6px 0;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        color: #4fc3f7;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .fault-badge {
        display: inline-block;
        background: #2d1b1b;
        border: 1px solid #e53e3e;
        color: #fc8181;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 3px;
        margin: 2px;
    }
    .normal-badge {
        display: inline-block;
        background: #1a2e1b;
        border: 1px solid #38a169;
        color: #68d391;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 3px;
    }
    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.15em;
        color: #4fc3f7;
        text-transform: uppercase;
        border-bottom: 1px solid #2d3748;
        padding-bottom: 4px;
        margin-bottom: 12px;
    }
    .stButton > button {
        background: #4fc3f7;
        color: #0f1117;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 2rem;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #81d4fa;
        transform: translateY(-1px);
    }
    div[data-testid="stSidebarContent"] {
        background: #141824;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Заголовок
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# ⚗️ Куб-испаритель")
st.markdown(
    "<p style='color:#718096; font-size:0.9rem; margin-top:-12px;'>"
    "Математическая модель + бенчмарк обнаружения неисправностей"
    "</p>", unsafe_allow_html=True
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Боковая панель — все параметры
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Параметры")

    # ── Симуляция ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⏱ Симуляция</div>', unsafe_allow_html=True)
    t_end       = st.slider("Длительность, с",    100, 1000, 400, step=50)
    dt          = st.select_slider("Шаг дискретизации, с", options=[0.5, 1.0, 2.0], value=1.0)
    fault_start = st.slider("Начало неисправности, с", 10, int(t_end * 0.8), 150, step=10)

    # ── Неисправность ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⚠️ Неисправность</div>', unsafe_allow_html=True)
    fault_options = {v["name"]: k for k, v in FAULT_CATALOG.items()}
    fault_name_sel = st.selectbox("Тип неисправности", list(fault_options.keys()))
    fault_id = fault_options[fault_name_sel]

    # ── Номинальные условия ───────────────────────────────────────────────
    st.markdown('<div class="section-header">🏭 Номинальный режим</div>', unsafe_allow_html=True)
    v_in_nom = st.slider("Клапан питания v_in",     0.1, 1.0, 0.5, step=0.05)
    x_in_nom = st.slider("Состав питания x_in",     0.1, 0.9, 0.4, step=0.05)
    T_in_nom = st.slider("Температура питания, К",  300, 380, 340, step=5)

    # ── ПИД по уровню ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📐 ПИД — уровень → v_out</div>', unsafe_allow_html=True)
    pid_L_sp = st.slider("Уставка уровня, м",  0.3, 2.5, 1.0, step=0.1)
    pid_L_Kp = st.slider("Kp (уровень)",       0.1, 3.0, 0.8, step=0.1)
    pid_L_Ki = st.slider("Ki (уровень)",        0.0, 0.5, 0.05, step=0.01)
    pid_L_Kd = st.slider("Kd (уровень)",        0.0, 1.0, 0.1, step=0.05)

    # ── ПИД по температуре ────────────────────────────────────────────────
    st.markdown('<div class="section-header">🌡 ПИД — температура → пар</div>', unsafe_allow_html=True)
    pid_T_sp = st.slider("Уставка температуры, К", 330, 390, 360, step=5)
    pid_T_Kp = st.slider("Kp (температура)",       0.001, 0.02, 0.005, step=0.001, format="%.3f")
    pid_T_Ki = st.slider("Ki (температура)",        0.0, 0.005, 0.0005, step=0.0001, format="%.4f")
    pid_T_Kd = st.slider("Kd (температура)",        0.0, 0.01, 0.001, step=0.001, format="%.3f")

    # ── Бенчмарк ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Детекторы</div>', unsafe_allow_html=True)
    use_pca    = st.checkbox("PCA (T² + SPE)",  value=True)
    use_cusum  = st.checkbox("CUSUM",           value=True)
    use_zscore = st.checkbox("Z-score",         value=True)
    pca_alpha  = st.slider("Порог PCA (α)",     0.90, 0.999, 0.99, step=0.005, format="%.3f")
    cusum_h    = st.slider("Порог CUSUM (h)",   1.0, 10.0, 4.0, step=0.5)
    zscore_thr = st.slider("Порог Z-score (σ)", 2.0, 5.0, 3.5, step=0.25)

    st.divider()
    run_btn = st.button("▶  Запустить симуляцию", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Вкладки
# ─────────────────────────────────────────────────────────────────────────────

tab_sim, tab_bench, tab_info = st.tabs(["📈 Симуляция", "🔬 Бенчмарк", "📖 Справка"])

# ─────────────────────────────────────────────────────────────────────────────
# Кэшированная функция обучения (только на нормальных данных)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_train_data(v_in, x_in, T_in, dt_val,
                   pid_L_sp_, pid_L_Kp_, pid_L_Ki_, pid_L_Kd_,
                   pid_T_sp_, pid_T_Kp_, pid_T_Ki_, pid_T_Kd_):
    """Генерация обучающих данных (нормальный режим). Кэшируется."""
    proc = ReboilerProcess()
    pid_l = PIDController(pid_L_Kp_, pid_L_Ki_, pid_L_Kd_, setpoint=pid_L_sp_)
    pid_t = PIDController(pid_T_Kp_, pid_T_Ki_, pid_T_Kd_, setpoint=pid_T_sp_)
    sim = Simulator(proc, pid_l, pid_t, fault_id=0, fault_start=99999)
    res = sim.run(t_end=300.0, dt=dt_val,
                  v_in_nom=v_in, x_in_nom=x_in, T_in_nom=T_in)
    return np.hstack([res.y, res.u])


def run_simulation(fault_id_, fault_start_, t_end_, dt_,
                   v_in_, x_in_, T_in_,
                   pid_L_sp_, pid_L_Kp_, pid_L_Ki_, pid_L_Kd_,
                   pid_T_sp_, pid_T_Kp_, pid_T_Ki_, pid_T_Kd_):
    proc  = ReboilerProcess()
    pid_l = PIDController(pid_L_Kp_, pid_L_Ki_, pid_L_Kd_, setpoint=pid_L_sp_)
    pid_t = PIDController(pid_T_Kp_, pid_T_Ki_, pid_T_Kd_, setpoint=pid_T_sp_)
    sim   = Simulator(proc, pid_l, pid_t,
                      fault_id=fault_id_, fault_start=fault_start_)
    return sim.run(t_end=t_end_, dt=dt_,
                   v_in_nom=v_in_, x_in_nom=x_in_, T_in_nom=T_in_)


# ─────────────────────────────────────────────────────────────────────────────
# Состояние сессии
# ─────────────────────────────────────────────────────────────────────────────

if "res" not in st.session_state:
    st.session_state.res = None

if run_btn:
    with st.spinner("Симуляция..."):
        st.session_state.res = run_simulation(
            fault_id, fault_start, t_end, dt,
            v_in_nom, x_in_nom, T_in_nom,
            pid_L_sp, pid_L_Kp, pid_L_Ki, pid_L_Kd,
            pid_T_sp, pid_T_Kp, pid_T_Ki, pid_T_Kd,
        )

res = st.session_state.res

# ─────────────────────────────────────────────────────────────────────────────
# Вкладка 1: Симуляция
# ─────────────────────────────────────────────────────────────────────────────

with tab_sim:
    if res is None:
        st.info("Настройте параметры в боковой панели и нажмите **▶ Запустить симуляцию**.")
    else:
        fault_info = FAULT_CATALOG[res.fault_id]
        badge_html = (
            f'<span class="fault-badge">FAULT {res.fault_id}: {fault_info["name"]}</span>'
            if res.fault_id > 0
            else '<span class="normal-badge">НОРМА</span>'
        )
        st.markdown(f"**Режим:** {badge_html}", unsafe_allow_html=True)

        # ── Метрики итогового состояния ───────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Уровень (конец)</div>
                <div class="metric-value">{res.y[-1, 0]:.3f} м</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{res.y[-1, 1]:.1f} К</div>
                <div class="metric-label">Температура (конец)</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{res.y[-1, 2]:.3f}</div>
                <div class="metric-label">Состав x (конец)</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{res.t[-1]:.0f} с</div>
                <div class="metric-label">Время симуляции</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── Графики ───────────────────────────────────────────────────────
        fig = plt.figure(figsize=(14, 8), facecolor='#0f1117')
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        plot_cfg = [
            (0, 0, res.y[:, 0], "Уровень L, м",          "#4fc3f7"),
            (0, 1, res.y[:, 1], "Температура T, К",       "#f06292"),
            (0, 2, res.y[:, 2], "Состав x",               "#81c784"),
            (1, 0, res.u[:, 0], "Клапан питания v_in",    "#ffb74d"),
            (1, 1, res.u[:, 1], "Клапан выхода v_out",    "#ce93d8"),
            (1, 2, res.u[:, 2], "Расход пара F_steam",    "#80cbc4"),
        ]

        for row, col, data, label, color in plot_cfg:
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor('#141824')
            ax.plot(res.t, data, color=color, linewidth=1.5)
            if res.fault_id > 0:
                ax.axvline(res.fault_start, color='#e53e3e',
                           linestyle='--', linewidth=1.2, alpha=0.8)
            ax.set_ylabel(label, color='#a0aec0', fontsize=8)
            ax.set_xlabel("t, с", color='#718096', fontsize=7)
            ax.tick_params(colors='#718096', labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor('#2d3748')
            ax.grid(True, color='#2d3748', linewidth=0.5, linestyle=':')

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches='tight',
                    facecolor='#0f1117')
        st.image(buf.getvalue(), use_container_width=True)
        plt.close(fig)

        # Скачать CSV
        import pandas as pd
        df_export = pd.DataFrame({
            "t":       res.t,
            "L":       res.y[:, 0],
            "T":       res.y[:, 1],
            "x":       res.y[:, 2],
            "v_in":    res.u[:, 0],
            "v_out":   res.u[:, 1],
            "F_steam": res.u[:, 2],
            "x_in":    res.d[:, 0],
            "T_in":    res.d[:, 1],
        })
        st.download_button(
            "⬇ Скачать данные CSV",
            df_export.to_csv(index=False).encode(),
            file_name=f"simulation_fault{res.fault_id}.csv",
            mime="text/csv",
        )

# ─────────────────────────────────────────────────────────────────────────────
# Вкладка 2: Бенчмарк
# ─────────────────────────────────────────────────────────────────────────────

with tab_bench:
    if res is None:
        st.info("Сначала запустите симуляцию.")
    else:
        if not any([use_pca, use_cusum, use_zscore]):
            st.warning("Выберите хотя бы один детектор в боковой панели.")
        else:
            with st.spinner("Обучение детекторов и расчёт метрик..."):

                # Обучающие данные
                X_train = get_train_data(
                    v_in_nom, x_in_nom, T_in_nom, dt,
                    pid_L_sp, pid_L_Kp, pid_L_Ki, pid_L_Kd,
                    pid_T_sp, pid_T_Kp, pid_T_Ki, pid_T_Kd,
                )
                X_test = np.hstack([res.y, res.u])

                detectors   = {}
                alarms      = {}
                stat_series = {}

                if use_pca:
                    d = PCADetector(n_components=3, alpha=pca_alpha)
                    d.fit(X_train)
                    alarm, T2, Q = d.predict(X_test)
                    detectors["PCA"]   = d
                    alarms["PCA"]      = alarm
                    stat_series["PCA"] = {"T2": T2, "Q": Q}

                if use_cusum:
                    d = CUSUMDetector(k=0.5, h=cusum_h)
                    d.fit(X_train)
                    alarm, Sp, Sn = d.predict(X_test)
                    detectors["CUSUM"]   = d
                    alarms["CUSUM"]      = alarm
                    stat_series["CUSUM"] = {"S+": Sp, "S-": Sn}

                if use_zscore:
                    d = ZScoreDetector(threshold=zscore_thr)
                    d.fit(X_train)
                    alarm, Zsc = d.predict(X_test)
                    detectors["Z-score"]   = d
                    alarms["Z-score"]      = alarm
                    stat_series["Z-score"] = {"Z_max": Zsc.max(axis=1)}

                # ── Метрики ────────────────────────────────────────────────
                st.markdown('<div class="section-header">Метрики детекторов</div>',
                            unsafe_allow_html=True)

                cols = st.columns(len(alarms))
                color_map = {"PCA": "#4fc3f7", "CUSUM": "#f06292", "Z-score": "#81c784"}

                for col, (name, alarm) in zip(cols, alarms.items()):
                    m = compute_metrics(alarm, res.t, res.fault_start)
                    add_str = f"{m['ADD']:.0f} с" if not np.isnan(m['ADD']) else "—"
                    color   = color_map.get(name, "#fff")
                    with col:
                        st.markdown(f"""
                        <div class="metric-card" style="border-left-color:{color}">
                            <div class="metric-label">{name}</div>
                            <div style="margin-top:8px">
                                <span style="color:{color};font-family:monospace;font-size:1.1rem">
                                    DR = {m['DR']:.2f}
                                </span><br>
                                <span style="color:#718096;font-size:0.85rem">
                                    FAR = {m['FAR']:.2f} &nbsp;|&nbsp; ADD = {add_str}
                                </span>
                            </div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("")

                # ── Графики статистик + тревог ─────────────────────────────
                st.markdown('<div class="section-header">Статистики детекторов</div>',
                            unsafe_allow_html=True)

                n_det = len(alarms)
                fig2, axes = plt.subplots(n_det, 1,
                                          figsize=(13, 3.2 * n_det),
                                          facecolor='#0f1117')
                if n_det == 1:
                    axes = [axes]

                for ax, (name, alarm) in zip(axes, alarms.items()):
                    ax.set_facecolor('#141824')
                    color = color_map.get(name, "#fff")

                    # Основная статистика
                    stat_key = list(stat_series[name].keys())[0]
                    stat_val = stat_series[name][stat_key]
                    ax.plot(res.t, stat_val, color=color,
                            linewidth=1.2, label=stat_key, alpha=0.9)

                    # Зона тревог
                    ax.fill_between(res.t, 0, stat_val.max(),
                                    where=alarm,
                                    color='#e53e3e', alpha=0.15, label='Тревога')

                    # Линия начала неисправности
                    if res.fault_id > 0:
                        ax.axvline(res.fault_start, color='#e53e3e',
                                   linestyle='--', linewidth=1.2,
                                   label=f'Fault start (t={res.fault_start}с)')

                    ax.set_title(f"{name}", color='#e8f4fd',
                                 fontsize=9, fontfamily='monospace')
                    ax.set_ylabel(stat_key, color='#a0aec0', fontsize=8)
                    ax.set_xlabel("t, с", color='#718096', fontsize=7)
                    ax.tick_params(colors='#718096', labelsize=7)
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#2d3748')
                    ax.grid(True, color='#2d3748', linewidth=0.5, linestyle=':')
                    ax.legend(fontsize=7, facecolor='#1a1f2e',
                              labelcolor='#a0aec0', framealpha=0.8)

                buf2 = BytesIO()
                fig2.savefig(buf2, format="png", dpi=110,
                             bbox_inches='tight', facecolor='#0f1117')
                st.image(buf2.getvalue(), use_container_width=True)
                plt.close(fig2)

                # ── Сводная таблица по всем неисправностям ─────────────────
                if st.checkbox("Показать сводную таблицу по всем неисправностям"):
                    st.markdown('<div class="section-header">Сводная таблица DR</div>',
                                unsafe_allow_html=True)

                    with st.spinner("Прогон всех 7 неисправностей..."):
                        rows = []
                        for fid, finfo in FAULT_CATALOG.items():
                            if fid == 0:
                                continue
                            sim_f = Simulator(
                                fault_id=fid, fault_start=fault_start,
                                pid_level=PIDController(
                                    pid_L_Kp, pid_L_Ki, pid_L_Kd,
                                    setpoint=pid_L_sp),
                                pid_temp=PIDController(
                                    pid_T_Kp, pid_T_Ki, pid_T_Kd,
                                    setpoint=pid_T_sp),
                            )
                            r_f = sim_f.run(t_end=t_end, dt=dt,
                                            v_in_nom=v_in_nom,
                                            x_in_nom=x_in_nom,
                                            T_in_nom=T_in_nom)
                            Xf = np.hstack([r_f.y, r_f.u])
                            row = {"Неисправность": f"F{fid}: {finfo['name']}"}
                            for dname, det in detectors.items():
                                al, *_ = det.predict(Xf)
                                m = compute_metrics(al, r_f.t, fault_start)
                                row[f"{dname} DR"]  = f"{m['DR']:.2f}"
                                row[f"{dname} FAR"] = f"{m['FAR']:.2f}"
                                row[f"{dname} ADD"] = (f"{m['ADD']:.0f}с"
                                                       if not np.isnan(m['ADD']) else "—")
                            rows.append(row)

                    import pandas as pd
                    df_table = pd.DataFrame(rows).set_index("Неисправность")
                    st.dataframe(df_table, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Вкладка 3: Справка
# ─────────────────────────────────────────────────────────────────────────────

with tab_info:
    st.markdown("## Структура модели")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Внутренний блок — процесс")
        st.markdown("""
Система из **3 ОДУ** (баланс массы, компонента, энергии):

$$\\frac{dM}{dt} = F_{in} - F_{out} - F_{vap}$$

$$\\frac{d(Mx)}{dt} = F_{in}x_{in} - F_{out}x - F_{vap}y$$

$$\\frac{dU}{dt} = Q_{steam} - Q_{loss} - F_{vap}H_{vap}$$

**Управляющие переменные:**
- `v_in` — клапан питания
- `v_out` — клапан выхода жидкости  
- `F_steam` — расход греющего пара

**Возмущения:**
- `x_in` — состав питания
- `T_in` — температура питания
        """)

    with col2:
        st.markdown("### Внешний блок — управление")
        st.markdown("""
Два **ПИД-контура**:

| Контур | Вход | Выход |
|--------|------|-------|
| ПИД по уровню | L (м) | v_out |
| ПИД по температуре | T (К) | F_steam |

**Свойства объекта:**
- ⚠️ Без самовыравнивания по уровню (интегрирующий)
- ⚠️ Возможны автоколебания при неоптимальных ПИД
        """)

    st.divider()
    st.markdown("### Каталог неисправностей")
    fault_df_data = [
        {"ID": k, "Название": v["name"], "Тип": v["type"] or "—"}
        for k, v in FAULT_CATALOG.items()
    ]
    import pandas as pd
    st.dataframe(pd.DataFrame(fault_df_data).set_index("ID"),
                 use_container_width=True)

    st.divider()
    st.markdown("### Метрики бенчмарка")
    st.markdown("""
| Метрика | Описание |
|---------|----------|
| **DR** — Detection Rate | Доля шагов с тревогой после начала неисправности |
| **FAR** — False Alarm Rate | Доля ложных тревог в нормальном режиме |
| **ADD** — Avg. Detection Delay | Время от начала неисправности до первой тревоги |
    """)
