"""
Игрушечная визуализация логистической сети: без диспетчера vs с AI-диспетчером.

Запуск:
    source venv/bin/activate
    streamlit run viz_app.py

Что показывает:
  * узлы-домики (🏭 склад, 🏢 хаб, 🏠 ПВЗ) краснеют по мере загрузки и 
    перегружаются 💥 (queue > capacity);
  * заказы 🚆 едут по дорогам (рёбрам графа);
  * слева — пассивная сеть , справа — та же сеть под управлением
    AI-диспетчера (LightGBM -> Decision Tree-правила -> объезд/приоритет).

"""

import copy
import random
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import run_closed_loop as R
from data.simulation import build_network
from src.data_loaders.loaders import FEATURE_COLUMNS
from src.dispatcher.ai_dispatcher import AIDispatcher
from src.dispatcher.closed_loop import simulate_with_dispatcher


FEATURE_RU = {
    "node_type_code": "тип узла",
    "capacity": "ёмкость",
    "queue": "очередь",
    "load_ratio": "загрузка",
    "congestion": "затор",
    "active_orders": "активных заказов",
    "delivered_orders": "доставлено всего",
    "stuck_orders": "застряло всего",
    "load_ratio_lag1": "загрузка 1 шаг назад",
    "load_ratio_lag2": "загрузка 2 шага назад",
    "load_ratio_lag3": "загрузка 3 шага назад",
    "queue_delta": "прирост очереди",
    "upstream_load": "загрузка источников (склады/хабы выше)",
    "downstream_load": "загрузка получателей (хабы/ПВЗ ниже)",
}


def humanize_rules(tree, feature_names, threshold, max_rules=8):
    t = tree.tree_
    rules = []

    def walk(node, conds):
        if t.feature[node] == -2:  # лист
            counts = t.value[node][0]
            p1 = counts[1] / counts.sum() if counts.sum() else 0.0
            if p1 > threshold and conds:
                rules.append((p1, list(conds)))
            return
        name = FEATURE_RU.get(feature_names[t.feature[node]], feature_names[t.feature[node]])
        thr = t.threshold[node]
        walk(t.children_left[node], conds + [f"{name} ≤ {thr:.2f}"])
        walk(t.children_right[node], conds + [f"{name} > {thr:.2f}"])

    walk(0, [])
    rules.sort(reverse=True)  # сначала самые рискованные
    out = []
    for p1, conds in rules[:max_rules]:
        out.append(f"ЕСЛИ " + " И ".join(conds) +
                   f"  →  риск перегрузки {p1:.0%}: объехать узел / поднять приоритет")
    return out

VIZ_STEPS = 120
ORDER_CAP = 70
VIZ_SEED = 7
PLOTLY_FRAME_DURATION_MS = 180

EMOJI = {"warehouse": "🏭", "hub": "🏢", "pvz": "🏠"}
XCOL = {"warehouse": 0.0, "hub": 1.3, "pvz": 2.6}
SIDE_OFFSET = 4.2
LOAD_SCALE = [[0.0, "#2ecc71"], [0.5, "#f4d03f"], [1.0, "#e74c3c"]]


def init_session_state():
    if "demo_result" not in st.session_state:
        st.session_state["demo_result"] = None
    if "animation_done" not in st.session_state:
        st.session_state["animation_done"] = False


def layout_positions(graph, x_offset):
    by_type = {"warehouse": [], "hub": [], "pvz": []}
    for n, a in graph.nodes(data=True):
        by_type[a["type"]].append(n)
    pos = {}
    for t, nodes in by_type.items():
        nodes = sorted(nodes)
        k = max(len(nodes), 1)
        for i, n in enumerate(nodes):
            y = 4.0 * (1 - (i + 0.5) / k)
            pos[n] = (XCOL[t] + x_offset, y)
    return pos


def step_index(df):
    """time -> {node_id: (load_ratio, is_overload, queue)} и time -> delivered."""
    loads, delivered = {}, {}
    for row in df.itertuples():
        loads.setdefault(row.time, {})[row.node_id] = (row.load_ratio, row.is_overload, row.queue)
        delivered[row.time] = row.delivered_orders
    return loads, delivered


def autoplay_plotly_animation(fig):
    autoplay_fig = copy.deepcopy(fig)
    autoplay_fig.update_layout(updatemenus=[], sliders=[])
    html = autoplay_fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        auto_play=True,
        animation_opts={
            "frame": {"duration": PLOTLY_FRAME_DURATION_MS, "redraw": True},
            "transition": {"duration": 0},
            "fromcurrent": True,
        },
        config={"displayModeBar": False, "responsive": True},
    )
    components.html(html, height=680, scrolling=False)
    return len(autoplay_fig.frames) * PLOTLY_FRAME_DURATION_MS / 1000 + 1.2


@st.cache_resource(show_spinner="Готовлю симуляцию и обучаю модели...")
def build_demo():
    random.seed(0)
    np.random.seed(0)

    graph = build_network(**R.NETWORK)
    csv_path = R.generate_training_data(graph)
    model, X_train = R.train_model(csv_path)
    rule_tree, rules, fidelity = R.train_rule_tree(model, X_train)
    human_rules = humanize_rules(rule_tree, FEATURE_COLUMNS, R.RISK_THRESHOLD)

    # Пассивный режим (без управления)
    g_p = copy.deepcopy(graph)
    log_p = []
    df_p, orders_p, actions_p = simulate_with_dispatcher(
        g_p, dispatcher=None, use_reroute=True,
        n_steps=VIZ_STEPS, seed=VIZ_SEED,
        order_log=log_p, order_log_cap=ORDER_CAP, **R.REGIME,
    )

    # AI-диспетчер (Decision Tree-правила поверх LightGBM)
    g_a = copy.deepcopy(graph)
    log_a = []
    disp = AIDispatcher(model, rule_tree=rule_tree, risk_threshold=R.RISK_THRESHOLD)
    df_a, orders_a, actions = simulate_with_dispatcher(
        g_a, dispatcher=disp, use_reroute=True,
        proactive_reroute_fraction=R.PROACTIVE_FRACTION,
        risk_weight=R.RISK_WEIGHT,
        n_steps=VIZ_STEPS, seed=VIZ_SEED,
        order_log=log_a, order_log_cap=ORDER_CAP, **R.REGIME,
    )

    pos_p = layout_positions(graph, 0.0)
    pos_a = layout_positions(graph, SIDE_OFFSET)
    L_p, del_p = step_index(df_p)
    L_a, del_a = step_index(df_a)

    node_points, nx_, ny_, nemoji = [], [], [], []
    for side, pos in (("p", pos_p), ("a", pos_a)):
        for n, a in graph.nodes(data=True):
            node_points.append((n, side))
            x, y = pos[n]
            nx_.append(x)
            ny_.append(y)
            nemoji.append(EMOJI[a["type"]])

    # Статичные дороги (рёбра) для обеих панелей
    ex, ey = [], []
    for u, v in graph.edges():
        for pos in (pos_p, pos_a):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ex += [x0, x1, None]
            ey += [y0, y1, None]
    edge_trace = go.Scatter(
        x=ex, y=ey, mode="lines",
        line=dict(color="rgba(130,130,130,0.30)", width=1),
        hoverinfo="skip", showlegend=False,
    )

    def nodes_trace(step):
        colors, sizes, hover = [], [], []
        for (n, side) in node_points:
            L = L_p if side == "p" else L_a
            lr, ov, q = L[step][n]
            colors.append(min(lr, 1.5))
            sizes.append(16 + 26 * min(lr, 1.6))
            hover.append(f"{n}<br>load={lr:.2f}<br>queue={q}")
        return go.Scatter(
            x=nx_, y=ny_, mode="markers+text", text=nemoji,
            textposition="middle center", textfont=dict(size=13),
            marker=dict(color=colors, colorscale=LOAD_SCALE, cmin=0.0, cmax=1.5,
                        size=sizes, line=dict(color="white", width=1)),
            hovertext=hover, hoverinfo="text", showlegend=False,
        )

    def expl_trace(step):
        xs, ys = [], []
        for i, (n, side) in enumerate(node_points):
            L = L_p if side == "p" else L_a
            if L[step][n][1] == 1:
                xs.append(nx_[i])
                ys.append(ny_[i])
        return go.Scatter(
            x=xs, y=ys, mode="text", text=["💥"] * len(xs),
            textfont=dict(size=28), hoverinfo="skip", showlegend=False,
        )

    def _orders_xy(step, log, pos):
        xs, ys = [], []
        snap = log[step] if step < len(log) else []
        for (cur, nxt) in snap:
            if cur not in pos or nxt not in pos:
                continue
            x0, y0 = pos[cur]
            x1, y1 = pos[nxt]
            f = 0.2 + 0.6 * random.random()
            xs.append(x0 + (x1 - x0) * f)
            ys.append(y0 + (y1 - y0) * f)
        return xs, ys

    def orders_trace(step):
        xp, yp = _orders_xy(step, log_p, pos_p)
        xa, ya = _orders_xy(step, log_a, pos_a)
        xs, ys = xp + xa, yp + ya
        return go.Scatter(
            x=xs, y=ys, mode="text", text=["🚆"] * len(xs),
            textfont=dict(size=11), hoverinfo="skip", showlegend=False,
        )

    def annotations(step):
        ov_p = sum(1 for i, (n, s) in enumerate(node_points) if s == "p" and L_p[step][n][1] == 1)
        ov_a = sum(1 for i, (n, s) in enumerate(node_points) if s == "a" and L_a[step][n][1] == 1)
        cx_p = XCOL["hub"] + 0.0
        cx_a = XCOL["hub"] + SIDE_OFFSET
        return [
            dict(x=cx_p, y=4.55, xref="x", yref="y", showarrow=False,
                 text="🚫 <b>Без диспетчера</b>", font=dict(size=18, color="#c0392b")),
            dict(x=cx_a, y=4.55, xref="x", yref="y", showarrow=False,
                 text="🤖 <b>С AI-диспетчером</b>", font=dict(size=18, color="#1e8449")),
            dict(x=cx_p, y=4.2, xref="x", yref="y", showarrow=False,
                 text=f"шаг {step} · 💥 перегружено: {ov_p} · доставлено: {del_p[step]}",
                 font=dict(size=13, color="#555")),
            dict(x=cx_a, y=4.2, xref="x", yref="y", showarrow=False,
                 text=f"шаг {step} · 💥 перегружено: {ov_a} · доставлено: {del_a[step]}",
                 font=dict(size=13, color="#555")),
        ]

    frames = [
        go.Frame(
            name=str(t),
            data=[orders_trace(t), nodes_trace(t), expl_trace(t)],
            traces=[1, 2, 3],
            layout=go.Layout(annotations=annotations(t)),
        )
        for t in range(VIZ_STEPS)
    ]

    fig = go.Figure(
        data=[edge_trace, orders_trace(0), nodes_trace(0), expl_trace(0)],
        frames=frames,
    )

    slider = dict(
        active=0, y=0, x=0.05, len=0.9,
        currentvalue=dict(prefix="Шаг: ", font=dict(size=14)),
        steps=[dict(method="animate", label=str(t),
                    args=[[str(t)], dict(mode="immediate",
                                         frame=dict(duration=0, redraw=True),
                                         transition=dict(duration=0))])
               for t in range(VIZ_STEPS)],
    )
    play = dict(
        type="buttons", showactive=False, x=0.05, y=1.12,
        buttons=[
            dict(label="▶ Играть", method="animate",
                 args=[None, dict(frame=dict(duration=140, redraw=True),
                                  fromcurrent=True, transition=dict(duration=0))]),
            dict(label="⏸ Пауза", method="animate",
                 args=[[None], dict(mode="immediate",
                                    frame=dict(duration=0, redraw=False))]),
        ],
    )

    fig.update_layout(
        height=640, plot_bgcolor="#f7f9fb", paper_bgcolor="white",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False, range=[-0.6, 2.6 + SIDE_OFFSET + 0.4]),
        yaxis=dict(visible=False, range=[-0.3, 4.8]),
        annotations=annotations(0),
        updatemenus=[play], sliders=[slider],
    )

    # Перегрузки возникают только на хабах (узкое место); склады/ПВЗ имеют запас и
    # «разбавляют» среднее по всей сети. Поэтому ключевую метрику считаем по хабам.
    hub_p = df_p[df_p["node_type"] == "hub"]
    hub_a = df_a[df_a["node_type"] == "hub"]
    delivered_p = [o for o in orders_p if o.delivered]
    delivered_a = [o for o in orders_a if o.delivered]

    mean_queue_wait_p = sum(o.queue_wait_time for o in delivered_p) / max(len(delivered_p), 1)
    mean_queue_wait_a = sum(o.queue_wait_time for o in delivered_a) / max(len(delivered_a), 1)
    summary = {
        "overload_rate": (float(df_p["is_overload"].mean()), float(df_a["is_overload"].mean())),
        "hub_overload_rate": (float(hub_p["is_overload"].mean()), float(hub_a["is_overload"].mean())),
        "max_load_ratio": (float(df_p["load_ratio"].max()), float(df_a["load_ratio"].max())),
        "mean_hub_load": (float(hub_p["load_ratio"].mean()), float(hub_a["load_ratio"].mean())),
        "delivered": (int(df_p["delivered_orders"].max()), int(df_a["delivered_orders"].max())),
        "created": (len(orders_p), len(orders_a)),
        "mean_queue_wait": (mean_queue_wait_p, mean_queue_wait_a),
        "reroutes": (int(actions_p.get("reroute", 0)), int(actions.get("reroute", 0))),
    }
    return fig, rules, human_rules, fidelity, summary, actions


def main():
    st.set_page_config(layout="wide", page_title="WB · AI-диспетчер")
    init_session_state()

    st.title("📦 Динамическая симуляция логистики: без диспетчера vs AI-диспетчер")
    st.caption("Домики — узлы сети (🏭 склад · 🏢 хаб · 🏠 ПВЗ). 🚆 — заказы на дорогах. "
               "💥 — перегрузка узла (queue > capacity).")

    if st.session_state["demo_result"] is None:
        with st.spinner("Создаю модель и считаю симуляцию..."):
            st.session_state["demo_result"] = build_demo()

    fig, rules, human_rules, fidelity, summary, actions = st.session_state["demo_result"]

    if not st.session_state["animation_done"]:
        duration = autoplay_plotly_animation(fig)
        time.sleep(duration)
        st.session_state["animation_done"] = True
        st.rerun()
        return

    if st.button("▶ Запустить симуляцию", type="primary"):
        st.session_state["animation_done"] = False
        st.rerun()

    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    hop, hoa = summary["hub_overload_rate"]
    mlp, mla = summary["mean_hub_load"]
    mxp, mxa = summary["max_load_ratio"]
    dp, da = summary["delivered"]
    mtp, mta = summary["mean_queue_wait"]
    rp, ra = summary["reroutes"]
    c1.metric("Перегрузки хабов", f"{hoa:.1%}", f"{(hoa - hop):.1%} vs без AI", delta_color="inverse")
    c2.metric("Средняя загрузка хабов", f"{mla:.2f}", f"{(mla - mlp):.2f} vs без AI", delta_color="inverse")
    c3.metric("Пиковая загрузка", f"{mxa:.2f}×", f"{(mxa - mxp):.2f} vs без AI", delta_color="inverse")
    c4.metric("Доставлено заказов", f"{da}", f"{da - dp} vs без AI")
    c5.metric("Среднее время в очереди обработки", f"{mta:.2f}", f"{(mta - mtp):.2f} vs без AI", delta_color="inverse")
    c6.metric("Перемаршруты", f"{ra}", f"без AI: {rp}")
    st.caption("Перегрузки считаются по хабам — это узкое место сети (склады и ПВЗ имеют "
               "запас ёмкости). По всей сети доля перегрузок: "
               f"без AI {summary['overload_rate'][0]:.3f} → с AI {summary['overload_rate'][1]:.3f}.")

    st.subheader("📋 Правила, которые исполняет диспетчер")
    st.caption(f"Decision Tree обучен на предсказаниях LightGBM (fidelity {fidelity:.3f}). "
               "Каждое правило — путь в дереве, ведущий к высокому риску перегрузки:")
    if human_rules:
        for r in human_rules:
            st.markdown(f"- {r}")
    else:
        st.info("В текущем режиме дерево не выделило правил выше порога риска.")

    with st.expander("⚙️ Действия диспетчера и полное дерево (export_text)"):
        st.write({k: int(v) for k, v in actions.items()})
        st.code(rules)


if __name__ == "__main__":
    main()
