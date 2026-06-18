"""
Пайплайн:
  1. Генерируем обучающие данные пассивным прогоном (без объездов) — там
     перегрузки возникают естественно, и модели есть что предсказывать.
  2. Обучаем LightGBM: P(overload) на прогнозируемый горизонт шагов вперёд.
  3. На нескольких сценариях (seed) сравниваем 3 режима на одном потоке заказов:
        passive  — без объездов (что было бы без управления);
        reactive — объезд по факту пробки (реагируем, когда уже поздно);
        ai       — объезд по предсказанию модели (+ смена ПВЗ, приоритет).
  4. Усредняем метрики, печатаем сравнение и SHAP-объяснение.

Запуск:  python run_closed_loop.py
"""

import copy
import os
import random

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text

from data.simulation import build_network
from src.data_handlers.prepare_data import prepare_simulation_data
from src.data_loaders.loaders import CSVDataLoader, FEATURE_COLUMNS
from src.dispatcher.ai_dispatcher import AIDispatcher
from src.dispatcher.closed_loop import simulate_with_dispatcher

# ── Сеть: узкое место на хабах, просторные склады/ПВЗ, плотные связи ──────────
NETWORK = dict(
    n_nodes=85, n_warehouses=10, n_hubs=25, n_pvz=50,
    min_capacity_wh=60, max_capacity_wh=100,
    min_capacity_hub=10, max_capacity_hub=14,
    min_capacity_pvz=60, max_capacity_pvz=90,
    warehouse_link_prob=0.6, hub_link_prob=0.8,
    wh_hub_link_prob=0.9, hub_pvz_link_prob=0.9,
)

# ── Режим спроса: умеренная нагрузка + локальные горячие точки ────────────────
REGIME = dict(
    min_orders_per_step=100,
    max_orders_per_step=150,
    hot_frac=0.4,
    hot_multiplier=5.0,
)

# Умеренная нагрузка: перегрузки транзиентные, а не постоянные.
# Это позволяет модели учиться на ранних признаках (queue_delta, upstream_load),
# а не просто фиксировать уже случившийся коллапс.
TRAIN_REGIME = dict(
    min_orders_per_step=80,
    max_orders_per_step=140,
    hot_frac=0.3,
    hot_multiplier=3.0,
)
N_STEPS = 150
RISK_THRESHOLD = 0.4
# Верхняя граница адаптивной доли рероутинга; фактическая = min(pred_risk, fraction).
PROACTIVE_FRACTION = 0.35
# Вес P(overload) в стоимости рёбер графа (Layer 3).
RISK_WEIGHT = 3.0
EVAL_SEEDS = [1, 2, 3, 4, 5]
TRAIN_SEED = 100
TRAIN_STEPS = 800  # увеличено: компенсация за фильтрацию is_overload==0
TRAIN_CSV = "data/train_sim.csv"


def generate_training_data(base_graph):
    g = copy.deepcopy(base_graph)
    df, _, _ = simulate_with_dispatcher(
        g, dispatcher=None, use_reroute=False,
        n_steps=TRAIN_STEPS, seed=TRAIN_SEED, **TRAIN_REGIME,
    )
    df.to_csv(TRAIN_CSV, index=False)
    return TRAIN_CSV


def train_model(csv_path):
    data = CSVDataLoader(csv_path).load_data()
    X_train, X_test, y_train, y_test = prepare_simulation_data(data)
    model = lgb.LGBMClassifier(
        objective="binary", metric="binary_logloss",
        num_leaves=63, learning_rate=0.05, n_estimators=400,
        class_weight="balanced", random_state=42, verbose=1,
    )
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"[train] overload rate: {y_train.mean():.3f} | holdout ROC-AUC: {auc:.3f}")
    return model, X_train


def train_rule_tree(model, X_train, max_depth=6, min_samples_leaf=30):
    """
    Decision Tree обучается на предсказаниях LightGBM -> человекочитаемые правила,
    которые затем исполняет диспетчер.
    """
    lgbm_labels = (model.predict_proba(X_train)[:, 1] > RISK_THRESHOLD).astype(int)
    dt = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42
    )
    dt.fit(X_train, lgbm_labels)
    fidelity = float((dt.predict(X_train) == lgbm_labels).mean())
    rules = export_text(dt, feature_names=list(X_train.columns))
    return dt, rules, fidelity


def run_mode(base_graph, seed, mode, model=None, rule_tree=None, explainer=None):
    g = copy.deepcopy(base_graph)
    dispatcher = None
    use_reroute = True
    if mode == "passive":
        use_reroute = False
    elif mode == "ai":
        dispatcher = AIDispatcher(
            model, rule_tree=rule_tree, risk_threshold=RISK_THRESHOLD, explainer=explainer
        )

    df, orders, actions = simulate_with_dispatcher(
        g, dispatcher=dispatcher, use_reroute=use_reroute,
        proactive_reroute_fraction=PROACTIVE_FRACTION,
        risk_weight=RISK_WEIGHT,
        n_steps=N_STEPS, seed=seed, **REGIME,
    )
    delivered = [o for o in orders if o.delivered]
    mean_queue_wait = (
        float(np.mean([o.queue_wait_time for o in delivered]))
        if delivered else float("nan")
    )

    # Показывает долю заказов без критических задержек.
    # В перегружённой сети заказы доедут, но намного позже прогноза.
    on_time = [
        o for o in delivered
        if o.initial_eta > 0
        and (o.delivered_at - o.created_at) <= o.initial_eta * 2
    ]
    on_time_rate = float(len(on_time) / max(len(orders), 1))

    metrics = {
        "overload_rate": float(df["is_overload"].mean()),
        "mean_load_ratio": float(df["load_ratio"].mean()),
        "max_load_ratio": float(df["load_ratio"].max()),
        "delivery_rate": float(df["delivered_orders"].max() / max(len(orders), 1)),
        "on_time_rate": on_time_rate,
        "mean_queue_wait": mean_queue_wait,
    }
    return metrics, actions, df, dispatcher


def main():
    random.seed(0)
    np.random.seed(0)

    base_graph = build_network(**NETWORK)

    print("[1/3] Генерация обучающих данных (пассивный режим)...")
    csv_path = generate_training_data(base_graph)

    print("[2/3] Обучение LightGBM + Decision Tree (исполняемые правила)...")
    model, X_train = train_model(csv_path)
    rule_tree, rules, fidelity = train_rule_tree(model, X_train)
    print(f"[rules] Decision Tree fidelity к LightGBM: {fidelity:.3f} "
          f"(насколько точно дерево воспроизводит решения LightGBM)")

    try:
        import shap
        import matplotlib.pyplot as plt
        explainer = shap.TreeExplainer(model)

        # Глобальный SHAP summary_plot по обучающей выборке.
        # Показывает реальный вклад каждого признака: если growth_rate / upstream_load
        # поднялись вверх относительно load_ratio — модель больше не вырождается в Дейкстру.
        sample = X_train.sample(min(2000, len(X_train)), random_state=42)
        shap_vals = np.asarray(explainer.shap_values(sample))
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[1]
        plt.figure()
        shap.summary_plot(shap_vals, sample, show=False, plot_type="bar")
        plt.tight_layout()
        plt.savefig("output_shap_global.png", dpi=120)
        plt.close()
        print("[shap] Глобальный график сохранён → output_shap_global.png")
    except Exception as exc:
        print(f"[warn] SHAP недоступен ({exc}); объяснения пропущены")
        explainer = None

    print(f"[3/3] Сравнение passive / reactive / ai на {len(EVAL_SEEDS)} сценариях...")
    rows = {"passive": [], "reactive": [], "ai": []}
    total_actions = {}
    last_ai_df, last_dispatcher = None, None

    for seed in EVAL_SEEDS:
        for mode in ("passive", "reactive", "ai"):
            m, actions, df, disp = run_mode(
                base_graph, seed, mode, model, rule_tree, explainer
            )
            rows[mode].append(m)
            if mode == "ai":
                for k, v in actions.items():
                    total_actions[k] = total_actions.get(k, 0) + v
                last_ai_df, last_dispatcher = df, disp

    avg = {mode: pd.DataFrame(r).mean() for mode, r in rows.items()}
    comp = pd.DataFrame(avg)
    comp["AI vs passive, %"] = (avg["passive"] - avg["ai"]) / avg["passive"].abs() * 100
    comp["AI vs reactive, %"] = (avg["reactive"] - avg["ai"]) / avg["reactive"].abs() * 100
    # для delivery_rate "улучшение" — это рост
    for col in ("AI vs passive, %", "AI vs reactive, %"):
        comp.loc["delivery_rate", col] *= -1

    SEP = "─" * 62
    print(f"\n{SEP}")
    print("  РЕЗУЛЬТАТЫ СИМУЛЯЦИИ")
    print(f"  Усреднено по {len(EVAL_SEEDS)} сценариям, "
          f"{N_STEPS} шагов каждый")
    print(SEP)
    print()
    print("  Режимы сравнения:")
    print("    passive  — нет рероутинга; заказы идут только по")
    print("               начальному пути (нижняя граница качества)")
    print("    reactive — рероутинг при ФАКТЕ пробки (классическая")
    print("               Дейкстра с динамическими весами)")
    print("    ai       — рероутинг по ПРЕДСКАЗАНИЮ LightGBM:")
    print("               видит тренды, upstream-давление, историю")
    print()
    print("  Метрики:")
    print("    overload_rate   — доля шагов с перегрузкой хотя бы")
    print("                      одного узла  (↓ лучше)")
    print("    mean_load_ratio — средняя загрузка по всем узлам")
    print("                      (↓ лучше, норма < 1.0)")
    print("    max_load_ratio  — пиковая загрузка самого нагру-")
    print("                      женного узла  (↓ лучше)")
    print("    delivery_rate   — доля доставленных заказов за")
    print("                      горизонт симуляции  (↑ лучше)")
    print("    on_time_rate    — доля заказов доставленных вовремя")
    print("                      (факт ≤ initial_eta × 2)  (↑ лучше)")
    print("    mean_queue_wait — среднее время ожидания в очередях,")
    print("                      шагов на доставленный заказ  (↓ лучше)")
    print()
    print(comp.round(4).to_string())
    print()
    print(SEP)
    print("  ЧТО ДЕЛАЛ AI-ДИСПЕТЧЕР (суммарно)")
    print(SEP)
    action_desc = {
        "reroute":           "обходов перегруженных / рискованных узлов",
        "proactive_reroute": "из них — до пробки, по предсказанию модели",
        "pvz_switch":        "смен пункта выдачи на менее рискованный",
        "priority_boost":    "приоритетных обработок заказов",
    }
    for k, v in total_actions.items():
        desc = action_desc.get(k, "")
        print(f"  {k:20s}: {v:6d}   {desc}")

    FEAT_DESC = {
        "load_ratio":       "текущая загрузка узла (очередь / ёмкость)",
        "load_ratio_lag1":  "загрузка 1 шаг назад",
        "load_ratio_lag2":  "загрузка 2 шага назад",
        "load_ratio_lag3":  "загрузка 3 шага назад (тренд)",
        "queue":            "длина очереди сейчас",
        "queue_delta":      "изменение очереди за шаг",
        "growth_rate":      "скорость роста очереди / ёмкость",
        "delta2":           "ускорение роста очереди",
        "upstream_load":    "средняя загрузка входящих узлов",
        "downstream_load":  "средняя загрузка исходящих узлов",
        "capacity":         "пропускная способность узла",
        "active_orders":    "заказов в сети прямо сейчас",
        "delivered_orders": "уже доставлено заказов",
        "stuck_orders":     "застрявших заказов",
        "node_type_code":   "тип узла (склад / хаб / ПВЗ)",
    }

    if explainer is not None and last_ai_df is not None:
        last_step = last_ai_df[last_ai_df["time"] == last_ai_df["time"].max()]
        hottest = last_step.loc[last_step["predicted_risk"].idxmax()]
        node_id = hottest["node_id"]
        print(f"\n{SEP}")
        print(f"  SHAP: ОБЪЯСНЕНИЕ РИСКА — узел {node_id}")
        print(f"  P(overload) = {hottest['predicted_risk']:.2f}  "
              f"(порог срабатывания: {RISK_THRESHOLD})")
        print(SEP)
        print("  Почему модель считает этот узел рискованным:")
        print()
        explanation = last_dispatcher.explain(node_id, top_k=len(FEATURE_COLUMNS))
        for feat, val in explanation:
            if abs(val) < 0.001:
                continue
            arrow = "↑ повышает риск" if val > 0 else "↓ снижает риск"
            desc = FEAT_DESC.get(feat, "")
            print(f"  {feat:20s} {val:+.3f}  {arrow}"
                  f"   [{desc}]")

        try:
            import matplotlib.pyplot as plt
            feats = [f for f, _ in explanation]
            vals = [v for _, v in explanation]
            colors = ["#e63946" if v > 0 else "#457b9d" for v in vals]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(feats[::-1], vals[::-1], color=colors[::-1])
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(
                f"SHAP: узел {node_id}  "
                f"(P(overload)={hottest['predicted_risk']:.2f})"
            )
            ax.set_xlabel("SHAP value")
            plt.tight_layout()
            plt.savefig("output_shap_node.png", dpi=120)
            plt.close()
            print("[shap] График узла сохранён → output_shap_node.png")
        except Exception:
            pass


if __name__ == "__main__":
    main()
