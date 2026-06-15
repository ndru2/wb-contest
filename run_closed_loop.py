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
import random

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text

from data.simulation import build_network
from src.data_handlers.prepare_data import prepare_simulation_data
from src.data_loaders.loaders import CSVDataLoader
from src.dispatcher.ai_dispatcher import AIDispatcher
from src.dispatcher.closed_loop import simulate_with_dispatcher

# ── Сеть: узкое место на хабах, просторные склады/ПВЗ, плотные связи ──────────
NETWORK = dict(
    n_nodes=39, n_warehouses=5, n_hubs=14, n_pvz=20,
    min_capacity_wh=60, max_capacity_wh=100,
    min_capacity_hub=10, max_capacity_hub=14,
    min_capacity_pvz=60, max_capacity_pvz=90,
    warehouse_link_prob=0.6, hub_link_prob=0.8,
    wh_hub_link_prob=0.9, hub_pvz_link_prob=0.9,
)

# ── Режим спроса: умеренная нагрузка + локальные горячие точки ────────────────
REGIME = dict(
    min_orders_per_step=50,
    max_orders_per_step=90,
    hot_frac=0.3,
    hot_multiplier=3.0,
)

# Отдельный, более тяжёлый режим ТОЛЬКО для генерации обучающих данных:
TRAIN_REGIME = dict(
    min_orders_per_step=60,
    max_orders_per_step=110,
    hot_frac=0.4,
    hot_multiplier=4.5,
)
N_STEPS = 150
RISK_THRESHOLD = 0.4
PROACTIVE_FRACTION = 0.3   # доля рискового потока
RISK_WEIGHT = 10.0         # вес прогноза P(overload) в стоимости рёбер графа (Layer 3)
EVAL_SEEDS = [1, 2, 3, 4, 5]
TRAIN_SEED = 100
TRAIN_STEPS = 500
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
        class_weight="balanced", random_state=42, verbose=-1,
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
    metrics = {
        "overload_rate": float(df["is_overload"].mean()),
        "mean_load_ratio": float(df["load_ratio"].mean()),
        "max_load_ratio": float(df["load_ratio"].max()),
        "delivery_rate": float(df["delivered_orders"].max() / max(len(orders), 1)),
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
    print(f"[rules] Decision Tree fidelity к LightGBM: {fidelity:.3f}")
    print("\n=== Исполняемые правила диспетчера (Decision Tree поверх LightGBM) ===")
    print(rules)

    try:
        import shap
        explainer = shap.TreeExplainer(model)
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

    print("\n=== Сравнение режимов (среднее по сценариям) ===")
    print(comp.round(4).to_string())

    print("\n=== Действия AI-диспетчера (суммарно по сценариям) ===")
    for k, v in total_actions.items():
        print(f"  {k:18s}: {v}")

    if explainer is not None and last_ai_df is not None:
        last_step = last_ai_df[last_ai_df["time"] == last_ai_df["time"].max()]
        hottest = last_step.loc[last_step["predicted_risk"].idxmax()]
        node_id = hottest["node_id"]
        print(f"\n=== SHAP: почему узел {node_id} в зоне риска "
              f"(P(overload)={hottest['predicted_risk']:.2f}) ===")
        for feat, val in last_dispatcher.explain(node_id, top_k=4):
            direction = "повышает" if val > 0 else "снижает"
            print(f"  {feat:18s}: {val:+.3f}  ({direction} риск)")


if __name__ == "__main__":
    main()
