"""
AI-диспетчер

Разделение ролей (как в спецификации):
  * LightGBM (model)      — точное предсказание риска P(overload), на нём строится SHAP;
  * Decision Tree (rule_tree) — правила: дерево обучено на предсказаниях
    LightGBM и именно его решение запускает действия диспетчера (объезд/смена ПВЗ/приоритет);
  * SHAP (explainer)      — объяснение причины риска.

На каждом шаге диспетчер:
  1. сериализует текущее состояние графа в признаки (GraphDataLoader),
  2. считает риск: LightGBM (для логов/SHAP) и Decision Tree (для решения),
  3. (опционально) объясняет причину риска через SHAP,
  4. обновляет rolling-буфер истории для лаговых признаков.

"""

from collections import deque

import numpy as np

from src.data_loaders.loaders import GraphDataLoader, FEATURE_COLUMNS


def _proba_positive(estimator, X):
    proba = estimator.predict_proba(X)
    classes = list(getattr(estimator, "classes_", [0, 1]))
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(proba.shape[0])


class AIDispatcher:
    def __init__(
        self,
        model,
        rule_tree=None,
        risk_threshold: float = 0.5,
        history_len: int = 3,
        explainer=None,
    ):
        self.model = model
        self.rule_tree = rule_tree
        self.risk_threshold = risk_threshold
        self.history_len = history_len
        self.explainer = explainer
        self.history = {}
        self._last_features = None
        self._last_lgbm_risk = {}

    def reset(self, graph):
        self.history = {node: deque(maxlen=self.history_len) for node in graph.nodes}
        self._last_features = None
        self._last_lgbm_risk = {}

    def _history_as_lists(self):
        return {node: list(buf) for node, buf in self.history.items()}

    def predict_risk(self, graph, step, active_orders, delivered_orders, stuck_orders):
        loader = GraphDataLoader(
            graph=graph,
            step=step,
            active_orders=active_orders,
            delivered_orders=delivered_orders,
            stuck_orders=stuck_orders,
            history=self._history_as_lists(),
        )
        df = loader.load_data()
        X = df[FEATURE_COLUMNS]

        lgbm_proba = _proba_positive(self.model, X)
        self._last_lgbm_risk = dict(zip(df["node_id"], lgbm_proba))
        self._last_features = df.set_index("node_id")

        if self.rule_tree is not None:
            decision_proba = _proba_positive(self.rule_tree, X)
        else:
            decision_proba = lgbm_proba

        return dict(zip(df["node_id"], decision_proba))

    def update_history(self, graph):
        for node, attrs in graph.nodes(data=True):
            capacity = attrs.get("capacity", 1)
            queue = len(attrs.get("queue", []))
            load_ratio = queue / capacity if capacity > 0 else 0.0
            if node not in self.history:
                self.history[node] = deque(maxlen=self.history_len)
            self.history[node].append({"queue": queue, "load_ratio": load_ratio})

    def explain(self, node_id, top_k: int = 3):
        if self.explainer is None or self._last_features is None:
            return []
        if node_id not in self._last_features.index:
            return []
        row = self._last_features.loc[node_id, FEATURE_COLUMNS].to_numpy().reshape(1, -1)
        shap_values = np.asarray(self.explainer.shap_values(row)).reshape(-1)
        order = np.argsort(np.abs(shap_values))[::-1][:top_k]
        return [(FEATURE_COLUMNS[i], float(shap_values[i])) for i in order]
