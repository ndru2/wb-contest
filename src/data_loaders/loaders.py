from abc import ABC, abstractmethod
import pandas as pd
import networkx as nx


NODE_TYPE_CODE = {"warehouse": 0, "hub": 1, "pvz": 2}

FEATURE_COLUMNS = [
    "node_type_code",
    "capacity",
    "queue",
    "load_ratio",
    # "congestion" удалён: load_ratio * 3.0 — линейный дубль load_ratio, LightGBM не получает
    # новой информации, а в SHAP он поглощал вес load_ratio, искажая интерпретацию.
    "active_orders",
    "delivered_orders",
    "stuck_orders",
    "load_ratio_lag1",
    "load_ratio_lag2",
    "load_ratio_lag3",
    "queue_delta",
    # Скорость роста очереди относительно ёмкости: позволяет модели видеть нарастающий
    # тренд на ещё незагруженных узлах (то, чего не видит Дейкстра по текущему load_ratio).
    "growth_rate",
    # Ускорение роста очереди (вторая производная): резкий скачок delta говорит о всплеске
    # входящего потока за последние два шага.
    "delta2",
    # Загрузка соседей: давление «сверху» (склады → хабы) и «снизу» (хабы → ПВЗ).
    "upstream_load",
    "downstream_load",
]

TARGET_COLUMN = "future_overload"


class DataLoader(ABC):
    @abstractmethod
    def load_data(self):
        pass


class XLSXDataLoader(DataLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        df = pd.read_excel(self.file_path)
        df = df.drop(['Unnamed: 0', 'path_indices', 'leaf_index'], axis=1).copy()
        return df


class CSVDataLoader(DataLoader):
    def __init__(self, file_path: str, forecast_horizon: int = 5):
        self.file_path = file_path
        self.forecast_horizon = forecast_horizon

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.file_path)

        df["node_type_code"] = df["node_type"].map(NODE_TYPE_CODE).fillna(-1).astype(int)

        df = df.sort_values(["node_id", "time"]).reset_index(drop=True)

        grp = df.groupby("node_id")
        df["load_ratio_lag1"] = grp["load_ratio"].shift(1)
        df["load_ratio_lag2"] = grp["load_ratio"].shift(2)
        df["load_ratio_lag3"] = grp["load_ratio"].shift(3)
        df["queue_delta"]     = df["queue"] - grp["queue"].shift(1)

        # Скорость роста очереди нормированная на ёмкость: ненулевая даже при низком load_ratio,
        # если очередь резко растёт. Это главный "ранний сигнал" для модели.
        df["growth_rate"] = df["queue_delta"] / df["capacity"].clip(lower=1)

        # Ускорение: разность двух последовательных queue_delta.
        # Резкий положительный скачок → всплеск входящего потока.
        df["delta2"] = df["queue_delta"] - grp["queue_delta"].shift(1)

        # Загрузка соседей: записывается симуляцией в CSV.
        # Фоллбэк 0.0 для совместимости со старыми CSV-файлами без этих колонок.
        for col in ("upstream_load", "downstream_load"):
            if col not in df.columns:
                df[col] = 0.0

        df[TARGET_COLUMN] = grp["is_overload"].shift(-self.forecast_horizon)

        df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

        # Обучаем только на незагруженных состояниях: модель учится предсказывать
        # ПЕРВОЕ наступление перегрузки по трендам, а не фиксировать уже существующий коллапс.
        # Без этого фильтра load_ratio доминирует в SHAP и модель вырождается в Дейкстру.
        df = df[df["is_overload"] == 0].copy()

        return df[["node_id", "time"] + FEATURE_COLUMNS + [TARGET_COLUMN]]


class GraphDataLoader(DataLoader):
    def __init__(
        self,
        graph: nx.DiGraph,
        step: int,
        active_orders: int = 0,
        delivered_orders: int = 0,
        stuck_orders: int = 0,
        history: dict = None,
    ):
        self.graph = graph
        self.step = step
        self.active_orders = active_orders
        self.delivered_orders = delivered_orders
        self.stuck_orders = stuck_orders
        self.history = history or {}

    def load_data(self) -> pd.DataFrame:
        rows = []
        for node_id, attrs in self.graph.nodes(data=True):
            capacity   = attrs.get("capacity", 1)
            queue      = len(attrs.get("queue", []))
            load_ratio = queue / capacity if capacity > 0 else 0.0

            past = self.history.get(node_id, [])
            lr_lag1 = past[-1]["load_ratio"] if len(past) >= 1 else 0.0
            lr_lag2 = past[-2]["load_ratio"] if len(past) >= 2 else 0.0
            lr_lag3 = past[-3]["load_ratio"] if len(past) >= 3 else 0.0

            q_prev       = past[-1]["queue"] if len(past) >= 1 else queue
            q_prev2      = past[-2]["queue"] if len(past) >= 2 else q_prev
            queue_delta  = queue - q_prev
            queue_delta_prev = q_prev - q_prev2
            growth_rate  = queue_delta / max(capacity, 1)
            delta2       = queue_delta - queue_delta_prev

            pred_lr = [self.graph.nodes[p]["load_ratio"] for p in self.graph.predecessors(node_id)]
            succ_lr = [self.graph.nodes[s]["load_ratio"] for s in self.graph.successors(node_id)]
            upstream_load   = sum(pred_lr) / len(pred_lr) if pred_lr else 0.0
            downstream_load = sum(succ_lr) / len(succ_lr) if succ_lr else 0.0

            rows.append({
                "node_id":          node_id,
                "node_type_code":   NODE_TYPE_CODE.get(attrs.get("type", ""), -1),
                "capacity":         capacity,
                "queue":            queue,
                "load_ratio":       load_ratio,
                "active_orders":    self.active_orders,
                "delivered_orders": self.delivered_orders,
                "stuck_orders":     self.stuck_orders,
                "load_ratio_lag1":  lr_lag1,
                "load_ratio_lag2":  lr_lag2,
                "load_ratio_lag3":  lr_lag3,
                "queue_delta":      queue_delta,
                "growth_rate":      growth_rate,
                "delta2":           delta2,
                "upstream_load":    upstream_load,
                "downstream_load":  downstream_load,
            })

        return pd.DataFrame(rows)
