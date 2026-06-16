"""
Замкнутый управляющий контур: симуляция <-> AI-диспетчер.

Цикл одного шага:
    1. Симуляция генерирует новые заказы.
    2. LightGBM предсказывает P(overload) по узлам (через AIDispatcher).
    3. (SHAP объясняет причины — доступно через dispatcher.explain).
    4. Диспетчер применяет правила:
         - reroute        : обойти узел, который предсказан перегруженным (а не уже стоящий в пробке);
         - смена ПВЗ      : если целевой ПВЗ предсказан перегруженным, перенаправить на менее рискованный;
         - приоритет      : заказы, идущие в рискованные узлы, обрабатываются первыми.
    5. Симуляция обновляется, состояние пишется в историю.

Если dispatcher=None — получаем baseline (reroute только по факту пробки),
что эквивалентно исходной simulate() и служит точкой сравнения.
"""

import random

import pandas as pd

from data.simulation import (
    INF,
    Order,
    find_path,
    get_nodes_by_type,
    update_node_loads,
)


def _pick_alt_pvz(graph, current_node, pvzs, risk, exclude):
    base_risk = risk.get(exclude, 0.0)
    ranked = sorted((p for p in pvzs if p != exclude), key=lambda p: risk.get(p, 1.0))
    for p in ranked:
        if risk.get(p, 1.0) >= base_risk:
            break
        if find_path(graph, current_node, p, start_search=True) is not None:
            return p
    return None


def simulate_with_dispatcher(
    graph,
    dispatcher=None,
    n_steps=400,
    min_orders_per_step=15,
    max_orders_per_step=30,
    reroute_threshold=1.0,
    use_reroute=True,
    use_pvz_switch=True,
    use_priority=True,
    max_pvz_switches=5,
    block_if_full=False,
    proactive_reroute_fraction=0.4,
    risk_weight=0.0,
    hot_frac=0.0,
    hot_multiplier=1.0,
    order_log=None,
    order_log_cap=150,
    seed=None,
):
    # rng — поток заказов (одинаков во всех режимах при одном seed).
    # aux_rng — стохастика решений диспетчера и сэмплирование лога, чтобы решения
    # диспетчера не сдвигали поток заказов (честное сравнение passive/reactive/ai).
    base_seed = 0 if seed is None else seed
    rng = random.Random(base_seed)
    aux_rng = random.Random(base_seed + 10007)

    orders = []
    history = []
    order_id = 0

    whs = get_nodes_by_type(graph, "warehouse")
    pvzs = get_nodes_by_type(graph, "pvz")

    # Неравномерный спрос: фиксированное подмножество горячих ПВЗ получает
    # повышенный вес => локальные перегрузки при наличии запаса ёмкости в других узлах.
    pvz_weights = None
    if hot_frac > 0.0 and hot_multiplier > 1.0:
        n_hot = max(1, int(len(pvzs) * hot_frac))
        hot_set = set(rng.sample(pvzs, n_hot))
        pvz_weights = [hot_multiplier if p in hot_set else 1.0 for p in pvzs]

    def sample_destination():
        if pvz_weights is None:
            return rng.choice(pvzs)
        return rng.choices(pvzs, weights=pvz_weights, k=1)[0]

    # Вес прогноза в стоимости рёбер (Layer 3). Действует только при наличии
    # диспетчера: тогда в узлы пишется pred_risk и Дейкстра обходит будущие пробки.
    graph.graph["risk_weight"] = risk_weight if dispatcher is not None else 0.0

    update_node_loads(graph)
    if dispatcher is not None:
        dispatcher.reset(graph)

    actions_log = {"reroute": 0, "pvz_switch": 0, "priority_boost": 0, "proactive_reroute": 0}

    for step in range(n_steps):
        # 1. Новые заказы
        n_new_orders = rng.randint(min_orders_per_step, max_orders_per_step)
        for _ in range(n_new_orders):
            start = rng.choice(whs)
            end = sample_destination()
            path = find_path(graph, start, end, start_search=True)
            while path is None:
                start = rng.choice(whs)
                end = sample_destination()
                path = find_path(graph, start, end, start_search=True)
            new_order = Order(order_id=order_id, start_point=start, end_point=end, path=path)
            orders.append(new_order)
            order_id += 1
            graph.nodes[start]["queue"].append(new_order)

        update_node_loads(graph)

        for node in graph.nodes:
            for order in graph.nodes[node]["queue"]:
                if not order.delivered and not order.stuck:
                    order.queue_wait_time += 1

        active_orders = sum(1 for o in orders if not o.delivered and not o.stuck)
        delivered_orders = sum(1 for o in orders if o.delivered)
        stuck_orders = sum(1 for o in orders if o.stuck)

        # 2. AI предсказывает риски по узлам
        risk = {}
        if dispatcher is not None:
            risk = dispatcher.predict_risk(
                graph, step, active_orders, delivered_orders, stuck_orders
            )
            # Прогноз -> в веса графа: пишем pred_risk в узлы, чтобы объезд Дейкстра
            # по cost_function глобально обходил узлы с высоким будущим риском.
            for node_id, p in risk.items():
                graph.nodes[node_id]["pred_risk"] = p
            # 4c. Приоритет: пометить заказы, идущие в рискованные узлы
            if use_priority:
                for o in orders:
                    if o.delivered or o.stuck:
                        continue
                    nxt = o.next_node()
                    o.priority = int(nxt is not None and risk.get(nxt, 0.0) > dispatcher.risk_threshold)

        # Лог позиций заказов для визуализации
        if order_log is not None:
            active = [o for o in orders if not o.delivered and not o.stuck]
            if len(active) > order_log_cap:
                active = aux_rng.sample(active, order_log_cap)
            snapshot = []
            for o in active:
                nn = o.next_node()
                snapshot.append((o.current_node, nn if nn is not None else o.current_node))
            order_log.append(snapshot)

        incoming = {node: [] for node in graph.nodes}

        for node in list(graph.nodes):
            node_queue = graph.nodes[node]["queue"]
            capacity = graph.nodes[node]["capacity"]

            # 4c. Приоритет: высокоприоритетные заказы попадают в обработку первыми
            if dispatcher is not None and use_priority:
                node_queue = sorted(node_queue, key=lambda o: o.priority, reverse=True)
                if any(o.priority for o in node_queue[:capacity]):
                    actions_log["priority_boost"] += 1

            orders_to_process = node_queue[:capacity]
            graph.nodes[node]["queue"] = node_queue[capacity:]

            for order in orders_to_process:
                if order.delivered or order.stuck:
                    continue

                current_node = order.current_node
                if current_node != node:
                    continue

                if current_node == order.end_point:
                    order.delivered = True
                    continue

                # 4b. Смена ПВЗ: если целевой ПВЗ предсказан перегруженным
                if (
                    dispatcher is not None
                    and use_pvz_switch
                    and order.reroute_count < max_pvz_switches
                    and risk.get(order.end_point, 0.0) > dispatcher.risk_threshold
                ):
                    alt = _pick_alt_pvz(graph, current_node, pvzs, risk, exclude=order.end_point)
                    if alt is not None:
                        new_path = find_path(graph, current_node, alt, start_search=False)
                        if new_path is not None:
                            order.end_point = alt
                            order.change_path(new_path)
                            actions_log["pvz_switch"] += 1

                next_node = order.next_node()
                if next_node is None:
                    order.stuck = True
                    continue

                next_capacity = graph.nodes[next_node]["capacity"]
                next_queue_len = len(graph.nodes[next_node]["queue"]) + len(incoming[next_node])
                next_load_ratio = next_queue_len / next_capacity if next_capacity > 0 else INF

                # 4a. Reroute: проактивно (по предсказанию) или реактивно (по факту пробки).
                # Проактивно уводим лишь долю потока (proactive_reroute_fraction),
                # чтобы разгрузить узел, но не перегрузить альтернативу ("стадность").
                predicted_overload = (
                    dispatcher is not None
                    and risk.get(next_node, 0.0) > dispatcher.risk_threshold
                    and aux_rng.random() < proactive_reroute_fraction
                )
                congested = next_load_ratio > reroute_threshold

                if use_reroute and (predicted_overload or congested):
                    new_path = find_path(
                        graph,
                        start_point=current_node,
                        end_point=order.end_point,
                        start_search=False,
                        excluded_node=next_node,
                    )
                    if new_path is not None:
                        order.change_path(new_path)
                        next_node = order.next_node()
                        actions_log["reroute"] += 1
                        if predicted_overload and not congested:
                            actions_log["proactive_reroute"] += 1
                    else:
                        incoming[current_node].append(order)
                        continue

                if next_node is None:
                    order.stuck = True
                    continue

                if block_if_full:
                    current_next_queue = len(graph.nodes[next_node]["queue"]) + len(incoming[next_node])
                    if current_next_queue >= graph.nodes[next_node]["capacity"]:
                        incoming[current_node].append(order)
                        continue

                order.path_index += 1
                incoming[next_node].append(order)

        for node, arrived_orders in incoming.items():
            graph.nodes[node]["queue"].extend(arrived_orders)

        update_node_loads(graph)
        if dispatcher is not None:
            dispatcher.update_history(graph)

        active_orders = sum(1 for o in orders if not o.delivered and not o.stuck)
        delivered_orders = sum(1 for o in orders if o.delivered)
        stuck_orders = sum(1 for o in orders if o.stuck)

        for node, attrs in graph.nodes(data=True):
            history.append({
                "time": step,
                "node_id": node,
                "node_type": attrs["type"],
                "capacity": attrs["capacity"],
                "queue": len(attrs["queue"]),
                "load_ratio": attrs["load_ratio"],
                "is_overload": int(attrs["is_overload"]),
                "active_orders": active_orders,
                "delivered_orders": delivered_orders,
                "stuck_orders": stuck_orders,
                "predicted_risk": risk.get(node, float("nan")),
            })

    return pd.DataFrame(history), orders, actions_log
