import networkx as nx
import random
import json
from dataclasses import dataclass, field
from typing import List, Optional
import random
import pandas as pd

INF = float('inf')

@dataclass
class Order:
    order_id: int
    start_point: str
    end_point: str
    path: List[str]
    delivered: bool = False
    stuck: bool = False
    path_index: int = 0
    reroute_count: int = 0
    priority: int = 0
    queue_wait_time: int = 0
    created_at: int = 0
    delivered_at: int = -1
    initial_eta: float = 0.0

    @property
    def current_node(self):
        return self.path[self.path_index]

    def next_node(self):
        if self.path_index + 1 < len(self.path):
            return self.path[self.path_index + 1]
        return None

    def change_path(self, new_path):
        self.path = new_path
        self.path_index = 0
        self.reroute_count += 1


def build_network(n_nodes=31, n_warehouses=4, n_hubs=7, n_pvz=20, 
                  min_capacity_wh=100, max_capacity_wh=200,
                  min_capacity_hub=50, max_capacity_hub=100,
                  min_capacity_pvz=10, max_capacity_pvz=50,
                  min_base_time=3, max_base_time=10,
                  warehouse_link_prob=0.5,
                  hub_link_prob=0.6,
                  wh_hub_link_prob=0.5,
                  hub_pvz_link_prob=0.5):
    nodes = []
    edges = []
    G = nx.DiGraph()
    if n_nodes != n_warehouses + n_hubs + n_pvz:
        raise ValueError("Количество вершин не равно сумме складов, хабов и пвз")
    for i in range(n_warehouses):
        nodes.append((f"wh{i}", "warehouse", random.randint(min_capacity_wh, max_capacity_wh)))
    for i in range(n_hubs):
        nodes.append((f"hub{i}", "hub", random.randint(min_capacity_hub, max_capacity_hub)))
    for i in range(n_pvz):
        nodes.append((f"pvz{i}", "pvz", random.randint(min_capacity_pvz, max_capacity_pvz)))
    for node_id, node_type, cap in nodes:
        G.add_node(node_id, type=node_type, capacity=cap, queue=[], load_ratio=0.0, is_overload=False)
    whs = nodes[:n_warehouses]
    hubs = nodes[n_warehouses:n_warehouses + n_hubs]
    pvzs = nodes[n_warehouses + n_hubs:]
    # added_nodes = set()
    from_nodes = set()
    to_nodes = set()
    wh_out_to_hub = set()
    hub_in_from_wh = set()
    hub_out_to_pvz = set()
    pvz_in_from_hub = set()

    for wh_from in whs:
        for wh_to in whs:
            if wh_from == wh_to:
                continue
            if random.random() < warehouse_link_prob:
                edges.append((wh_from[0], wh_to[0], random.randint(min_base_time, max_base_time)))
                from_nodes.add(wh_from)
                to_nodes.add(wh_to)

    for wh in whs:
        for hub in hubs:
            if random.random() < wh_hub_link_prob:
                edges.append((wh[0], hub[0], random.randint(min_base_time, max_base_time)))
                from_nodes.add(wh)
                to_nodes.add(hub)
                wh_out_to_hub.add(wh)
                hub_in_from_wh.add(hub)

    for hub_from in hubs:
        for hub_to in hubs:
            if hub_from == hub_to:
                continue
            if random.random() < hub_link_prob:
                edges.append((hub_from[0], hub_to[0], random.randint(min_base_time, max_base_time)))
                from_nodes.add(hub_from)
                to_nodes.add(hub_to)

    for hub in hubs:
        for pvz in pvzs:
            if random.random() < hub_pvz_link_prob:
                edges.append((hub[0], pvz[0], random.randint(min_base_time, max_base_time)))
                from_nodes.add(hub)
                to_nodes.add(pvz)
                hub_out_to_pvz.add(hub)
                pvz_in_from_hub.add(pvz)
    for wh in whs:
        if wh not in wh_out_to_hub:
            sel_hub = random.choice(hubs)
            edges.append((wh[0], sel_hub[0], random.randint(min_base_time, max_base_time)))
            from_nodes.add(wh)
            to_nodes.add(sel_hub)
            wh_out_to_hub.add(wh)
            hub_in_from_wh.add(sel_hub)
    for hub in hubs:
        if hub not in hub_out_to_pvz:
            sel_pvz = random.choice(pvzs)
            edges.append((hub[0], sel_pvz[0], random.randint(min_base_time, max_base_time)))
            from_nodes.add(hub)
            to_nodes.add(sel_pvz)
            hub_out_to_pvz.add(hub)
            pvz_in_from_hub.add(sel_pvz)
        if hub not in hub_in_from_wh:
            sel_wh = random.choice(whs)
            edges.append((sel_wh[0], hub[0], random.randint(min_base_time, max_base_time)))
            from_nodes.add(sel_wh)
            to_nodes.add(hub)
            wh_out_to_hub.add(sel_wh)
            hub_in_from_wh.add(hub)
    for pvz in pvzs:
        if pvz not in pvz_in_from_hub:
            sel_hub = random.choice(hubs)
            edges.append((sel_hub[0], pvz[0], random.randint(min_base_time, max_base_time)))
            from_nodes.add(sel_hub)
            to_nodes.add(pvz)
            hub_out_to_pvz.add(sel_hub)
            pvz_in_from_hub.add(pvz)
    
    for src, dst, bt in edges:
        G.add_edge(src, dst, base_time=bt)
    return G


def cost_function(graph, start_node, end_node):
    capacity = graph.nodes[end_node]["capacity"]
    queue = len(graph.nodes[end_node]["queue"])
    base_time = graph[start_node][end_node]["base_time"]

    load_ratio = queue / capacity if capacity > 0 else INF
    risk_weight = graph.graph.get("risk_weight", 0.0)
    pred_risk = graph.nodes[end_node].get("pred_risk", 0.0)

    if risk_weight > 0.0:
        # AI-режим: congestion = P(overload) × risk_weight.
        # pred_risk уже содержит load_ratio_lag, upstream_load, delta2 —
        # Дейкстра обходит узлы с нарастающим трендом, а не только уже перегруженные.
        congestion = risk_weight * pred_risk
    else:
        # Базовый/реактивный режим: штраф по текущей загрузке.
        congestion = load_ratio * 3.0

    return base_time + congestion

def find_path(graph, start_point, end_point, start_search=True, excluded_node=None):
    if start_search:
        try:
            path = nx.dijkstra_path(graph, start_point, end_point, weight="base_time")
            return path
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            return None
    try:
        G_temp = graph.copy()

        if excluded_node is not None and excluded_node not in {start_point, end_point}:
            if excluded_node in G_temp:
                G_temp.remove_node(excluded_node)
        path = nx.dijkstra_path(
            G_temp, start_point, end_point,
            weight=lambda start, end, _: cost_function(G_temp, start, end)
        )
        return path
    except nx.NetworkXNoPath:
        return None
    except nx.NodeNotFound:
        return None


def update_node_loads(graph):
    for node in graph.nodes:
        capacity = graph.nodes[node]["capacity"]
        queue = len(graph.nodes[node]["queue"])

        load_ratio = queue / capacity if capacity > 0 else INF
        is_overload = queue > capacity

        graph.nodes[node]["load_ratio"] = load_ratio
        graph.nodes[node]["is_overload"] = is_overload

def get_nodes_by_type(graph, node_type):
    return [
        node
        for node, attrs in graph.nodes(data=True)
        if attrs["type"] == node_type
    ]

def simulate(
    graph,
    n_steps=200,
    min_orders_per_step=5,
    max_orders_per_step=20,
    reroute_threshold=1.0,
    block_if_full=False
):
    orders = []
    history = []
    order_id = 0

    whs = get_nodes_by_type(graph, "warehouse")
    pvzs = get_nodes_by_type(graph, "pvz")

    update_node_loads(graph)

    for step in range(n_steps):

        n_new_orders = random.randint(min_orders_per_step, max_orders_per_step)

        for _ in range(n_new_orders):
            start = random.choice(whs)
            end = random.choice(pvzs)

            path = find_path(
                graph,
                start,
                end,
                start_search=True
            )

            while path is None:
                start = random.choice(whs)
                end = random.choice(pvzs)
                path = find_path(
                    graph,
                    start,
                    end,
                    start_search=True
                )

            new_order = Order(
                order_id=order_id,
                start_point=start,
                end_point=end,
                path=path
            )

            orders.append(new_order)
            order_id += 1

            graph.nodes[start]["queue"].append(new_order)

        update_node_loads(graph)

        for node in graph.nodes:
            for order in graph.nodes[node]["queue"]:
                if not order.delivered and not order.stuck:
                    order.queue_wait_time += 1
        incoming = {node: [] for node in graph.nodes}

        for node in list(graph.nodes):
            node_queue = graph.nodes[node]["queue"]
            capacity = graph.nodes[node]["capacity"]

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

                next_node = order.next_node()

                if next_node is None:
                    order.stuck = True
                    continue

                next_capacity = graph.nodes[next_node]["capacity"]
                next_queue_len = len(graph.nodes[next_node]["queue"]) + len(incoming[next_node])
                next_load_ratio = next_queue_len / next_capacity if next_capacity > 0 else INF

                if next_load_ratio > reroute_threshold:
                    new_path = find_path(
                        graph,
                        start_point=current_node,
                        end_point=order.end_point,
                        start_search=False,
                        excluded_node=next_node
                    )

                    if new_path is not None:
                        order.change_path(new_path)
                        next_node = order.next_node()
                    else:
                        incoming[current_node].append(order)
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

        active_orders = sum(1 for o in orders if not o.delivered and not o.stuck)
        delivered_orders = sum(1 for o in orders if o.delivered)
        stuck_orders = sum(1 for o in orders if o.stuck)

        for node, attrs in graph.nodes(data=True):
            queue_len = len(attrs["queue"])

            history.append({
                "time": step,
                "node_id": node,
                "node_type": attrs["type"],
                "capacity": attrs["capacity"],
                "queue": queue_len,
                "load_ratio": attrs["load_ratio"],
                "is_overload": int(attrs["is_overload"]),
                "active_orders": active_orders,
                "delivered_orders": delivered_orders,
                "stuck_orders": stuck_orders,
            })

    return history, orders


if __name__ == "__main__":
    G = build_network(
        n_nodes=31,
        n_warehouses=4,
        n_hubs=7,
        n_pvz=20,
        min_capacity_wh=20,
        max_capacity_wh=40,
        min_capacity_hub=10,
        max_capacity_hub=25,
        min_capacity_pvz=5,
        max_capacity_pvz=15,
    )

    history, orders = simulate(
        G,
        n_steps=1000,
        min_orders_per_step=70,
        max_orders_per_step=130,
        reroute_threshold=1.0,
        block_if_full=False
    )

    df = pd.DataFrame(history)

    OUTPUT_PATH = "data.csv"

    print(df["is_overload"].value_counts())
    print(df["is_overload"].mean())

    df.to_csv(OUTPUT_PATH, index=False)
