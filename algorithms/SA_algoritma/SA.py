import sys
import os
import math
import random
from typing import Optional, Tuple

import networkx as nx

# ==================================================
# ROOT projesini sys.path’e ekle
# ==================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# ==================================================
# topology ice aktarma
# ==================================================
from data.network_topology import (
    generate_connected_topology,
    assign_node_attributes,
    assign_edge_attributes
)

# ==================================================
# Import METRICS 
# ==================================================
from metrics.metric import MetricsEngine, Weights


# ==================================================
# ADAPTER: attribute graph’u metric.py ile eşleştir
# ==================================================
def adapt_graph_for_metrics(G: nx.Graph) -> nx.Graph:
    # ---------- NODE ----------
    for n, a in G.nodes(data=True):
        a.setdefault("processing_delay_ms",
                     a.get("processing_delay", a.get("proc_delay", 0.0)))
        a.setdefault("node_reliability",
                     a.get("node_reliability", a.get("reliability", 0.99)))

    # ---------- EDGE ----------
    for u, v, a in G.edges(data=True):
        a.setdefault("link_delay_ms",
                     a.get("link_delay", a.get("link_delay_ms", 1.0)))
        a.setdefault("capacity_mbps",
                     a.get("bandwidth_mbps", a.get("bandwidth", 100.0)))
        a.setdefault("link_reliability",
                     a.get("link_reliability", a.get("reliability", 0.99)))

    return G


# ==================================================
# SIMULATED ANNEALING
# ==================================================
def baslangic_yolu(G, s, d):
    return nx.shortest_path(G, s, d, weight="link_delay_ms")


def komsu_yol(G, yol, s, d):
    if len(yol) <= 2:
        return yol[:]

    idx = list(range(1, len(yol) - 1))
    random.shuffle(idx)

    for i in idx:
        try:
            yeni = nx.shortest_path(G, yol[i], d, weight="link_delay_ms")
            return yol[:i] + yeni
        except nx.NetworkXNoPath:
            continue

    return yol[:]


def simulated_annealing(
    G, s, d,
    *,
    weights: Weights,
    demand_mbps: Optional[float] = None,
    T0=5.0,
    alpha=0.995,
    max_iter=5000
) -> Tuple[list, float, object]:

    engine = MetricsEngine(G)

    cur = baslangic_yolu(G, s, d)
    cur_m = engine.compute(cur, demand_mbps=demand_mbps)
    cur_score = engine.weighted_sum(cur_m, weights)

    best = cur[:]
    best_score = cur_score
    best_m = cur_m

    T = T0

    for _ in range(max_iter):
        cand = komsu_yol(G, cur, s, d)
        cand_m = engine.compute(cand, demand_mbps=demand_mbps)
        cand_score = engine.weighted_sum(cand_m, weights)

        delta = cand_score - cur_score
        if delta <= 0 or random.random() < math.exp(-delta / T):
            cur, cur_score, cur_m = cand, cand_score, cand_m

            if cur_score < best_score:
                best, best_score, best_m = cur[:], cur_score, cur_m

        T *= alpha
        if T < 1e-6:
            break

    return best, best_score, best_m


# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":
    G = generate_connected_topology(250, 0.4, 100)
    assign_node_attributes(G)
    assign_edge_attributes(G)

    G = adapt_graph_for_metrics(G)

    kaynak, hedef = 0, 49
    weights = Weights(0.33, 0.33, 0.34)

    yol, skor, m = simulated_annealing(
        G, kaynak, hedef,
        weights=weights
    )

    print("En İyi Yol:", yol)
    print("Toplam Skor:", skor)
    print("Delay (ms):", m.total_delay_ms)
    print("Reliability cost:", m.reliability_cost)
    print("Resource cost:", m.resource_cost)
    print("Bottleneck (Mbps):", m.bottleneck_capacity_mbps)
