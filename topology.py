# topology.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Callable

import networkx as nx
import numpy as np


@dataclass
class RoutingResult:
    path: List[int]
    metrics: Dict[str, Any]
    note: str = ""
    hops: Optional[List[Dict[str, Any]]] = None


def generate_graph(n: int = 250, p: float = 0.40, seed: int = 42, ensure_connected: bool = True) -> nx.Graph:
    """
    Erdos-Renyi grafiği üretir ve node/edge attribute ekler.
    PDF gereği grafiğin connected olması (veya en azından S-D arasında yol olması) istenir. :contentReference[oaicite:3]{index=3}
    """
    rng = np.random.default_rng(seed)

    # Bağlı graf üretmek için birkaç deneme yapalım
    for attempt in range(30):
        G = nx.erdos_renyi_graph(n=n, p=p, seed=seed + attempt)

        if (not ensure_connected) or nx.is_connected(G):
            break
    else:
        # hala bağlı değilse en büyük bileşeni al
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    # Node attributes (standardized keys + legacy aliases)
    for node in G.nodes():
        processing_ms = float(rng.uniform(0.5, 2.0))
        G.nodes[node]["processing_delay_ms"] = processing_ms      # standardized
        G.nodes[node]["proc_delay"] = processing_ms               # legacy alias
        G.nodes[node]["node_reliability"] = float(rng.uniform(0.95, 0.999)) # NodeReliability_i

    # Edge attributes (standardized keys + legacy aliases)
    for u, v in G.edges():
        ld = float(rng.uniform(3, 15))
        bw = float(rng.uniform(100, 1000))
        G.edges[u, v]["link_delay_ms"] = ld              # standardized
        G.edges[u, v]["link_delay"] = ld                 # legacy alias
        G.edges[u, v]["bandwidth_mbps"] = bw             # standardized
        G.edges[u, v]["bandwidth"] = bw                  # legacy alias
        G.edges[u, v]["capacity_mbps"] = bw              # alias used by some modules
        G.edges[u, v]["link_reliability"] = float(rng.uniform(0.95, 0.999))  # LinkReliability_ij

    return G


def _normalize_weights(w_delay: float, w_rel: float, w_res: float, normalize: bool) -> Tuple[float, float, float, Optional[str]]:
    if not normalize:
        return w_delay, w_rel, w_res, None

    total = w_delay + w_rel + w_res
    if total <= 1e-12:
        return w_delay, w_rel, w_res, "Ağırlıklar toplamı 0 olamaz."
    return w_delay / total, w_rel / total, w_res / total, None


def compute_path_weighted(
    G: nx.Graph,
    source: int,
    target: int,
    w_delay: float,
    w_rel: float,
    w_res: float,
    normalize: bool = True,
) -> RoutingResult:
    """
    PDF'deki Weighted Sum Method'a göre (Dijkstra) en düşük TotalCost yolu.
    TotalDelay, ReliabilityCost (-log), ResourceCost (1Gbps/BW) PDF'deki tanımlara göre. :contentReference[oaicite:4]{index=4}
    """
    if source == target:
        return RoutingResult(path=[], metrics={}, note="Kaynak ve hedef aynı olamaz.")

    w_delay, w_rel, w_res, err = _normalize_weights(w_delay, w_rel, w_res, normalize)
    if err:
        return RoutingResult(path=[], metrics={}, note=err)

    if source not in G or target not in G:
        return RoutingResult(path=[], metrics={}, note="Girilen düğüm bulunamadı.")

    # NetworkX ağırlık fonksiyonu: (u, v, edge_attr) -> weight
    # Burada v düğümünün maliyetlerini kenara ekliyoruz.
    # TotalDelay'de S ve D hariç ara düğümlerin proc_delay'i eklenir. :contentReference[oaicite:5]{index=5}
    # ReliabilityCost: kenarlar + tüm düğümler için -log(...) :contentReference[oaicite:6]{index=6}
    # ResourceCost: 1Gbps/Bandwidth (Bandwidth Mbps ise 1000/BW) :contentReference[oaicite:7]{index=7}

    def edge_weight(u: int, v: int, ed: Dict[str, Any]) -> float:
        link_delay = float(ed.get("link_delay_ms", ed.get("link_delay", 0.0)))
        link_rel = float(ed.get("link_reliability", 1.0))
        bw_mbps = float(ed.get("bandwidth_mbps", ed.get("bandwidth", 1000.0)))

        # Delay part: link delay + (v ara düğümse processing_delay_ms / legacy proc_delay)
        node_proc = float(G.nodes[v].get("processing_delay_ms", G.nodes[v].get("proc_delay", 0.0)))
        delay_cost = link_delay + (0.0 if (v == source or v == target) else node_proc)

        # ReliabilityCost part: -log(link_rel) + -log(node_rel(v))
        node_rel = float(G.nodes[v].get("node_reliability", 1.0))
        rel_cost = (-np.log(max(link_rel, 1e-12))) + (-np.log(max(node_rel, 1e-12)))

        # ResourceCost part: 1Gbps / Bandwidth_ij  (Mbps -> 1000 / Mbps)
        resource_cost = 1000.0 / max(bw_mbps, 1e-9)

        return (w_delay * delay_cost) + (w_rel * rel_cost) + (w_res * resource_cost)

    try:
        path = nx.shortest_path(G, source=source, target=target, weight=edge_weight, method="dijkstra")
    except nx.NetworkXNoPath:
        return RoutingResult(path=[], metrics={}, note="Kaynak ve hedef arasında yol bulunamadı.")
    except Exception as e:
        return RoutingResult(path=[], metrics={}, note=f"Hata: {e}")

    # --- Metrikleri PDF formüllerine göre hesaplayalım (ekranda göstermek için) ---
    total_delay = 0.0
    reliability_cost = 0.0
    resource_cost = 0.0

    # TotalDelay: link_delay toplamı + ara düğümlerin proc_delay toplamı
    for u, v in zip(path, path[1:]):
        ed = G.edges[u, v]
        total_delay += float(ed.get("link_delay_ms", ed.get("link_delay", 0.0)))
    for k in path:
        if k != source and k != target:
            total_delay += float(G.nodes[k].get("processing_delay_ms", G.nodes[k].get("proc_delay", 0.0)))

    # ReliabilityCost: Σ[-log(link_rel)] + Σ[-log(node_rel)]
    for u, v in zip(path, path[1:]):
        link_rel = float(G.edges[u, v].get("link_reliability", 1.0))
        reliability_cost += (-np.log(max(link_rel, 1e-12)))
    for k in path:
        node_rel = float(G.nodes[k].get("node_reliability", 1.0))
        reliability_cost += (-np.log(max(node_rel, 1e-12)))

    # TotalReliability (göstermek istersen): exp(-ReliabilityCost)
    total_reliability = float(np.exp(-reliability_cost))

    # ResourceCost: Σ(1000/BW_mbps)
    for u, v in zip(path, path[1:]):
        bw = float(G.edges[u, v].get("bandwidth_mbps", G.edges[u, v].get("bandwidth", 1000.0)))
        resource_cost += (1000.0 / max(bw, 1e-9))

    total_cost = (w_delay * total_delay) + (w_rel * reliability_cost) + (w_res * resource_cost)

    # Build per-hop breakdown (for UI / inspection)
    hops = []
    for idx, v in enumerate(path):
        node_proc = float(G.nodes[v].get("processing_delay_ms", G.nodes[v].get("proc_delay", 0.0)))
        node_rel = float(G.nodes[v].get("node_reliability", 1.0))
        hop = {
            "node": int(v),
            "proc_delay": node_proc,
            "node_reliability": node_rel,
            "edge": None,
            "costs": None,
        }
        if idx > 0:
            u = path[idx - 1]
            ed = G.edges[u, v]
            link_delay = float(ed.get("link_delay_ms", ed.get("link_delay", 0.0)))
            link_rel = float(ed.get("link_reliability", 1.0))
            bw_mbps = float(ed.get("bandwidth_mbps", ed.get("bandwidth", 1000.0)))
            delay_cost = link_delay + (0.0 if (v == source or v == target) else node_proc)
            rel_cost = (-np.log(max(link_rel, 1e-12))) + (-np.log(max(node_rel, 1e-12)))
            resource_cost = 1000.0 / max(bw_mbps, 1e-9)
            total_cost_hop = (w_delay * delay_cost) + (w_rel * rel_cost) + (w_res * resource_cost)
            hop["edge"] = {"from": int(u), "to": int(v), "link_delay_ms": link_delay, "link_delay": link_delay, "link_reliability": link_rel, "bandwidth_mbps": bw_mbps, "bandwidth": bw_mbps}
            hop["costs"] = {"delay_cost": delay_cost, "rel_cost": rel_cost, "resource_cost": resource_cost, "total_cost": total_cost_hop}
        hops.append(hop)

    metrics = {
        "total_delay_ms": total_delay,
        "total_reliability": total_reliability,
        "total_reliability_pct": total_reliability * 100.0,
        "reliability_cost": reliability_cost,
        "resource_cost": resource_cost,
        "total_cost": total_cost,
        "weights": {"w_delay": w_delay, "w_rel": w_rel, "w_res": w_res},
        "hops": hops,
    }

    return RoutingResult(path=path, metrics=metrics, note="OK", hops=hops)


def compute_layout(
    G: nx.Graph,
    seed: int = 42,
    layout: str = "spring",
    spread: float = 2.8,      # büyütme katsayısı (artır: daha geniş)
    k: Optional[float] = None,
    iterations: int = 200
) -> Dict[int, Tuple[float, float]]:
    """
    Daha geniş ve okunur layout üretir.
    spread: pozisyonları ölçekler (grafiği büyütür)
    k: node'lar arası mesafe parametresi (None ise otomatik seçilir)
    """
    n = max(G.number_of_nodes(), 1)

    # k otomatik: n büyüdükçe daha seyrek olsun
    if k is None:
        # eski default ~ 1/sqrt(n). Biz bunu büyütüyoruz.
        k = 2.5 / np.sqrt(n)

    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed, k=k, iterations=iterations)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G, seed=seed, k=k, iterations=iterations)

    # Normalize + büyüt (grafiği genişletir)
    xs = np.array([p[0] for p in pos.values()], dtype=float)
    ys = np.array([p[1] for p in pos.values()], dtype=float)
    xs = (xs - xs.mean()) / (xs.std() + 1e-9)
    ys = (ys - ys.mean()) / (ys.std() + 1e-9)

    keys = list(pos.keys())
    for i, node in enumerate(keys):
        pos[node] = (float(xs[i] * spread), float(ys[i] * spread))

    return pos



def build_hops_for_path(G: nx.Graph, path: List[int], w_delay: float, w_rel: float, w_res: float, source: int, target: int) -> List[Dict[str, Any]]:
    """Build per-hop breakdown (same format as compute_path_weighted uses).
    Returns a list of hops where each hop is a dict with node, proc_delay, node_reliability, edge, costs
    """
    hops: List[Dict[str, Any]] = []
    for idx, v in enumerate(path):
        node_proc = float(G.nodes[v].get("processing_delay_ms", G.nodes[v].get("proc_delay", 0.0)))
        node_rel = float(G.nodes[v].get("node_reliability", 1.0))
        hop = {
            "node": int(v),
            "proc_delay": node_proc,
            "node_reliability": node_rel,
            "edge": None,
            "costs": None,
        }
        if idx > 0:
            u = path[idx - 1]
            ed = G.edges[u, v]
            link_delay = float(ed.get("link_delay_ms", ed.get("link_delay", 0.0)))
            link_rel = float(ed.get("link_reliability", 1.0))
            bw_mbps = float(ed.get("bandwidth_mbps", ed.get("bandwidth", 1000.0)))
            delay_cost = link_delay + (0.0 if (v == source or v == target) else node_proc)
            rel_cost = (-np.log(max(link_rel, 1e-12))) + (-np.log(max(node_rel, 1e-12)))
            resource_cost = 1000.0 / max(bw_mbps, 1e-9)
            total_cost_hop = (w_delay * delay_cost) + (w_rel * rel_cost) + (w_res * resource_cost)
            hop["edge"] = {"from": int(u), "to": int(v), "link_delay_ms": link_delay, "link_delay": link_delay, "link_reliability": link_rel, "bandwidth_mbps": bw_mbps, "bandwidth": bw_mbps}
            hop["costs"] = {"delay_cost": delay_cost, "rel_cost": rel_cost, "resource_cost": resource_cost, "total_cost": total_cost_hop}
        hops.append(hop)
    return hops
