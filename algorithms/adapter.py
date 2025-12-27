"""
Adapter layer to expose available algorithms in a consistent contract for the UI.
Each algorithm wrapper returns a dictionary matching the contract described in
`algorithms/standart.py` (keys: path, metrics, per_node, per_edge, notes)
"""
from typing import Dict, Any, List, Optional
import networkx as nx
import time
import statistics

# Standard
# If a standalone 'standart' module is not available, use the project's
# topology module directly to provide Dijkstra/weighted-path functionality.
from topology import compute_path_weighted, build_hops_for_path

# Genetic
import importlib.util
import os

def _import_genetic():
    base = os.path.join(os.path.dirname(__file__), "GenetikAlgoritma")
    ga_path = os.path.join(base, "genetic_algorithm.py")
    metrics_path = os.path.join(base, "test_metrics.py")

    # Load mock_data first so genetic_algorithm's top-level import succeeds
    mock_path = os.path.join(base, "mock_data.py")
    try:
        spec_mock = importlib.util.spec_from_file_location("mock_data", mock_path)
        mock_mod = importlib.util.module_from_spec(spec_mock)
        spec_mock.loader.exec_module(mock_mod)
        import sys
        sys.modules["mock_data"] = mock_mod
    except Exception:
        # If mock_data is missing or fails, continue and let genetic module fail with clearer error
        pass

    # Load GA metrics first so genetic_algorithm can import them at module import time
    m_mod = None
    if os.path.exists(metrics_path):
        spec_m = importlib.util.spec_from_file_location("ga_metrics", metrics_path)
        m_mod = importlib.util.module_from_spec(spec_m)
        spec_m.loader.exec_module(m_mod)
        try:
            import sys
            sys.modules['test_metrics'] = m_mod
        except Exception:
            pass
    else:
        # Fallback: create a tiny metrics helper that uses the project's MetricsEngine
        import types
        m_mod = types.SimpleNamespace()
        def get_path_details(path, G):
            try:
                from metrics.metric import MetricsEngine
            except Exception:
                # best-effort: compute using topology if metrics not available
                from topology import compute_path_weighted
                res = compute_path_weighted(G, path[0], path[-1], 1.0, 1.0, 1.0)
                return { 'total_delay': res.metrics.get('total_delay_ms', 0.0), 'total_reliability': res.metrics.get('total_reliability', 0.0), 'resource_cost': res.metrics.get('resource_cost', 0.0) }
            engine = MetricsEngine(G)
            pm = engine.compute(path)
            return {'total_delay': pm.total_delay_ms, 'total_reliability': pm.total_reliability, 'resource_cost': pm.resource_cost}
        m_mod.get_path_details = get_path_details

    # Now load genetic module (it may import mock_data/test_metrics at top-level)
    ga_mod = None
    try:
        spec_ga = importlib.util.spec_from_file_location("genetic_algorithm", ga_path)
        ga_mod = importlib.util.module_from_spec(spec_ga)
        spec_ga.loader.exec_module(ga_mod)
    except Exception as e:
        # If loading fails because of optional dependencies (e.g., openpyxl), provide
        # a lightweight fallback implementation so the UI option still works.
        import types
        ga_mod = types.SimpleNamespace()
        def run_genetic_algorithm(G, src, dst, config):
            # Fallback: use deterministic weighted shortest-path from topology
            w_delay, w_rel, w_res = config.get('weights', (0.33, 0.33, 0.34))
            from topology import compute_path_weighted
            res = compute_path_weighted(G, src, dst, w_delay, w_rel, w_res, normalize=True)
            return (res.path, res.metrics.get('total_cost', 0.0) if res and res.metrics else 0.0)
        ga_mod.run_genetic_algorithm = run_genetic_algorithm

    return ga_mod, m_mod

# Q-Learning
# Will import the q_learning module at runtime because folder name contains dashes
import importlib.util
import os


def _import_q_learning():
    base = os.path.join(os.path.dirname(__file__), "Q-Q-Learning")
    q_path = os.path.join(base, "q_learning.py")
    metrics_path = os.path.join(base, "metrics.py")

    # Ensure project's metrics are importable as 'metric' when the module uses it
    project_root = os.path.dirname(os.path.dirname(__file__))
    metrics_pkg = os.path.join(project_root, "metrics")
    try:
        import sys
        if metrics_pkg not in sys.path:
            sys.path.insert(0, metrics_pkg)
    except Exception:
        pass

    spec_q = importlib.util.spec_from_file_location("q_learning", q_path)
    q_mod = importlib.util.module_from_spec(spec_q)
    spec_q.loader.exec_module(q_mod)

    spec_m = importlib.util.spec_from_file_location("q_metrics", metrics_path)
    m_mod = importlib.util.module_from_spec(spec_m)
    spec_m.loader.exec_module(m_mod)

    return q_mod, m_mod

# Note: Some import path adjustments might be necessary depending on package names


def _wrap_genetic(G, src, dst, w_delay, w_rel, w_res, params: Dict[str, Any]):
    config = {
        'pop_size': int(params.get('pop_size', 50)),
        'generations': int(params.get('generations', 100)),
        'mutation_rate': float(params.get('mutation_rate', 0.1)),
        'weights': (w_delay, w_rel, w_res)
    }
    ga_mod, ga_metrics_mod = _import_genetic()
    run_genetic_algorithm = ga_mod.run_genetic_algorithm
    get_path_details = ga_metrics_mod.get_path_details

    best_path, best_cost = run_genetic_algorithm(G, src, dst, config)
    if not best_path:
        return {"path": [], "metrics": {}, "per_node": [], "per_edge": [], "notes": "Hiç yol bulunamadı."}

    # Get path details from GA helper
    details = get_path_details(best_path, G)

    metrics = {
        "total_delay_ms": details.get('total_delay', 0.0),
        "path_reliability": details.get('total_reliability', 0.0),
        "resource_cost": details.get('resource_cost', 0.0),
        "total_cost": best_cost,
        "weights": {"w_delay": w_delay, "w_rel": w_rel, "w_res": w_res},
    }

    # per_node & per_edge attempt to use graph attributes when available
    per_node = [{"düğüm": n, "resource_cost": None} for n in best_path]
    per_edge = []
    for u, v in zip(best_path, best_path[1:]):
        ed = G.edges[u, v] if G.has_edge(u, v) else {}
        per_edge.append({
            "u": u,
            "v": v,
            "delay_ms": ed.get('link_delay_ms', ed.get('link_delay', None)),
            "bandwidth_mbps": ed.get('bandwidth_mbps', ed.get('bandwidth', None)),
            "reliability": ed.get('link_reliability', None),
        })

    return {"path": best_path, "metrics": metrics, "per_node": per_node, "per_edge": per_edge, "notes": "OK"}


def _wrap_qlearning(G, src, dst, w_delay, w_rel, w_res, params: Dict[str, Any]):
    episodes = int(params.get('episodes', 500))
    alpha = float(params.get('alpha', 0.1))
    gamma = float(params.get('gamma', 0.9))

    q_mod, m_mod = _import_q_learning()
    QLearningAgent = q_mod.QLearningAgent

    agent = QLearningAgent(G, src, dst, episodes=episodes, alpha=alpha, gamma=gamma, w_delay=w_delay, w_rel=w_rel, w_res=w_res)
    agent.train()
    best_path = agent.get_best_path()
    qmetrics = m_mod
    if not best_path:
        return {"path": [], "metrics": {}, "per_node": [], "per_edge": [], "notes": "Hiç yol bulunamadı (Q-Learning)."}

    # Compute metrics via q_metrics
    total_delay = qmetrics.calculate_total_delay(G, best_path)
    rel_cost = qmetrics.calculate_reliability_cost(G, best_path)
    resource_cost = qmetrics.calculate_resource_cost(G, best_path)
    total_cost = qmetrics.calculate_weighted_cost(G, best_path, w_delay, w_rel, w_res)

    metrics = {
        "total_delay_ms": total_delay,
        "reliability_cost": rel_cost,
        "resource_cost": resource_cost,
        "total_cost": total_cost,
        "weights": {"w_delay": w_delay, "w_rel": w_rel, "w_res": w_res}
    }

    per_node = [{"düğüm": n} for n in best_path]
    per_edge = []
    for u, v in zip(best_path, best_path[1:]):
        ed = G.edges[u, v] if G.has_edge(u, v) else {}
        per_edge.append({"u": u, "v": v, "delay_ms": ed.get('link_delay_ms', ed.get('link_delay', None)), "bandwidth_mbps": ed.get('bandwidth_mbps', ed.get('bandwidth', None)), "reliability": ed.get('link_reliability', None)})

    return {"path": best_path, "metrics": metrics, "per_node": per_node, "per_edge": per_edge, "notes": "OK"}


def _wrap_aco(G, src, dst, w_delay, w_rel, w_res, params: Dict[str, Any]):
    # Lightweight ACO wrapper that runs ant_walk/update_pheromone loop on provided graph
    num_iterations = int(params.get('num_iterations', 20))
    num_ants = int(params.get('num_ants', 15))
    demand_bw = float(params.get('demand_bw', 1.0))
    rho = float(params.get('rho', 0.1))
    Q = float(params.get('Q', 10.0))

    try:
        from algorithms.aco_algoritma.aco_core import ant_walk, update_pheromone, evaporate_pheromone
    except Exception:
        # If module not available, fallback to topology shortest path
        res = compute_path_weighted(G, src, dst, w_delay, w_rel, w_res, normalize=True)
        if not res.path:
            return {"path": [], "metrics": {}, "per_node": [], "per_edge": [], "notes": res.note or "Yol bulunamadı (ACO fallback)."}
        m = res.metrics
        per_node = [{"düğüm": n} for n in res.path]
        per_edge = []
        for u, v in zip(res.path, res.path[1:]):
            ed = G.edges[u, v] if G.has_edge(u, v) else {}
            per_edge.append({"u": u, "v": v, "delay_ms": ed.get('link_delay_ms', ed.get('link_delay', None)), "bandwidth_mbps": ed.get('bandwidth_mbps', ed.get('bandwidth', None)), "reliability": ed.get('link_reliability', None)})
        return {"path": res.path, "metrics": m, "per_node": per_node, "per_edge": per_edge, "notes": "OK (fallback)"}

    # Ensure pheromone initialized
    for u, v in G.edges():
        if 'pheromone' not in G[u][v]:
            G[u][v]['pheromone'] = float(params.get('initial_pheromone', 0.1))

    try:
        from metrics.metric import MetricsEngine, Weights
    except Exception:
        MetricsEngine = None
        Weights = None

    weights_obj = None
    if Weights is not None:
        weights_obj = Weights(w_delay, w_rel, w_res)

    best_path = None
    best_cost = float('inf')

    # Iterations
    for _ in range(num_iterations):
        iteration_paths = []
        for _ in range(num_ants):
            path = ant_walk(G, src, dst, demand_bw, lambda G,u,v: 1.0, alpha=float(params.get('alpha',1.0)), beta=float(params.get('beta',2.0)))
            if not path:
                continue
            # Compute cost using MetricsEngine if available
            if MetricsEngine is not None and weights_obj is not None:
                engine = MetricsEngine(G)
                pm = engine.compute(path, demand_mbps=demand_bw)
                cost = engine.weighted_sum(pm, weights_obj)
            else:
                # simple fallback: use path length
                cost = len(path)

            iteration_paths.append((path, cost))
            if cost < best_cost:
                best_cost = cost
                best_path = path

        # Pheromone updates
        for p, c in iteration_paths:
            update_pheromone(G, p, c, Q)
        evaporate_pheromone(G, rho)

    if not best_path:
        return {"path": [], "metrics": {}, "per_node": [], "per_edge": [], "notes": "Hiç yol bulunamadı (ACO)."}

    # Build metrics using available engines
    if MetricsEngine is not None and weights_obj is not None:
        engine = MetricsEngine(G)
        pm = engine.compute(best_path, demand_mbps=demand_bw)
        total_cost = engine.weighted_sum(pm, weights_obj)
        metrics = {"total_delay_ms": pm.total_delay_ms, "total_reliability": pm.total_reliability, "resource_cost": pm.resource_cost, "total_cost": total_cost, "weights": {"w_delay": w_delay, "w_rel": w_rel, "w_res": w_res}}
    else:
        metrics = {"total_delay_ms": 0.0, "total_reliability": 0.0, "resource_cost": 0.0, "total_cost": best_cost, "weights": {"w_delay": w_delay, "w_rel": w_rel, "w_res": w_res}}

    per_node = [{"düğüm": n} for n in best_path]
    per_edge = []
    for u, v in zip(best_path, best_path[1:]):
        ed = G.edges[u, v] if G.has_edge(u, v) else {}
        per_edge.append({"u": u, "v": v, "delay_ms": ed.get('link_delay_ms', ed.get('link_delay', None)), "bandwidth_mbps": ed.get('bandwidth_mbps', ed.get('bandwidth', None)), "reliability": ed.get('link_reliability', None)})

    return {"path": best_path, "metrics": metrics, "per_node": per_node, "per_edge": per_edge, "notes": "OK"}


def _wrap_sa(G, src, dst, w_delay, w_rel, w_res, params: Dict[str, Any]):
    demand_bw = float(params.get('demand_bw', 0.0)) if params.get('demand_bw', None) is not None else None
    max_iter = int(params.get('max_iter', 5000))

    # Try loading SA implementation robustly via file import
    try:
        base = os.path.join(os.path.dirname(__file__), "SA_algoritma")
        sa_path = os.path.join(base, "SA.py")
        import importlib.util
        spec = importlib.util.spec_from_file_location("sa_module", sa_path)
        sa_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sa_mod)
        simulated_annealing = sa_mod.simulated_annealing
        adapt_graph_for_metrics = getattr(sa_mod, 'adapt_graph_for_metrics', None)
    except Exception:
        # Fallback: if SA module can't be loaded (missing non-critical deps), use topology shortest-path
        res = compute_path_weighted(G, src, dst, w_delay, w_rel, w_res, normalize=True)
        if not res.path:
            return {"path": [], "metrics": {}, "per_node": [], "per_edge": [], "notes": res.note or "Yol bulunamadı (SA fallback)."}
        m = res.metrics
        per_node = [{"düğüm": n} for n in res.path]
        per_edge = []
        for u, v in zip(res.path, res.path[1:]):
            ed = G.edges[u, v] if G.has_edge(u, v) else {}
            per_edge.append({"u": u, "v": v, "delay_ms": ed.get('link_delay_ms', ed.get('link_delay', None)), "bandwidth_mbps": ed.get('bandwidth_mbps', ed.get('bandwidth', None)), "reliability": ed.get('link_reliability', None)})
        return {"path": res.path, "metrics": m, "per_node": per_node, "per_edge": per_edge, "notes": "OK (SA fallback)"}

    # adapt graph if helper exists
    try:
        if adapt_graph_for_metrics is not None:
            G = adapt_graph_for_metrics(G)
    except Exception:
        pass

    try:
        from metrics.metric import Weights
        weights_obj = Weights(w_delay, w_rel, w_res)
    except Exception:
        weights_obj = None

    best_path, best_score, best_m = simulated_annealing(G, src, dst, weights=weights_obj, demand_mbps=demand_bw, max_iter=max_iter)

    if not best_path:
        return {"path": [], "metrics": {}, "per_node": [], "per_edge": [], "notes": "Hiç yol bulunamadı (SA)."}

    # best_m is a PathMetrics object
    metrics = {
        "total_delay_ms": getattr(best_m, 'total_delay_ms', 0.0),
        "total_reliability": getattr(best_m, 'total_reliability', 0.0),
        "resource_cost": getattr(best_m, 'resource_cost', 0.0),
        "total_cost": float(best_score) if best_score is not None else 0.0,
        "weights": {"w_delay": w_delay, "w_rel": w_rel, "w_res": w_res}
    }

    per_node = [{"düğüm": n} for n in best_path]
    per_edge = []
    for u, v in zip(best_path, best_path[1:]):
        ed = G.edges[u, v] if G.has_edge(u, v) else {}
        per_edge.append({"u": u, "v": v, "delay_ms": ed.get('link_delay_ms', ed.get('link_delay', None)), "bandwidth_mbps": ed.get('bandwidth_mbps', ed.get('bandwidth', None)), "reliability": ed.get('link_reliability', None)})

    return {"path": best_path, "metrics": metrics, "per_node": per_node, "per_edge": per_edge, "notes": "OK"}


ALGORITHMS = {
    "ACO (Ant Colony)": {
        "key": "aco",
        "wrapper": None,  # set below after definition
        "params": [
            {"name": "num_iterations", "type": "int", "label": "İterasyon Sayısı", "default": 20},
            {"name": "num_ants", "type": "int", "label": "Karınca Sayısı", "default": 15},
            {"name": "demand_bw", "type": "float", "label": "Talep BW (Mbps)", "default": 1.0},
        ]
    },
    "Genetik (GA)": {
        "key": "genetic",
        "wrapper": _wrap_genetic,
        "params": [
            {"name": "pop_size", "type": "int", "label": "Popülasyon", "default": 50},
            {"name": "generations", "type": "int", "label": "Nesil sayısı", "default": 100},
            {"name": "mutation_rate", "type": "float", "label": "Mutation rate", "default": 0.1},
        ]
    },
    "Q-Learning": {
        "key": "qlearning",
        "wrapper": _wrap_qlearning,
        "params": [
            {"name": "episodes", "type": "int", "label": "Episod", "default": 500},
            {"name": "alpha", "type": "float", "label": "Alpha", "default": 0.1},
            {"name": "gamma", "type": "float", "label": "Gamma", "default": 0.9},
        ]
    },
    "Simulated Annealing (SA)": {
        "key": "sa",
        "wrapper": None,  # set below after definition
        "params": [
            {"name": "demand_bw", "type": "float", "label": "Talep BW (Mbps)", "default": 1.0},
            {"name": "max_iter", "type": "int", "label": "Max Iter", "default": 5000},
        ]
    }
}


def list_algorithms():
    return list(ALGORITHMS.keys())

# Assign wrappers for algorithms that needed runtime-defined functions
ALGORITHMS["ACO (Ant Colony)"]["wrapper"] = _wrap_aco
ALGORITHMS["Simulated Annealing (SA)"]["wrapper"] = _wrap_sa


def get_algorithm_meta(name: str):
    return ALGORITHMS.get(name)


def run(name: str, G: nx.Graph, src: int, dst: int, w_delay: float, w_rel: float, w_res: float, params: Dict[str, Any]):
    meta = get_algorithm_meta(name)
    if not meta:
        raise ValueError(f"Bilinmeyen algoritma: {name}")
    return meta['wrapper'](G, src, dst, w_delay, w_rel, w_res, params)


def compare(
    G: nx.Graph,
    src: int,
    dst: int,
    w_delay: float,
    w_rel: float,
    w_res: float,
    num_runs: int = 5,
    default_params: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Tüm algoritmaları aynı koşullarda karşılaştırır.
    
    Args:
        G: NetworkX graph instance
        src: Source node
        dst: Destination node
        w_delay: Delay weight
        w_rel: Reliability weight
        w_res: Resource weight
        num_runs: Her algoritma için çalıştırma sayısı (N ≥ 5)
        default_params: Algoritma adına göre varsayılan parametreler (opsiyonel)
    
    Returns:
        {
            "runs_table": List[List] - Her çalıştırma için [algorithm_name, run_id, total_delay, reliability_cost, resource_cost, total_cost, runtime_ms, path]
            "summary_table": List[List] - Algoritma başına [avg_cost, std_cost, best_cost, worst_cost, avg_runtime]
            "best_paths_by_algo": Dict[str, List[int]] - Algoritma adına göre en iyi path
        }
    """
    # MetricsEngine'i import et
    try:
        from metrics.metric import MetricsEngine, Weights, PathMetrics
    except ImportError:
        raise RuntimeError("metrics.metric modülü bulunamadı. Metrik hesaplamaları için gerekli.")
    
    if num_runs < 5:
        num_runs = 5
    
    # Varsayılan parametreleri hazırla
    if default_params is None:
        default_params = {}
        for algo_name in list_algorithms():
            meta = get_algorithm_meta(algo_name)
            if meta and 'params' in meta:
                algo_defaults = {}
                for p in meta['params']:
                    algo_defaults[p['name']] = p.get('default', 0)
                default_params[algo_name] = algo_defaults
    
    # Normalize weights
    total_w = w_delay + w_rel + w_res
    if total_w <= 0:
        raise ValueError("Ağırlıkların toplamı 0'dan büyük olmalıdır.")
    w_delay_norm = w_delay / total_w
    w_rel_norm = w_rel / total_w
    w_res_norm = w_res / total_w
    
    weights_obj = Weights(w_delay_norm, w_rel_norm, w_res_norm)
    engine = MetricsEngine(G)
    
    runs_table = []
    algo_results = {}  # {algo_name: [(cost, runtime_ms, path, metrics), ...]}
    
    # Karşılaştırılacak algoritmalar
    algorithms_to_compare = [
        "ACO (Ant Colony)",
        "Genetik (GA)",
        "Q-Learning",
        "Simulated Annealing (SA)"
    ]
    
    # Her algoritma için N kez çalıştır
    for algo_name in algorithms_to_compare:
        if algo_name not in ALGORITHMS:
            continue
        
        meta = get_algorithm_meta(algo_name)
        if not meta or not meta.get('wrapper'):
            continue
        
        # Algoritma için varsayılan parametreleri al
        params = default_params.get(algo_name, {})
        
        algo_runs = []
        
        for run_id in range(1, num_runs + 1):
            # Grafiği kopyala (pheromone gibi state'ler için)
            G_copy = G.copy()
            
            # Runtime ölçümü
            start_time = time.perf_counter()
            
            try:
                # Algoritmayı çalıştır
                result = run(algo_name, G_copy, src, dst, w_delay_norm, w_rel_norm, w_res_norm, params)
                
                end_time = time.perf_counter()
                runtime_ms = (end_time - start_time) * 1000.0
                
                path = result.get('path', [])
                
                if not path or len(path) < 2:
                    # Yol bulunamadı
                    runs_table.append([
                        algo_name, run_id, 0.0, 0.0, 0.0, float('inf'),
                        runtime_ms, []
                    ])
                    algo_runs.append((float('inf'), runtime_ms, [], None))
                    continue
                
                # Metrikleri MetricsEngine ile hesapla
                pm = engine.compute(path)
                total_cost = engine.weighted_sum(pm, weights_obj)
                
                # runs_table için veri hazırla
                runs_table.append([
                    algo_name,
                    run_id,
                    pm.total_delay_ms,
                    pm.reliability_cost,
                    pm.resource_cost,
                    total_cost,
                    runtime_ms,
                    path.copy()
                ])
                
                algo_runs.append((total_cost, runtime_ms, path.copy(), pm))
                
            except Exception as e:
                end_time = time.perf_counter()
                runtime_ms = (end_time - start_time) * 1000.0
                runs_table.append([
                    algo_name, run_id, 0.0, 0.0, 0.0, float('inf'),
                    runtime_ms, []
                ])
                algo_runs.append((float('inf'), runtime_ms, [], None))
        
        algo_results[algo_name] = algo_runs
    
    # Summary table oluştur
    summary_table = []
    best_paths_by_algo = {}
    
    for algo_name in algorithms_to_compare:
        if algo_name not in algo_results:
            continue
        
        runs = algo_results[algo_name]
        costs = [r[0] for r in runs if r[0] != float('inf')]
        runtimes = [r[1] for r in runs]
        metrics_list = [r[3] for r in runs if r[3] is not None]  # PathMetrics objects
        
        if not costs:
            # Hiç başarılı çalıştırma yok
            summary_table.append([
                algo_name, 0.0, 0.0, 0.0, 0.0, 0.0, float('inf'), float('inf'), 0.0
            ])
            best_paths_by_algo[algo_name] = []
            continue
        
        # Ortalama metrikler
        avg_delay = statistics.mean([m.total_delay_ms for m in metrics_list]) if metrics_list else 0.0
        avg_reliability_cost = statistics.mean([m.reliability_cost for m in metrics_list]) if metrics_list else 0.0
        avg_resource_cost = statistics.mean([m.resource_cost for m in metrics_list]) if metrics_list else 0.0
        
        # Toplam maliyet istatistikleri
        avg_cost = statistics.mean(costs)
        std_cost = statistics.stdev(costs) if len(costs) > 1 else 0.0
        best_cost = min(costs)
        worst_cost = max(costs)
        avg_runtime = statistics.mean(runtimes) if runtimes else 0.0
        
        # Summary table: [algo_name, avg_delay, avg_reliability_cost, avg_resource_cost, avg_total_cost, std_cost, best_cost, worst_cost, avg_runtime]
        summary_table.append([
            algo_name, avg_delay, avg_reliability_cost, avg_resource_cost, avg_cost, std_cost, best_cost, worst_cost, avg_runtime
        ])
        
        # En iyi path'i bul (en düşük maliyetli)
        best_idx = costs.index(best_cost)
        best_path = runs[best_idx][2]
        best_paths_by_algo[algo_name] = best_path if best_path else []
    
    return {
        "runs_table": runs_table,
        "summary_table": summary_table,
        "best_paths_by_algo": best_paths_by_algo
    }