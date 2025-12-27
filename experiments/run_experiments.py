"""
Batch deney modu: 20 farklı (S,D,B) senaryosu için algoritmaları karşılaştırır.
Graph seed=42 ile 1 kez oluşturulur ve tüm senaryolarda aynı graph kullanılır.
Her senaryo için 4 algoritma (ACO, GA, Q-Learning, SA) 5 tekrar çalıştırılır.
Sonuçlar experiments/results/ klasörüne CSV olarak kaydedilir.
"""
import os
import sys
import csv
import time
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Proje root'unu path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import networkx as nx
from topology import generate_graph
from algorithms.adapter import compare, list_algorithms, get_algorithm_meta
from metrics.metric import MetricsEngine, Weights, PathMetrics


# Sabit parametreler
GRAPH_SEED = 42
GRAPH_N = 250
GRAPH_P = 0.40
W_DELAY = 0.33
W_REL = 0.33
W_RES = 0.34
NUM_RUNS = 5
BASE_SEED = 1000  # Algoritma seed'leri için base


def validate_path(G: nx.Graph, path: List[int], S: int, D: int, B: float) -> Tuple[bool, Optional[str]]:
    """
    Path'in geçerli olup olmadığını kontrol eder.
    Returns: (is_valid, fail_reason)
    """
    if not path or len(path) < 2:
        return False, "no_path"
    
    # S ve D kontrolü
    if path[0] != S:
        return False, "invalid_path"
    if path[-1] != D:
        return False, "invalid_path"
    
    # Edge kontrolü
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if not G.has_edge(u, v):
            return False, "invalid_path"
        
        # Bandwidth kontrolü
        edge_data = G.edges[u, v]
        capacity = edge_data.get('capacity_mbps', edge_data.get('bandwidth_mbps', 0.0))
        if capacity < B:
            return False, "bandwidth_constraint"
    
    return True, None


def run_single_scenario(
    G: nx.Graph,
    scenario_id: int,
    S: int,
    D: int,
    B: float,
    base_seed: int
) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]]]:
    """
    Tek bir senaryo için tüm algoritmaları çalıştırır.
    
    Returns:
        runs_data: Her run için detaylı veri
        algo_stats: Algoritma başına istatistikler (summary için)
    """
    runs_data = []
    algo_stats = {}  # {algo_name: [costs...]}
    
    # Algoritmalar
    algorithms = [
        "ACO (Ant Colony)",
        "Genetik (GA)",
        "Q-Learning",
        "Simulated Annealing (SA)"
    ]
    
    # Her algoritma için varsayılan parametreleri hazırla (B'yi ekle)
    default_params = {}
    for algo_name in algorithms:
        meta = get_algorithm_meta(algo_name)
        if meta and 'params' in meta:
            algo_defaults = {}
            for p in meta['params']:
                algo_defaults[p['name']] = p.get('default', 0)
            # ACO ve SA için demand_bw ekle
            if algo_name in ["ACO (Ant Colony)", "Simulated Annealing (SA)"]:
                algo_defaults['demand_bw'] = B
            default_params[algo_name] = algo_defaults
    
    # compare() fonksiyonunu kullanarak tüm algoritmaları çalıştır
    try:
        result = compare(
            G=G,
            src=S,
            dst=D,
            w_delay=W_DELAY,
            w_rel=W_REL,
            w_res=W_RES,
            num_runs=NUM_RUNS,
            default_params=default_params,
            demand_mbps=B
        )
        
        runs_table = result.get('runs_table', [])
        
        # runs_table formatı: [algorithm_name, run_id, total_delay, reliability_cost, resource_cost, total_cost, runtime_ms, path]
        for row in runs_table:
            algo_name = row[0]
            run_id = row[1]
            total_delay = row[2]
            reliability_cost = row[3]
            resource_cost = row[4]
            total_cost = row[5]
            runtime_ms = row[6]
            path = row[7] if len(row) > 7 else []
            
            # Path validasyonu
            is_valid, fail_reason = validate_path(G, path, S, D, B)
            
            if not is_valid or total_cost == float('inf'):
                status = "FAIL"
                if not fail_reason:
                    fail_reason = "no_path" if not path else "invalid_path"
            else:
                status = "SUCCESS"
                fail_reason = None
                # İstatistikler için başarılı cost'ları topla
                if algo_name not in algo_stats:
                    algo_stats[algo_name] = []
                algo_stats[algo_name].append(total_cost)
            
            # Path'i string'e çevir
            path_str = "->".join(map(str, path)) if path else ""
            
            runs_data.append({
                'scenario_id': scenario_id,
                'S': S,
                'D': D,
                'B': B,
                'algorithm': algo_name,
                'run_id': run_id,
                'status': status,
                'fail_reason': fail_reason or "",
                'total_delay': total_delay,
                'reliability_cost': reliability_cost,
                'resource_cost': resource_cost,
                'total_cost': total_cost if total_cost != float('inf') else None,
                'runtime_ms': runtime_ms,
                'path': path_str
            })
    
    except Exception as e:
        # Hata durumunda tüm algoritmalar için FAIL kaydı oluştur
        for algo_name in algorithms:
            for run_id in range(1, NUM_RUNS + 1):
                runs_data.append({
                    'scenario_id': scenario_id,
                    'S': S,
                    'D': D,
                    'B': B,
                    'algorithm': algo_name,
                    'run_id': run_id,
                    'status': "FAIL",
                    'fail_reason': f"runtime_error: {str(e)}",
                    'total_delay': None,
                    'reliability_cost': None,
                    'resource_cost': None,
                    'total_cost': None,
                    'runtime_ms': 0.0,
                    'path': ""
                })
    
    return runs_data, algo_stats


def compute_summary(runs_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    runs_data'dan summary_table oluşturur.
    """
    # Senaryo ve algoritma bazında grupla
    grouped = {}  # {(scenario_id, algo_name): [runs...]}
    
    for run in runs_data:
        key = (run['scenario_id'], run['algorithm'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(run)
    
    summary_rows = []
    
    for (scenario_id, algo_name), runs in grouped.items():
        # Başarılı run'ları filtrele
        success_runs = [r for r in runs if r['status'] == 'SUCCESS' and r['total_cost'] is not None]
        success_count = len(success_runs)
        total_count = len(runs)
        success_rate = success_count / total_count if total_count > 0 else 0.0
        
        if success_count > 0:
            costs = [r['total_cost'] for r in success_runs]
            runtimes = [r['runtime_ms'] for r in success_runs]
            
            avg_total_cost = statistics.mean(costs)
            std_total_cost = statistics.stdev(costs) if len(costs) > 1 else 0.0
            best_total_cost = min(costs)
            worst_total_cost = max(costs)
            avg_runtime_ms = statistics.mean(runtimes)
        else:
            avg_total_cost = None
            std_total_cost = None
            best_total_cost = None
            worst_total_cost = None
            avg_runtime_ms = statistics.mean([r['runtime_ms'] for r in runs]) if runs else 0.0
        
        summary_rows.append({
            'scenario_id': scenario_id,
            'algorithm': algo_name,
            'success_count': success_count,
            'success_rate': success_rate,
            'avg_total_cost': avg_total_cost,
            'std_total_cost': std_total_cost,
            'best_total_cost': best_total_cost,
            'worst_total_cost': worst_total_cost,
            'avg_runtime_ms': avg_runtime_ms
        })
    
    return summary_rows


def main():
    """Ana batch deney fonksiyonu"""
    print("=" * 60)
    print("BATCH DENEY MODU - QoS Routing Algoritma Karşılaştırması")
    print("=" * 60)
    
    # Klasörleri oluştur
    experiments_dir = Path(__file__).parent
    results_dir = experiments_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    scenarios_file = experiments_dir / "scenarios.csv"
    
    if not scenarios_file.exists():
        print(f"HATA: {scenarios_file} bulunamadı!")
        return
    
    # Graph'ı 1 kez oluştur (reproducible)
    print(f"\n[1/4] Graph oluşturuluyor (seed={GRAPH_SEED}, n={GRAPH_N}, p={GRAPH_P})...")
    G = generate_graph(n=GRAPH_N, p=GRAPH_P, seed=GRAPH_SEED, ensure_connected=True)
    print(f"      Graph hazır: {G.number_of_nodes()} node, {G.number_of_edges()} edge")
    
    # Senaryoları oku
    print(f"\n[2/4] Senaryolar okunuyor: {scenarios_file}")
    scenarios = []
    with open(scenarios_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenarios.append({
                'scenario_id': int(row['scenario_id']),
                'S': int(row['S']),
                'D': int(row['D']),
                'B': float(row['B'])
            })
    
    print(f"      {len(scenarios)} senaryo yüklendi")
    
    # Senaryo validasyonu
    print(f"\n[3/4] Senaryo validasyonu yapılıyor...")
    valid_scenarios = []
    invalid_scenarios = []
    
    for sc in scenarios:
        S, D, B = sc['S'], sc['D'], sc['B']
        if S == D:
            invalid_scenarios.append((sc['scenario_id'], "S == D"))
            continue
        if S not in G.nodes():
            invalid_scenarios.append((sc['scenario_id'], f"S={S} graph'ta yok"))
            continue
        if D not in G.nodes():
            invalid_scenarios.append((sc['scenario_id'], f"D={D} graph'ta yok"))
            continue
        if not nx.has_path(G, S, D):
            invalid_scenarios.append((sc['scenario_id'], f"S={S} -> D={D} arasında yol yok"))
            continue
        valid_scenarios.append(sc)
    
    if invalid_scenarios:
        print(f"      UYARI: {len(invalid_scenarios)} geçersiz senaryo bulundu:")
        for sid, reason in invalid_scenarios:
            print(f"        - Senaryo {sid}: {reason}")
    
    print(f"      {len(valid_scenarios)} geçerli senaryo çalıştırılacak")
    
    # Tüm senaryoları çalıştır
    print(f"\n[4/4] Deneyler çalıştırılıyor...")
    print(f"      Her senaryo için {len(list_algorithms())} algoritma x {NUM_RUNS} tekrar = {len(list_algorithms()) * NUM_RUNS} run")
    print(f"      Toplam: {len(valid_scenarios)} senaryo x {len(list_algorithms()) * NUM_RUNS} run = {len(valid_scenarios) * len(list_algorithms()) * NUM_RUNS} run")
    
    all_runs_data = []
    start_time_total = time.time()
    
    for idx, sc in enumerate(valid_scenarios, 1):
        scenario_id = sc['scenario_id']
        S, D, B = sc['S'], sc['D'], sc['B']
        
        print(f"\n      [{idx}/{len(valid_scenarios)}] Senaryo {scenario_id}: S={S}, D={D}, B={B}")
        
        try:
            runs_data, algo_stats = run_single_scenario(G, scenario_id, S, D, B, BASE_SEED)
            all_runs_data.extend(runs_data)
            
            # Kısa özet
            for algo_name, costs in algo_stats.items():
                if costs:
                    print(f"        {algo_name}: {len(costs)}/{NUM_RUNS} başarılı, avg_cost={statistics.mean(costs):.2f}")
                else:
                    print(f"        {algo_name}: 0/{NUM_RUNS} başarılı")
        
        except Exception as e:
            print(f"        HATA: {e}")
            import traceback
            traceback.print_exc()
    
    elapsed_total = time.time() - start_time_total
    print(f"\n      Tüm deneyler tamamlandı! Süre: {elapsed_total:.2f} saniye")
    
    # Summary hesapla
    print(f"\n[5/5] Özet tabloları oluşturuluyor...")
    summary_data = compute_summary(all_runs_data)
    
    # CSV'lere yaz
    runs_file = results_dir / "runs_table.csv"
    summary_file = results_dir / "summary_table.csv"
    
    # runs_table.csv
    print(f"      {runs_file} yazılıyor...")
    with open(runs_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'scenario_id', 'S', 'D', 'B', 'algorithm', 'run_id', 'status', 'fail_reason',
            'total_delay', 'reliability_cost', 'resource_cost', 'total_cost', 'runtime_ms', 'path'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_runs_data:
            writer.writerow(row)
    
    # summary_table.csv
    print(f"      {summary_file} yazılıyor...")
    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'scenario_id', 'algorithm', 'success_count', 'success_rate',
            'avg_total_cost', 'std_total_cost', 'best_total_cost', 'worst_total_cost', 'avg_runtime_ms'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_data:
            writer.writerow(row)
    
    print(f"\n{'=' * 60}")
    print("BATCH DENEY TAMAMLANDI!")
    print(f"{'=' * 60}")
    print(f"\nSonuçlar:")
    print(f"  - runs_table.csv: {len(all_runs_data)} satır")
    print(f"  - summary_table.csv: {len(summary_data)} satır")
    print(f"\nDosya konumları:")
    print(f"  - {runs_file}")
    print(f"  - {summary_file}")


if __name__ == "__main__":
    main()

