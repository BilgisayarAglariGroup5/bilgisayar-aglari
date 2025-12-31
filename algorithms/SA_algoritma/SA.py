# algorithms/SA_algoritma/SA.py
import math
import random
import networkx as nx
from typing import Optional, List, Tuple
from metrics.metric import MetricsEngine, Weights


# ==================================================
# HARD BANDWIDTH CHECK (SERT KISIT)
# ==================================================
def path_is_feasible(G: nx.Graph, path: List[int], demand: Optional[float]) -> bool:
    """
    Bu fonksiyon, verilen yolun (path) bant genişliği talebini (demand) karşılayıp karşılamadığını kontrol eder.

    - demand None ise: bant genişliği kısıtı kapalı demektir, yol her zaman geçerli kabul edilir.
    - demand bir sayı ise: yol üzerindeki HER bir kenarın bandwidth değeri >= demand olmalıdır.
      Aksi halde yol infeasible (geçersiz) olur.
    """
    if demand is None:
        return True

    demand = float(demand) # demand string/int gelirse sorun olmasın diye float’a çeviriyor.

    # zip(path[:-1], path[1:]) => (u,v) kenar çiftlerini üretir
    for u, v in zip(path[:-1], path[1:]):
        # kenardaki bandwidth’i okuyor.
        # attribute bazen "bandwidth" bazen "bandwidth_mbps" olabilir diye iki ihtimal var.
        # ikisi de yoksa 0 kabul ediyor.
        bw = G[u][v].get("bandwidth", G[u][v].get("bandwidth_mbps", 0))

        # herhangi bir kenar yetmiyorsa → yol komple geçersiz.
        if bw < demand:
            return False

    # Tüm kenarlar demand'i sağlıyorsa yol geçerli
    return True


def build_feasible_subgraph(G: nx.Graph, demand: Optional[float]) -> nx.Graph:
    """
    Bu fonksiyon, 'demand' değerini sağlayan kenarlardan oluşan bir alt grafik (subgraph) üretir.

    Mantık:
    - demand None ise: filtreleme yok, G grafiğini olduğu gibi döndür.
    - demand varsa: capacity/bandwidth < demand olan kenarları tamamen grafikten çıkariyor.
      Böylece shortest_path asla bu kenarları kullanamaz. (Hard constraint)
    """
    if demand is None:
        return G

    # demand float’a çevrilir
    # H yeni boş graph (filtrelenmiş graph)
    demand = float(demand)
    H = nx.Graph()

    # tüm düğümleri aynen kopyalar (attribute’lar dahil)
    H.add_nodes_from(G.nodes(data=True))

    # tüm kenarları dolaşır, a o kenarın attribute’ları (delay, reliability, bandwidth vs.)
    for u, v, a in G.edges(data=True):
        # sadece bandwidth yeterliyse kenarı H’ye ekler (attribute’larıyla birlikte)
        bw = a.get("bandwidth", a.get("bandwidth_mbps", 0))
        if bw >= demand:
            H.add_edge(u, v, **a)
    # Bant genişliği yetmeyen linkleri silip yeni graph üretir.”
    return H


# ==================================================
# SIMULATED ANNEALING (HARD CONSTRAINT)
# ==================================================
def simulated_annealing(
    G: nx.Graph,
    source,
    target,
    *,
    weights: Weights,
    demand_mbps: Optional[float] = None,
    T0: float = 5.0,
    alpha: float = 0.995,
    max_iter: int = 5000,
) -> Tuple[Optional[List[int]], float, Optional[object]]:
    """
    Fungsi simulated_annealing, graph, source ve target alarak Simulated Annealing algoritmasını çalıştırır.
    weights metriklerin önemini, demand_mbps ise bant genişliği kısıtını belirler. 
    T0, alpha ve max_iter algoritmanın keşif davranışını kontrol eder. 
    Fonksiyon, en iyi yolu, bu yolun maliyetini ve hesaplanan metrikleri döndürür.

    """

    # Metrikleri hesaplayacak motor (delay, reliability, resource vs.)
    engine = MetricsEngine(G)

    # ---- HARD FILTER GRAPH ----
    # demand varsa: uygun olmayan kenarlar çıkarıldı.
    Gf = build_feasible_subgraph(G, demand_mbps)

    # İlk çözümü link_delay ağırlığına göre en kısa yol yapıyor.
    try:
        current = nx.shortest_path(Gf, source, target, weight="link_delay")
    except nx.NetworkXNoPath:
        # Eğer demand yüzünden yol yoksa → direkt “no solution”.
        return None, float("inf"), None

    # Başlangıç yolunun metriklerini ve skorunu hesapla
    current_m = engine.compute(current, demand_mbps=demand_mbps)
    current_score = engine.weighted_sum(current_m, weights)

    # Şu anki çözümü en iyisi sayıp kenara koyuyoruz
    best = current[:]
    best_score = current_score
    best_m = current_m

    # Başlangıç sıcaklığı
    T = T0

    # Her iterasyonda “komşu çözüm” üretip kabul edip etmeyeceğine bakıyor.
    for _ in range(max_iter):
        # Yol sadece [source, target] gibi ise pivot seçip parçalamak mantıksız.
        if len(current) <= 2:
            break

        # Komşu üretimi:
        # current yolunun içinden rastgele bir pivot seçiyoruz (baş ve son hariç)
        i = random.randint(1, len(current) - 2)
        pivot = current[i]

        # Pivot -> target arasını yeniden shortest_path ile hesaplayıp
        # current[:i] ile birleştirerek yeni aday yol oluşturuyoruz
        try:
            tail = nx.shortest_path(Gf, pivot, target, weight="link_delay")
            candidate = current[:i] + tail
        except nx.NetworkXNoPath:
            # Pivot'tan target'a giden yol yoksa bu iterasyonu geç, sıcaklığı düşürür.
            T *= alpha
            continue

        # Ek güvenlik kontrolü:
        # (Normalde Gf zaten filtreli, ama yine de kesin olsun diye kontrol ediyoruz)
        if not path_is_feasible(G, candidate, demand_mbps):
            T *= alpha
            continue

        # Aday yolun metriklerini ve skorunu hesapla
        cand_m = engine.compute(candidate, demand_mbps=demand_mbps)
        cand_score = engine.weighted_sum(cand_m, weights)

        # Skor farkı: negatifse aday daha iyi (minimizasyon varsayımı)
        delta = cand_score - current_score

        # Kabul kriteri:
        # - aday daha iyiyse (delta < 0) direkt kabul
        # - daha kötüyse de bazen kabul edebilir (exp(-delta/T)) -> SA'nın kaçış mekanizması
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-9)):
            current, current_score = candidate, cand_score

            # Eğer kabul edilen aday en iyiden de iyiyse best'i güncelle
            if cand_score < best_score:
                best, best_score, best_m = candidate[:], cand_score, cand_m

        # Sıcaklığı düşür (cooling)
        # çok küçülürse artık rastgele kabul etme neredeyse yok → dur.
        T *= alpha
        if T < 1e-6:
            break

    # Final güvenlik:
    # Her ihtimale karşı best yol demand'i sağlıyor mu?
    if not path_is_feasible(G, best, demand_mbps):
        return None, float("inf"), None

    return best, best_score, best_m
