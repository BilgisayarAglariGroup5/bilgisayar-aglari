"""
Metrik Hesaplama Fonksiyonları – ORNEK UYGULAMA

Bu modül, ACO algoritmasının çalışma prensibini göstermek amacıyla
kullanılan basit maliyet hesaplama fonksiyonlarını içerir.
Gerçek sistemlerde metrikler daha ayrıntılı şekilde modellenebilir.
"""

import math


def calculate_total_delay(G, path):
    """Yol üzerindeki toplam gecikmeyi (link + node) hesaplar."""
    if not path or len(path) < 2:
        return float("inf")

    total_delay = 0.0

    # Link gecikmeleri
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        total_delay += G[u][v].get("link_delay_ms", 0)

    # Ara düğüm işlem gecikmeleri
    for node in path[1:-1]:
        total_delay += G.nodes[node].get("processing_delay_ms", 0)

    return total_delay


def calculate_reliability_cost(G, path):
    """Yolun güvenilirlik maliyetini logaritmik olarak hesaplar."""
    if not path or len(path) < 2:
        return float("inf")

    cost = 0.0

    # Link güvenilirlikleri
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        rel = G[u][v].get("link_reliability", 0.99)
        cost += -math.log(rel if rel > 0 else 0.0001)

    # Düğüm güvenilirlikleri
    for node in path:
        n_rel = G.nodes[node].get("node_reliability", 0.99)
        cost += -math.log(n_rel if n_rel > 0 else 0.0001)

    return cost


def calculate_resource_cost(G, path):
    """Bant genişliğine bağlı kaynak kullanım maliyetini hesaplar."""
    if not path or len(path) < 2:
        return float("inf")

    cost = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        bw = G[u][v].get("bandwidth_mbps", 100)
        cost += 1000.0 / bw  # Düşük bant genişliği → yüksek maliyet

    return cost


def calculate_weighted_cost(
    G, path,
    w_delay=0.33,
    w_reliability=0.33,
    w_resource=0.34
):
    """
    ORNEK agirlikli maliyet fonksiyonu.
    Tum metrikleri tek bir skor altinda birlestirir.
    """
    if not path:
        return float("inf")

    d = calculate_total_delay(G, path)
    r = calculate_reliability_cost(G, path)
    c = calculate_resource_cost(G, path)

    return w_delay * d + w_reliability * r + w_resource * c
