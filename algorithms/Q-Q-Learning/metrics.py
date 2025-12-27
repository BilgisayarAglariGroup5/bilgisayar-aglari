import networkx as nx
import math

def calculate_total_delay(G, path):
    """Toplam Gecikme: Link Gecikmeleri + Node İşlem Süreleri"""
    total_delay = 0
    
    # 1. Link Gecikmeleri
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        total_delay += G[u][v].get('link_delay', 0)
        
    # 2. Node İşlem Süreleri (Kaynak ve Hedef hariç - Doküman isteği)
    # path[1:-1] -> İlk ve son düğümü hariç tutar
    if len(path) > 2: #Eğer başlangıç ve bitiş haricinde bir düğüm varsa 
        for node in path[1:-1]:
            total_delay += G.nodes[node].get('processing_delay', 0)
        
    return total_delay

def calculate_reliability_cost(G, path):
    """Güvenilirlik Maliyeti: -log(R_link) + -log(R_node)"""
    cost = 0
    
    # Link Güvenilirliği
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        rel = G[u][v].get('link_reliability', 0.99)
        if rel > 0:
            cost += -math.log(rel)
            
    # Node Güvenilirliği (Hepsini dahil ediyoruz)
    for node in path:
        rel = G.nodes[node].get('reliability', 0.99)
        if rel > 0:
            cost += -math.log(rel)
            
    return cost

def calculate_resource_cost(G, path):
    """Kaynak Maliyeti: 1000 / Bant Genişliği (Mbps)"""
    cost = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        bw = G[u][v].get('bandwidth', 100) # Varsayılan 100 Mbps
        if bw > 0:
            # Dokümanda 1 Gbps / BW denmiş. 1 Gbps = 1000 Mbps.
            cost += (1000.0 / bw)
    return cost

def calculate_weighted_cost(G, path, w_delay=0.33, w_rel=0.33, w_res=0.34):
    """Ağırlıklı Toplam Maliyet"""
    if not path:
        return float('inf')
        
    d = calculate_total_delay(G, path)
    r = calculate_reliability_cost(G, path)
    res = calculate_resource_cost(G, path)
    
    total_cost = (w_delay * d) + (w_rel * r) + (w_res * res)
    return total_cost
