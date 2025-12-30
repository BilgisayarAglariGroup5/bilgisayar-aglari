"""
ACO Tabanlı Ağ Optimizasyonu – Ana Çalıştırma Modülü
---------------------------------------------------
PROJE AMACI: Karınca Kolonisi Optimizasyonu kullanarak, karmaşık bir ağ topolojisi 
üzerinde gecikme (latency), güvenilirlik (reliability) ve bant genişliği (bandwidth) 
kriterlerini aynı anda optimize eden "en iyi yolu" bulmaktır.
"""

import sys
import os
import networkx as nx
import math
import random

# --- PROJE DİZİN YAPILANDIRMASI ---
# Bu bölüm, projenin farklı klasörlerdeki (data, metrics vb.) modüllerine erişebilmesini sağlar.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
METRICS_DIR = os.path.join(PROJECT_ROOT, "metrics")

# Python'un import arama listesine bu dizinleri ekleyerek 'custom' modülleri görünür kılıyoruz.
if DATA_DIR not in sys.path:
    sys.path.append(DATA_DIR)
if METRICS_DIR not in sys.path:
    sys.path.append(METRICS_DIR)
print("DEBUG DATA_DIR:", DATA_DIR)
print("DEBUG data/topology içeriği:", os.listdir(DATA_DIR))


# Gerekli özel motorların ve karınca çekirdek fonksiyonlarının içe aktarılması.
import network_topology
from metric import MetricsEngine, Weights
from aco_core import ant_walk, update_pheromone, evaporate_pheromone

node_csv_path = os.path.join(DATA_DIR, "NodeData.csv")
edge_csv_path = os.path.join(DATA_DIR, "EdgeData.csv")

def run_aco_main(
    num_iterations=20,     # Algoritmanın toplam kaç nesil/tur çalışacağı.
    num_ants=15,           # Her bir turda ağa salınan yapay karınca sayısı.
    initial_pheromone=0.1, # Başlangıçta tüm yolların (edge) sahip olduğu feromon miktarı.
    rho=0.1,               # Buharlaşma katsayısı; feromonların her turda ne kadarının silineceği.
    Q=10.0                 # Feromon yoğunluk sabiti; başarılı karıncanın bırakacağı iz gücü.
):
    """
    Optimizasyon sürecini yürüten ana motor fonksiyonu.
    """

    # --- 1. ADIM: AĞ TOPOLOJİSİNİN İNŞASI ---
    G = nx.Graph()
    print("[INFO] Topoloji olusturuluyor...")

    # n=50: Düğüm sayısı, p=0.3: Her iki düğümün birbirine bağlanma olasılığı (Yoğunluk).
    G, node_df, edge_df = network_topology.load_topology(
        node_csv=node_csv_path,
        edge_csv=edge_csv_path
)
    # --- 2. ADIM: RASTGELE KAYNAK / HEDEF / TALEP --- 
    # # Eğer kullanıcı değer girmezse rastgele düğümler ve rastgele bant genişliği seçilir 
    # 
    nodes = list(G.nodes()) 
    src, dst = random.sample(nodes, 2) 
    # Aynı düğüm seçilmesin diye sample kullanıyoruz
    bw = random.choice([10, 20, 50, 100]) # Mbps, örnek seçenekler

        


    # Teknik uyumluluk için bant genişliği verileri 'capacity' etiketiyle de kopyalanır.
    for u, v, data in G.edges(data=True):
        if "bandwidth_mbps" in data:
            data["capacity_mbps"] = data["bandwidth_mbps"]

    # --- 3. ADIM: OPTİMİZASYON KRİTERLERİ VE AĞIRLIKLANDIRMA ---
    metrics_engine = MetricsEngine(G)
    # MetricsEngine: Yolun kalitesini (gecikme, güvenilirlik, maliyet) hesaplayan matematiksel motor.
    # -------------------------------------------------
# METRICS ENGINE UYUMLULUK YAMASI (ZORUNLU)

    for n, data in G.nodes(data=True):
        if 's_ms' in data and 'processing_delay_ms' not in data:
            data['processing_delay_ms'] = data['s_ms']
        if 'r_node' in data and 'node_reliability' not in data:
            data['node_reliability'] = data['r_node']

    for u, v, data in G.edges(data=True):
        if 'delay_ms' in data and 'link_delay_ms' not in data:
            data['link_delay_ms'] = data['delay_ms']
        if 'r_link' in data and 'link_reliability' not in data:
            data['link_reliability'] = data['r_link']
        if 'bandwidth_mbps' in data and 'capacity_mbps' not in data:
            data['capacity_mbps'] = data['bandwidth_mbps']

    print("[INFO] Metric uyumluluk eslestirmesi tamamlandi")
# -------------------------------------------------


    # Weights: Kullanıcının önceliğini belirler. Burada gecikme ve güvenilirlik en ön planda.
    weights = Weights(w_delay=0.4, w_reliability=0.4, w_resource=0.2)

    print("\n" + "=" * 60)
    print("      ACO TABANLI AG OPTIMIZASYON SURECI")
    print("=" * 60)

    # --- 4. ADIM: SEZGİSEL (HEURISTIC) KARAR MEKANİZMASI ---
    def heuristic_from_metrics(G, u, v):
        """
        Karıncanın bir sonraki düğüme karar verirken kullandığı 'görüş yeteneği'.
        Sadece feromona değil, o yolun anlık maliyetine (gecikme/güvenilirlik) bakar.
        Maliyet ne kadar düşükse, sonuç (1/cost) o kadar yüksek ve cazip olur.
        """
        temp_path = [u, v]
        metrics = metrics_engine.compute(temp_path)
        cost = metrics_engine.weighted_sum(metrics, weights)
        return 1.0 / max(cost, 1e-6) # Sıfıra bölme hatasını engellemek için küçük bir epsilon.

    # --- 5. ADIM: ALGORİTMA BAŞLATMA VE FEROMON KURULUMU ---
    # Başlangıçta tüm yollar karıncalar için eşit cazibededir.
    for u, v in G.edges():
        G[u][v]["pheromone"] = initial_pheromone

    best_path = None
    min_cost = float("inf") # En iyi (en düşük) maliyeti takip etmek için başlangıç değeri.

    print(f"\n[TALEP] {src} -> {dst} | Bant Gen.: {bw} Mbps")
    print("-" * 50)

    # --- 6. ADIM: İTERASYON DÖNGÜSÜ (ÖĞRENME SÜRECİ) ---
    for i in range(num_iterations):
        iteration_paths = []

        # Her bir karınca kaynaktan hedefe ulaşmaya çalışır.
        for _ in range(num_ants):
            path = ant_walk(G, src, dst, bw, heuristic_from_metrics)

            if path:
                # Bulunan yolun performans metrikleri (toplam ms, paket kaybı riski vb.) hesaplanır.
                metrics = metrics_engine.compute(path, demand_mbps=bw)
                cost = metrics_engine.weighted_sum(metrics, weights)

                iteration_paths.append((path, cost))

                # Global en iyi yolu güncelleme kontrolü.
                if cost < min_cost:
                    min_cost = cost
                    best_path = path

        # --- 7. ADIM: FEROMON GÜNCELLEME (HAFIZA VE UNUTMA) ---
        # 1. Buharlaşma: Kimse tarafından kullanılmayan yolların feromonu azalır (Unutma).
        evaporate_pheromone(G, rho)


        # 2. Takviye: Başarılı yollardan geçen karıncalar o yolu feromonla işaretler (Öğrenme).
        for p, c in iteration_paths:
            update_pheromone(G, p, c, Q)

    # --- 8. ADIM: FİNAL RAPORLAMA ---
    if best_path:
        # En iyi bulunan yolun son kez tüm detaylarıyla analiz edilmesi.
        final_metrics = metrics_engine.compute(best_path, demand_mbps=bw)
        print("[BASARILI]")
        print("En Iyi Yol:", " -> ".join(map(str, best_path)))
        print("-" * 50)
        print(f"Toplam Gecikme      : {final_metrics.total_delay_ms:.2f} ms") # Düşük olması istenir.
        print(f"Gerçek Güvenilirlik : %{final_metrics.total_reliability * 100:.4f}") # Yüksek olması istenir.
        print(f"Kaynak Maliyeti     : {final_metrics.resource_cost:.2f}") # Verimlilik göstergesi.
        print(f"Darboğaz Kapasite   : {final_metrics.bottleneck_capacity_mbps} Mbps") # Yolun en zayıf halkası.
        print("-" * 50)
        print(f"Genel Kalite Skoru  : {min_cost:.6f}") # Ağırlıklı maliyet toplamı.
    else:
        print("[BASARISIZ] Hedefe ulasan bir yol bulunamadi. Ağ kopuk veya kapasite yetersiz olabilir.")

# Script doğrudan çalıştırıldığında optimizasyonu başlatır.
if __name__ == "__main__":
    run_aco_main()