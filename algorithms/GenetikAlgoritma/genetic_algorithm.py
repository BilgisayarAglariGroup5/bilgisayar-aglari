# Metin Öztaş
# node lar oluşturulacak sonra metrikler hesaplanıcak sonra o metrikleri algoritmada kullanacağız.
# genetic algoritma metriklere ihtiyaç duyar 
# genetik algoritma "Evrim teorisine" dayanır . Güçlü olan hayatta kalır.
# Algoritma Rastgele 100 tane yol üretti.
#    Bu 100 yoldan hangileri "anne/baba" olup bir sonraki nesli oluşturacak? Hangileri "çöp" olup silinecek?
#    Metrikler bu algoritmanın pusulası olucak
# En sonunda bulduğum en iyi yolun metriklerini tekrar hesaplattırıp arayüzde yazılacak

import random
import time
import os  
import sys
import networkx as nx 

# --- PATH AYARLAMALARI (Dinamik) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

data_folder_path = os.path.join(project_root, 'data')
metrics_folder_path = os.path.join(project_root, 'metrics')

if data_folder_path not in sys.path:
    sys.path.append(data_folder_path)
    
if metrics_folder_path not in sys.path:
    sys.path.append(metrics_folder_path)

# --- IMPORTLAR ---
import network_topology 
from metric import MetricsEngine, Weights 

# -------------------------------------------------------------------------------------------
# (BURADAKİ FONKSİYONLARIN AYNI KALIYOR: generate_random_path, create_initial_population, 
# selection, crossover, mutation, run_genetic_algorithm)
# KOD KALABALIKLIĞI OLMASIN DİYE TEKRAR YAZMIYORUM, YUKARIDAKİ GİBİ KALSINLAR.
# -------------------------------------------------------------------------------------------

# ... (Algoritma fonksiyonların burada duruyor) ...


# -------------------------------------------------------------------------------------------
# Adım - 1

# hedefe ulaşan random bir yol oluşturuyoruz
def generate_random_path(graph, src, dst):
    path = [src]       # 1. Yola başlangıç düğümüyle başla
    current_node = src # Şu an buradayız

    while current_node != dst:
        if current_node not in graph:
            return None 
            
        neighbors = graph[current_node] 
        possible_next_nodes = [] 
        
        for n in neighbors:      
            if n not in path:    
                possible_next_nodes.append(n) 
        if not possible_next_nodes:
            return None 
        
        next_node = random.choice(possible_next_nodes)
        path.append(next_node)
        current_node = next_node
    
    return path


# ---------------------------------------------------------------------------------------------
# --- ADIM 2: İLK POPÜLASYONU (NESLİ) OLUŞTURMA ---

def create_initial_population(graph, src, dst, pop_size):
    population = [] 
    attempts = 0    
    max_attempts = pop_size * 10 
    
    while len(population) < pop_size:
        path = generate_random_path(graph, src, dst)
        if path is not None:
            population.append(path)
        attempts = attempts + 1
        if attempts > max_attempts:
            print("Uyarı: İstenen sayıda yol bulunamadı, döngü durduruldu.")
            break 
    return population


# -------------------------------------------------------------------------------------------------
# --- ADIM 3: SEÇİM (SELECTION) ---

def selection(population, engine, num_parents, weights_obj):
    scored_paths = []
    for path in population:
        try:
            stats = engine.compute(path)
            cost = engine.weighted_sum(stats, weights_obj)
            scored_paths.append((path, cost))
        except ValueError:
            scored_paths.append((path, float('inf')))
    
    scored_paths.sort(key=lambda x: x[1]) 
    selected_parents = []
    for i in range(num_parents):
        path = scored_paths[i][0]
        selected_parents.append(path)
    return selected_parents


# -------------------------------------------------------------------------------------------------
# --- ADIM 4: ÇAPRAZLAMA (CROSSOVER) ---

def crossover(parent1, parent2):
    common_nodes = [] 
    for node in parent1:      
        if node in parent2:   
            common_nodes.append(node) 

    if len(common_nodes) > 2:
        candidates = common_nodes[1:-1]
    else:
        candidates = []
        
    if not candidates:
        return parent1, parent2
        
    cut_node = random.choice(candidates)
    idx1 = parent1.index(cut_node)
    idx2 = parent2.index(cut_node)
    
    child1 = parent1[:idx1+1] + parent2[idx2+1:]
    child2 = parent2[:idx2+1] + parent1[idx1+1:]
    
    return child1, child2


# -------------------------------------------------------------------------------------------------
# --- ADIM 5: MUTASYON (MUTATION) ---

def mutation(path, graph, src, dst, mutation_rate):
    if random.random() > mutation_rate:
        return path 
    if len(path) < 3:
        return path
    cut_index = random.randint(1, len(path) - 2)
    cut_node = path[cut_index]
    partial_path = path[:cut_index + 1]
    new_tail = generate_random_path(graph, cut_node, dst)
    if new_tail is None:
        return path
    final_path = partial_path[:-1] + new_tail
    return final_path


# -------------------------------------------------------------------------------------------------
# --- ADIM 6: ANA GENETİK ALGORİTMA FONKSİYONU ---

def run_genetic_algorithm(graph, src, dst, config):
    # --- GÜVENLİK YAMASI: İsim uyuşmazlığını otomatik düzelt ---
    # Arayüzden gelen ham veriyi (s_ms, delay_ms vb.) bizim anlayacağımız dile çevirir.
    for n, data in graph.nodes(data=True):
        if 's_ms' in data and 'processing_delay_ms' not in data: 
            data['processing_delay_ms'] = data['s_ms']
        if 'r_node' in data and 'node_reliability' not in data: 
            data['node_reliability'] = data['r_node']

    for u, v, data in graph.edges(data=True):
        if 'delay_ms' in data and 'link_delay_ms' not in data: 
            data['link_delay_ms'] = data['delay_ms']
        if 'r_link' in data and 'link_reliability' not in data: 
            data['link_reliability'] = data['r_link']
        if 'capacity_mbps' in data and 'capacity_mbps' not in data: # Zaten aynı ama garanti olsun
             data['capacity_mbps'] = data['capacity_mbps']
    pop_size = config['pop_size']
    generations = config['generations']
    mutation_rate = config['mutation_rate']
    
    w_tuple = config['weights']
    weights_obj = Weights(w_delay=w_tuple[0], w_reliability=w_tuple[1], w_resource=w_tuple[2])
    
    engine = MetricsEngine(graph)

    population = create_initial_population(graph, src, dst, pop_size)
    best_path = None
    best_cost = float('inf') 

    for gen in range(generations):
        num_parents = pop_size // 2 
        parents = selection(population, engine, num_parents, weights_obj)
        
        if not parents:
            print("Uyarı: Geçerli yol bulunamadı!")
            break
            
        current_best_path = parents[0] 
        
        try:
            stats = engine.compute(current_best_path)
            current_best_cost = engine.weighted_sum(stats, weights_obj)
            
            if current_best_cost < best_cost:
                best_cost = current_best_cost
                best_path = current_best_path
        except:
            continue

        new_population = []
        while len(new_population) < pop_size:
            p1 = random.choice(parents)
            p2 = random.choice(parents)
            c1, c2 = crossover(p1, p2)
            c1 = mutation(c1, graph, src, dst, mutation_rate)
            c2 = mutation(c2, graph, src, dst, mutation_rate)
            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)
        population = new_population

    return best_path, best_cost


# =========================================================================
# GÜNCELLENMİŞ MAIN BLOĞU (ARKADAŞININ YENİ KODUNA UYGUN)
# =========================================================================
if __name__ == "__main__":
    
    print("--- 1. Topoloji CSV Dosyalarından Yükleniyor ---")
    
    # CSV dosyalarının tam yolunu belirtmemiz lazım
    node_csv_path = os.path.join(data_folder_path, "NodeData.csv")
    edge_csv_path = os.path.join(data_folder_path, "EdgeData.csv")
    
    # generate_connected_topology YERİNE load_topology kullanıyoruz
    # Bu fonksiyon 3 değer döndürüyor: G, node_df, edge_df
    try:
        G, node_df, edge_df = network_topology.load_topology(
            node_csv=node_csv_path, 
            edge_csv=edge_csv_path
        )
    except FileNotFoundError:
        print("\nHATA: NodeData.csv veya EdgeData.csv dosyaları 'data' klasöründe bulunamadı!")
        print(f"Aranan yer: {data_folder_path}")
        sys.exit()

    # --- KRİTİK ADIM: İSİM EŞLEŞTİRME (MAPPING) ---
    # Arkadaşının CSV'den okuduğu isimler -> Melek'in Motorunun beklediği isimler
    # Node: s_ms -> processing_delay_ms, r_node -> node_reliability
    # Edge: delay_ms -> link_delay_ms, r_link -> link_reliability, capacity_mbps -> capacity_mbps
    
    for n, data in G.nodes(data=True):
        if 's_ms' in data: data['processing_delay_ms'] = data['s_ms']
        if 'r_node' in data: data['node_reliability'] = data['r_node']

    for u, v, data in G.edges(data=True):
        if 'delay_ms' in data: data['link_delay_ms'] = data['delay_ms']
        if 'r_link' in data: data['link_reliability'] = data['r_link']
        # capacity_mbps zaten aynı isimde, kopyalamaya gerek yok ama garanti olsun:
        if 'capacity_mbps' in data: data['capacity_mbps'] = data['capacity_mbps']
        
    print("--- Veri Eşleştirmesi Tamamlandı ---")

    # --- RASTGELE BİR TALEP SEÇELİM ---
    # generate_demands silindiği için grafikten rastgele iki nokta seçiyoruz
    nodes_list = list(G.nodes())
    if len(nodes_list) < 2:
        print("Hata: Grafik yeterli düğüme sahip değil.")
        sys.exit()
        
    src_node = random.choice(nodes_list)
    dst_node = random.choice(nodes_list)
    while dst_node == src_node:
        dst_node = random.choice(nodes_list)

    # Test için rastgele bir bant genişliği isteği uyduralım
    demand_mbps = 50 
    
    print(f"\n--- 2. ADIM: Genetik Algoritma Çalışıyor ---")
    print(f"Hedef: {src_node} -> {dst_node} (Talep: {demand_mbps} Mbps)")
    
    # Konfigürasyon
    config = {
        'pop_size': 50,           
        'generations': 50,        
        'mutation_rate': 0.1,     
        'weights': (0.4, 0.4, 0.2) 
    }
    
    # Algoritmayı Çalıştır
    baslangic_zamani = time.time()
    
    en_iyi_yol, en_iyi_maliyet = run_genetic_algorithm(G, src_node, dst_node, config)
    
    bitis_zamani = time.time()
    gecen_sure = bitis_zamani - baslangic_zamani
    
    # Sonuç Raporu
    if en_iyi_yol:
        # Sonuçları detaylandırmak için arkadaşının motorunu burada da kullanıyoruz
        engine = MetricsEngine(G)
        detaylar = engine.compute(en_iyi_yol)
        
        print("\n" + "="*50)
        print("          EN İYİ YOL BULUNDU (SONUÇ)")
        print("="*50)
        print(f"Rota: {en_iyi_yol}")
        print("-" * 50)
        
        print(f"Toplam Skor (Weighted Cost): {en_iyi_maliyet:.4f}")
        print(f"Çalışma Süresi             : {gecen_sure:.4f} saniye")
        print("-" * 50)
        print(f"METRİK DETAYLARI:")
        print(f" > Toplam Gecikme      : {detaylar.total_delay_ms:.2f} ms")
        print(f" > Gerçek Güvenilirlik : %{detaylar.total_reliability*100:.4f}")
        print(f" > Kaynak Maliyeti     : {detaylar.resource_cost:.2f}")
        print(f" > Darboğaz Kapasite   : {detaylar.bottleneck_capacity_mbps} Mbps")
        print("="*50)
    else:
        print("\nSonuç: Hiçbir yol bulunamadı!")