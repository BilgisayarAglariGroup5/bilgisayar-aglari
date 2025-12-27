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
# 1. Şu anki dosyanın yerini bul
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Proje ana dizinine çık (algorithms -> BILGISAYAR-AGLARI)
project_root = os.path.dirname(os.path.dirname(current_dir))

# 3. 'data' ve 'metrics' klasörlerini yola ekle
data_folder_path = os.path.join(project_root, 'data')
metrics_folder_path = os.path.join(project_root, 'metrics')

if data_folder_path not in sys.path:
    sys.path.append(data_folder_path)
    
if metrics_folder_path not in sys.path:
    sys.path.append(metrics_folder_path)

# --- IMPORTLAR ---
import network_topology 
# Arkadaşının metric.py dosyasını buradan çağırıyoruz
from metric import MetricsEngine, Weights 


# -------------------------------------------------------------------------------------------
# Adım - 1

# hedefe ulaşan random bir yol oluşturuyoruz
def generate_random_path(graph, src, dst):
    # graph(ağ) ,source(kaynak) ,destination(Hedef)

    """
    AMACI: Verilen haritada (graph) Başlangıçtan (src) Hedefe (dst) giden
    rastgele ama GEÇERLİ (kopuk olmayan) tek bir yol oluşturmaktır.
    
    ÇIKTI: [0, 2, 5, 10] gibi bir liste veya başarısız olursa None.
    """
    path = [src]       # 1. Yola başlangıç düğümüyle başla
    current_node = src # Şu an buradayız

    # Hedefe varana kadar döngü kuruyoruz
    # Şu anda bulunduğu konum hedefe ulaşasıya kadar döngü kur
    while current_node != dst:
        # a) Şu an bulunduğum düğümden nerelere gidebilirim?
        # (Eğer bulunduğum yer haritada yoksa dur)
        if current_node not in graph:
            return None 
            
        neighbors = graph[current_node] # Komşuları al (Örn: [1, 2, 5])

        # "Geriye dönmemek için sadece gitmediğim komşuları seç" mantığı
        
        possible_next_nodes = [] # Önce boş bir liste oluşturuyoruz
        
        for n in neighbors:      # Tüm komşuları tek tek geziyoruz
            if n not in path:    # Eğer bu komşu daha önce geçtiğimiz yolda YOKSA
                possible_next_nodes.append(n) # Listeye ekle (Burası gidilebilir bir yer)
        # c) Eğer gidecek hiçbir yer kalmadıysa (Çıkmaz Sokak)
        if not possible_next_nodes:
            return None # Bu deneme başarısız oldu, yolu iptal et.
        
        
        # d) Gidebileceğim yerlerden birini rastgele seç
        next_node = random.choice(possible_next_nodes)
        
        # Seçtiğim yeri yola ekle ve oraya git
        path.append(next_node)
        current_node = next_node
    
    # Döngü bittiyse hedefe varmışızdır, yolu döndür.
    return path


# ---------------------------------------------------------------------------------------------
# --- ADIM 2: İLK POPÜLASYONU (NESLİ) OLUŞTURMA ---

# Genetik algoritma için yarıştırabileceğimiz bir popülasyon tek bir yol ile karşılaştırma yapamayız.
# Burda generate_random_path fonksiyonunu kullanarak istediğimiz kadar rastgele yol oluşturuyoruz 
# Ve bu oluşturduğumuz bu rastgele yolları bir listede tutuyoruz.


def create_initial_population(graph, src, dst, pop_size):
    """
    AMACI: Belirtilen sayıda (pop_size) rastgele yol üretip bir liste (Popülasyon) döndürür.
    ÇIKTI: [[0,1,4], [0,2,3,4], [0,1,3,4]] gibi yollar listesi.
    """
    population = [] # Yolları biriktireceğimiz boş sepet
    
    attempts = 0    # Deneme sayacı
    # Sonsuz döngüden koruma kilidi:
    # Eğer harita çok zorsa ve yol bulunamıyorsa sonsuza kadar aramasın diye bir sınır koyuyoruz.
    # Örneğin 20 yol istediysek en fazla 200 kere denesin.
    max_attempts = pop_size * 10 
    
    # Sepet dolana kadar (istenen sayıya ulaşana kadar) döngü kur
    while len(population) < pop_size:
        
        # 1. Önceki yazdığımız fonksiyonu çağırıp BİR TANE yol iste
        path = generate_random_path(graph, src, dst)
        
        # 2. Eğer geçerli bir yol geldiyse (None değilse) sepete at
        if path is not None:
            population.append(path)
        
        # 3. Güvenlik kilidi kontrolü
        attempts = attempts + 1
        if attempts > max_attempts:
            print("Uyarı: İstenen sayıda yol bulunamadı, döngü durduruldu.")
            break # Döngüyü kır ve çık
    
    return population


# -------------------------------------------------------------------------------------------------


# --- ADIM 3: SEÇİM (SELECTION) - DÜŞÜK MALİYET İYİDİR ---
def selection(population, engine, num_parents, weights_obj):
    """
    Popülasyondaki yolları 'TotalCost' değerine göre sıralar.
    En DÜŞÜK maliyetli (en iyi) 'num_parents' kadar yolu seçer.
    
    GÜNCELLEME: Arkadaşının MetricsEngine'ini kullanıyor.
    """
    scored_paths = []
    
    # 1. Her yolun maliyetini hesapla
    for path in population:
        try:
            # Arkadaşının motorunu kullanarak hesapla
            stats = engine.compute(path)
            cost = engine.weighted_sum(stats, weights_obj)
            scored_paths.append((path, cost)) # (Yol, Maliyet) ikilisi
        except ValueError:
            # Eğer yol geçersizse (kopuksa vs) sonsuz maliyet ver
            scored_paths.append((path, float('inf')))
    
    # 2. Maliyete göre KÜÇÜKTEN BÜYÜĞE sırala (Çünkü az maliyet = iyi yol)
    scored_paths.sort(key=lambda x: x[1]) 
    
    # 3. En iyi (en düşük maliyetli) yolları seç
    selected_parents = []
    for i in range(num_parents):
        path = scored_paths[i][0] # Sadece yolu al
        selected_parents.append(path)
        
    return selected_parents


# -------------------------------------------------------------------------------------------------

# --- ADIM 4: ÇAPRAZLAMA (CROSSOVER) ---
def crossover(parent1, parent2):
    """
    İki ebeveyn yolu alır, ortak bir düğüm bulup parçalarını takas eder.
    """
    # 1. Ortak düğümleri bul (Başlangıç ve Bitiş hariç)
    
    # --- ESKİ TEK SATIRLIK KOD ---
    # common_nodes = [node for node in parent1 if node in parent2]
    
    # --- YENİ VE ANLAŞILIR KOD ---
    common_nodes = [] # Önce boş bir liste oluştur
    
    for node in parent1:      # Birinci ebeveynin duraklarını tek tek gez
        if node in parent2:   # Eğer bu durak İkinci ebeveynde de varsa
            common_nodes.append(node) # Listeye ekle (Ortak noktadır)

    # Başlangıç ve bitiş düğümlerini listeden çıkaralım
    # (Çünkü sadece baştan veya sondan kesersek yol değişmez)
    if len(common_nodes) > 2:
        # Listenin başını (0) ve sonunu (-1) at, arasını al
        candidates = common_nodes[1:-1]
    else:
        candidates = []
        
    # 2. Eğer kesişim noktası yoksa çaprazlama yapma, olduğu gibi geri döndür
    if not candidates:
        return parent1, parent2
        
    # 3. Rastgele bir kesişim noktası seç
    cut_node = random.choice(candidates)
    
    # 4. Parçaları Takas Et (.index burada kullanılıyor)
    idx1 = parent1.index(cut_node) # cut_node kaçıncı sırada?
    idx2 = parent2.index(cut_node)
    
    # Python'da 'slicing' (dilimleme) işlemi:
    # [:idx1+1] -> Baştan başla, o indise kadar (dahil) al.
    # [idx2+1:] -> O indisten sonraki her şeyi al.
    child1 = parent1[:idx1+1] + parent2[idx2+1:]
    child2 = parent2[:idx2+1] + parent1[idx1+1:]
    
    return child1, child2


# -------------------------------------------------------------------------------------------------
# --- ADIM 5: MUTASYON (MUTATION) ---
def mutation(path, graph, src, dst, mutation_rate):
    """
    Belirli bir olasılıkla yolun bir kısmını silip yeniden rastgele tamamlar.
    """
    # 1. Zar at: Mutasyon olacak mı?
    if random.random() > mutation_rate:
        return path # Olmadı, yolu değiştirmeden geri ver.
    
    # 2. Yol çok kısaysa (Sadece Başlangıç ve Bitiş varsa) kesip biçemeyiz.
    if len(path) < 3:
        return path
        
    # 3. Kesilecek noktayı seç (Başlangıç ve Bitiş hariç ara duraklar)
    # Örn: Yol [0, 1, 5, 4] ise, 1 veya 5'i seçebiliriz.
    cut_index = random.randint(1, len(path) - 2)
    cut_node = path[cut_index]
    
    # 4. Eski yolun baş tarafını al
    # Örn: Yol [0, 1, 5, 4] ve kesilen yer 1 ise -> [0, 1] elimizde kalır.
    partial_path = path[:cut_index + 1]
    
    # 5. Kesilen noktadan hedefe giden YENİ bir yol bul
    # (Adım 1'deki fonksiyonumuzu kullanıyoruz)
    new_tail = generate_random_path(graph, cut_node, dst)
    
    # Eğer kör kuyuya girdiyse ve yeni yol bulamadıysa, eski yolu bozma.
    if new_tail is None:
        return path
        
    # 6. Parçaları Birleştir
    # partial_path [0, 1] ile bitiyor.
    # new_tail [1, 8, 4] ile başlıyor.
    # 1 tekrar etmesin diye partial_path'in sonunu almıyoruz.
    final_path = partial_path[:-1] + new_tail
    
    return final_path


# -------------------------------------------------------------------------------------------------
# --- ADIM 6: ANA GENETİK ALGORİTMA FONKSİYONU ---
def run_genetic_algorithm(graph, src, dst, config):
    """
    Genetik Algoritmayı çalıştıran ana fonksiyon.
    config: Ayarları içeren bir sözlük (pop_size, generations, weights vb.)
    """
    # Ayarları çekelim
    pop_size = config['pop_size']
    generations = config['generations']
    mutation_rate = config['mutation_rate']
    
    # --- YENİ KISIM: Arkadaşının Motorunu Hazırlama ---
    w_tuple = config['weights']
    # Senin tuple'ını (0.4, 0.4, 0.2) arkadaşının class yapısına çeviriyoruz
    weights_obj = Weights(w_delay=w_tuple[0], w_reliability=w_tuple[1], w_resource=w_tuple[2])
    
    # Motoru bir kere başlat, hep kullan (graph nesnesini veriyoruz)
    engine = MetricsEngine(graph)
    # -------------------------------------------------

    # 1. İlk Popülasyonu Oluştur
    population = create_initial_population(graph, src, dst, pop_size)
    
    # En iyi çözümü saklamak için değişkenler
    best_path = None
    best_cost = float('inf') # Sonsuz ile başlatıyoruz (Küçük olan kazanır)

    # 2. Nesil Döngüsü (Evolution Loop)
    for gen in range(generations):
        
        # A) Seçim (En iyi %50'yi ebeveyn olarak seçelim)
        # Ebeveyn sayısı çift olmalı ki çaprazlama yapabilelim
        num_parents = pop_size // 2 
        
        # --- DEĞİŞİKLİK: Selection artık engine ve weights_obj alıyor ---
        parents = selection(population, engine, num_parents, weights_obj)
        
        # Eğer hiç ebeveyn seçilemediyse (yol bulunamadıysa) döngüyü kır
        if not parents:
            print("Uyarı: Geçerli yol bulunamadı!")
            break
            
        # O neslin en iyisini bulup kaydedelim (Raporlama için önemli)
        current_best_path = parents[0] # Selection zaten sıralı veriyor
        
        # --- DEĞİŞİKLİK: Maliyeti arkadaşının motoruyla hesapla ---
        try:
            stats = engine.compute(current_best_path)
            current_best_cost = engine.weighted_sum(stats, weights_obj)
            
            if current_best_cost < best_cost:
                best_cost = current_best_cost
                best_path = current_best_path
        except:
            continue

        # B) Yeni Nesil Oluşturma (Çaprazlama ve Mutasyon)
        new_population = []
        
        # Seçilen ebeveynleri (Elitizm) doğrudan yeni nesle ekleyebiliriz (İsteğe bağlı)
        # Biz çeşitlilik olsun diye sadece çocukları ekleyelim.
        
        while len(new_population) < pop_size:
            # Rastgele iki ebeveyn seç
            p1 = random.choice(parents)
            p2 = random.choice(parents)
            
            # Çaprazlama
            c1, c2 = crossover(p1, p2)
            
            # Mutasyon
            c1 = mutation(c1, graph, src, dst, mutation_rate)
            c2 = mutation(c2, graph, src, dst, mutation_rate)
            
            new_population.append(c1)
            # Eğer popülasyon dolduysa ikinci çocuğu ekleme
            if len(new_population) < pop_size:
                new_population.append(c2)
        
        # C) Nüfus Değişimi: Eski nesil öldü, yaşasın yeni nesil!
        population = new_population

    # Döngü bitti, elimizdeki (veya tarihçedeki) en iyi yolu döndür
    return best_path, best_cost


if __name__ == "__main__":
    
    print("--- 1. Topoloji Yükleniyor ---")
    # Arkadaşının fonksiyonlarını kullanarak gerçek bir ağ yaratıyoruz
    G = network_topology.generate_connected_topology(
        n=250, 
        p=0.4, 
        max_attempts=100
    )
    network_topology.assign_node_attributes(G)
    network_topology.assign_edge_attributes(G)
    
    # --- ÖNEMLİ VERİ DÜZELTMESİ (Mapping) ---
    # Arkadaşının metric.py dosyası "capacity_mbps" ismini arıyor.
    # Ancak topology.py dosyası "bandwidth_mbps" ismini üretiyor.
    # Kod patlamasın diye veriyi kopyalıyoruz:
    for u, v, data in G.edges(data=True):
        if 'bandwidth_mbps' in data:
            data['capacity_mbps'] = data['bandwidth_mbps']
    # ----------------------------------------
    
    # Rastgele bir talep (Source-Destination) seçelim
    demands = network_topology.generate_demands(G, num_demands=5)
    selected_demand = demands[0] # İlk talebi alalım
    src_node = selected_demand[0]
    dst_node = selected_demand[1]
    
    print(f"\n--- 2. ADIM: Genetik Algoritma Çalışıyor ---")
    print(f"Hedef: {src_node} -> {dst_node} (Talep edilen bant genişliği: {selected_demand[2]} Mbps)")
    
    # Konfigürasyon
    config = {
        'pop_size': 50,           
        'generations': 50,        
        'mutation_rate': 0.1,     
        'weights': (0.4, 0.4, 0.2) 
    }
    
    # Algoritmayı Çalıştır (Artık G nesnesini veriyoruz!)
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