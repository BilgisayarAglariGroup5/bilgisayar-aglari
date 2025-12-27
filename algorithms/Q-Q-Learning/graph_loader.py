import pandas as pd
import networkx as nx

def safe_float(value):
    """
    Gelen veriyi güvenli bir şekilde float'a çevirir.
    Virgül (,) varsa noktaya (.) dönüştürür.
    """
    try:
        val_str = str(value).replace(',', '.')
        return float(val_str)
    except (ValueError, TypeError):
        return 0.0

def load_graph(node_file, edge_file):
    print("Veriler yükleniyor...")

    # 1. DÜĞÜMLERİ OKU
    try:
        nodes_df = pd.read_csv(node_file, sep=';', encoding='utf-8-sig')
    except Exception as e:
        raise ValueError(f"Node dosyası okunamadı: {e}")

    # 2. KENARLARI OKU
    try:
        edges_df = pd.read_csv(edge_file, sep=';', encoding='utf-8-sig')
    except Exception as e:
        raise ValueError(f"Edge dosyası okunamadı: {e}")

    G = nx.Graph()

    # --- DÜĞÜMLERİ EKLEME (İndex bazlı: 0=ID, 1=Delay, 2=Reliability) ---
    # İsim ne olursa olsun sıraya göre alır.
    for i in range(len(nodes_df)):
        try:
            row = nodes_df.iloc[i]
            n_id = int(row.iloc[0])           # 1. Sütun: ID
            proc_delay = safe_float(row.iloc[1]) # 2. Sütun: İşlem Süresi
            reliability = safe_float(row.iloc[2]) # 3. Sütun: Güvenilirlik
            
            G.add_node(n_id, processing_delay=proc_delay, reliability=reliability)
        except Exception as e:
            print(f"Satır {i} okunurken hata (Node): {e}")

    # --- KENARLARI EKLEME (İndex bazlı: 0=Source, 1=Target, 2=BW, 3=Delay, 4=Rel) ---
    for i in range(len(edges_df)):
        try:
            row = edges_df.iloc[i]
            u = int(row.iloc[0])            # 1. Sütun: Kaynak
            v = int(row.iloc[1])            # 2. Sütun: Hedef
            bw = safe_float(row.iloc[2])    # 3. Sütun: Bant Genişliği
            delay = safe_float(row.iloc[3]) # 4. Sütun: Gecikme
            rel = safe_float(row.iloc[4])   # 5. Sütun: Güvenilirlik

            G.add_edge(u, v, bandwidth=bw, link_delay=delay, link_reliability=rel)
        except Exception as e:
            # Satır hatalıysa atla ama programı durdurma
            print(f"Satır {i} okunurken hata (Edge): {e}")
            continue

    print(f"Grafik BAŞARIYLA oluşturuldu: {G.number_of_nodes()} Düğüm, {G.number_of_edges()} Kenar.")
    
    # Sütun isimlerini de ekrana basalım ki neymiş görelim
    print(f"Düğüm Sütunları: {list(nodes_df.columns)}")
    print(f"Kenar Sütunları: {list(edges_df.columns)}")
    
    return G
