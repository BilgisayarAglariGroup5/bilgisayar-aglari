"""
QoS Odaklı Rotalama Simülasyonu - Streamlit Arayüzü

Bu script, kullanıcının belirttiği gereksinimlere tam olarak uyan, modüler ve
çalışır bir Streamlit arayüzü sunar. Algoritma hesaplama kısmı, ekip tarafından
doldurulmak üzere bir placeholder olarak bırakılmıştır.

Özellikler:
- 250 düğümlü Erdos-Renyi grafiği oluşturma
- Kaynak (S) ve Hedef (D) seçimi
- QoS ağırlıklarının (Gecikme, Güvenilirlik, Kaynak) slider ile ayarlanması
- Ağırlıkların toplamda 1'e normalizasyonu (isteğe bağlı)
- Talep edilen bant genişliği girişi
- Açık/Koyu tema seçeneği
- PyVis ile dinamik ağ görselleştirmesi (düğüm ID'leri, özel renklendirme, hover bilgileri)
- Hesaplama sonrası metriklerin gösterimi
- Sonuçların DataFrame tabloları ve CSV indirme seçenekleri ile sunulması
"""

import os
import tempfile
import random
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network

# --- UYGULAMA SABİTLERİ VE KONFİGÜRASYON ---
NODE_COUNT = 250
ERDOS_RENYI_P = 0.02  # Düğümler arası bağlantı olasılığı

# Sayfa ayarlarını başta bir kez yap
st.set_page_config(page_title="QoS Odaklı Rotalama Simülasyonu", layout="wide")

# --- ALGORİTMA ENTEGRASYON NOKTASI (PLACEHOLDER) ---
def compute_path(
    graph: nx.Graph,
    source: int,
    target: int,
    w_delay: float,
    w_rel: float,
    w_res: float,
    requested_bw: float,
    extra_opts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    BU FONKSİYON TAKIM TARAFINDAN DOLDURULACAK.
    Belirtilen QoS metriklerine göre en uygun yolu hesaplar.

    Dönüş Formatı (Sözleşme):
    {
        "path": [0, 5, 7, 20],  // Bulunan yol (düğüm listesi)
        "edges": [{...}],  // Yol üzerindeki kenarların detayları
        "metrics": {
            "total_delay_ms": ...,
            "path_reliability": ...,
            "reliability_cost": ...,
            "resource_cost": ...,
            "total_cost": ...,
            "min_bandwidth_on_path": ...,
            "meets_requested_bw": True/False
        },
        "per_node": [...], // Yol üzerindeki düğümlerin detayları
        "per_edge": [...], // Yol üzerindeki kenarların detayları (DataFrame için)
        "notes": "..."   // Algoritmadan ek notlar
    }
    """
    # --- ÖRNEK/MOCK HESAPLAMA ---
    # Bu kısım gerçek bir QoS hesaplaması yapmaz. Sadece en kısa yolu bulur
    # ve arayüzün çalışması için gerekli formatta sahte veriler üretir.
    # Ekibiniz kendi algoritmasını bu fonksiyonun içine entegre edecektir.

    try:
        # Sadece hop sayısına göre en kısa yolu bul (Dijkstra/A* gibi bir algoritma değil)
        path = nx.shortest_path(graph, source=source, target=target)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        # Yol bulunamazsa, boş bir sonuç döndür
        return {
            "path": [], "edges": [], "metrics": {}, "per_node": [], "per_edge": [],
            "notes": "Kaynak ve hedef arasında bir yol bulunamadı."
        }

    # Yol metriklerini hesapla (bu kısım da mock)
    edges_info, total_delay, reliabilities, bandwidths = [], 0.0, [], []
    resource_cost = sum(graph.nodes[n].get('resource_cost', 0) for n in path)

    for u, v in zip(path, path[1:]):
        edge_data = graph.get_edge_data(u, v, default={})
        delay = edge_data.get("delay_ms", 0.0)
        bw = edge_data.get("bandwidth_mbps", 0.0)
        rel = edge_data.get("reliability", 1.0)

        edges_info.append({
            "u": u, "v": v, "delay_ms": delay,
            "bandwidth_mbps": bw, "reliability": rel
        })
        total_delay += delay
        reliabilities.append(rel)
        bandwidths.append(bw)

    min_bw_on_path = min(bandwidths) if bandwidths else 0.0
    path_reliability = np.prod(reliabilities) if reliabilities else 1.0
    reliability_cost = 1.0 - path_reliability
    total_cost = (w_delay * total_delay) + (w_rel * reliability_cost) + (w_res * resource_cost)

    return {
        "path": path,
        "edges": edges_info,
        "metrics": {
            "total_delay_ms": total_delay,
            "path_reliability": path_reliability,
            "reliability_cost": reliability_cost,
            "resource_cost": resource_cost,
            "total_cost": total_cost,
            "min_bandwidth_on_path": min_bw_on_path,
            "meets_requested_bw": min_bw_on_path >= requested_bw,
        },
        "per_node": [{"düğüm": n, **graph.nodes[n]} for n in path],
        "per_edge": edges_info,
        "notes": "Bu sonuç, gerçek QoS algoritması yerine en kısa yol (hop) kullanılarak üretilmiştir.",
    }


# --- YARDIMCI FONKSİYONLAR ---
@st.cache_data
def generate_er_graph(n: int, p: float, seed: Optional[int] = None) -> nx.Graph:
    """
    Rastgele kenar özniteliklerine sahip bir Erdos-Renyi grafiği oluşturur.
    Streamlit'in cache mekanizması sayesinde aynı seed ile tekrar tekrar üretilmez.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    G = nx.erdos_renyi_graph(n=n, p=p, seed=seed)

    # Düğümlere ve kenarlara rastgele metrikler ata
    for node in G.nodes():
        G.nodes[node]["resource_cost"] = float(np.random.uniform(1.0, 20.0))

    for u, v in G.edges():
        G.edges[u, v]["delay_ms"] = float(np.round(np.random.uniform(1.0, 50.0), 2))
        G.edges[u, v]["bandwidth_mbps"] = float(np.round(random.choice([10, 50, 100, 250, 500, 1000])))
        G.edges[u, v]["reliability"] = float(np.round(np.random.uniform(0.95, 1.0), 4))

    return G

def draw_pyvis_network(
    graph: nx.Graph,
    path: List[int],
    source: int,
    target: int,
    theme: str = "Açık Tema"
) -> str:
    """PyVis kullanarak ağı görselleştirir ve geçici bir HTML dosyası olarak döndürür."""
    is_dark = theme == "Koyu Tema"
    bgcolor = "#1E1E1E" if is_dark else "#FFFFFF"
    font_color = "#FFFFFF" if is_dark else "#333333"
    edge_color = "#444444" if is_dark else "#CCCCCC"
    path_color = "#E74C3C"  # Kırmızı

    net = Network(height="700px", width="100%", bgcolor=bgcolor, font_color=font_color, notebook=True)
    net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=250, spring_strength=0.04)

    path_edges = set(zip(path, path[1:]))

    for n in graph.nodes():
        node_title = f"Düğüm {n}<br>Kaynak Maliyeti: {graph.nodes[n].get('resource_cost', 0):.2f}"
        color = "#97C2FC"  # Mavi (varsayılan)
        size = 15

        if n in path:
            color, size = path_color, 22
        if n == source:
            color, size = "#2ECC71", 30  # Yeşil
        if n == target:
            color, size = path_color, 30

        net.add_node(n, label=str(n), title=node_title, color=color, size=size)

    for u, v, data in graph.edges(data=True):
        edge_title = (
            f"Kenar: {u}-{v}<br>"
            f"Gecikme: {data.get('delay_ms', 0):.2f} ms<br>"
            f"Bant Genişliği: {data.get('bandwidth_mbps', 0)} Mbps<br>"
            f"Güvenilirlik: {data.get('reliability', 0):.4f}"
        )
        is_path_edge = (u, v) in path_edges or (v, u) in path_edges
        net.add_edge(
            u, v,
            title=edge_title,
            color=path_color if is_path_edge else edge_color,
            width=3 if is_path_edge else 1
        )

    # Grafiği geçici bir dosyaya yaz ve yolunu döndür
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        return tmp_file.name

# --- STREAMLIT ARAYÜZÜ ---
def main():
    """Ana Streamlit uygulama fonksiyonu."""
    
    st.title("📡 QoS Odaklı Rotalama Simülasyonu")

    # --- YAN PANEL (SIDEBAR) ---
    with st.sidebar:
        st.header("⚙️ Simülasyon Ayarları")
        
        seed = st.number_input("Rastgele Grafik Seed'i", value=42, help="Aynı seed, aynı ağ yapısını üretir.")
        
        st.subheader("Kaynak (S) ve Hedef (D)")
        nodes = list(range(NODE_COUNT))
        source = st.selectbox("Kaynak Düğüm (S)", nodes, index=10)
        target = st.selectbox("Hedef Düğüm (D)", nodes, index=240)

        st.subheader("QoS Ağırlıkları (W)")
        w_delay = st.slider("Gecikme Ağırlığı (W_delay)", 0.0, 1.0, 0.33)
        w_rel = st.slider("Güvenilirlik Ağırlığı (W_rel)", 0.0, 1.0, 0.33)
        w_res = st.slider("Kaynak Ağırlığı (W_res)", 0.0, 1.0, 0.34)

        if st.checkbox("Ağırlıkları Otomatik Normalleştir (Toplam=1)", value=True):
            total_w = w_delay + w_rel + w_res + 1e-9 # Sıfıra bölme hatasını önle
            w_delay, w_rel, w_res = w_delay / total_w, w_rel / total_w, w_res / total_w
            st.info(f"Normalleştirilmiş: G={w_delay:.2f}, Gv={w_rel:.2f}, K={w_res:.2f}")

        st.subheader("Ağ Talebi")
        requested_bw = st.number_input("Talep Edilen Bant Genişliği (Mbps)", min_value=1.0, max_value=10000.0, value=50.0)

        st.subheader("Görünüm")
        theme = st.selectbox("Tema Seçimi", ["Açık Tema", "Koyu Tema"])
        
        st.markdown("---")
        run_button = st.button("🚀 Rotalamayı Hesapla")

    # --- ANA İÇERİK ---
    
    # Grafiği oluştur (cache sayesinde sadece seed değiştiğinde yeniden çalışır)
    graph = generate_er_graph(NODE_COUNT, ERDOS_RENYI_P, seed=seed)

    if 'result' not in st.session_state:
        st.session_state.result = None

    if run_button:
        with st.spinner("En uygun yol hesaplanıyor..."):
            st.session_state.result = compute_path(
                graph, source, target, w_delay, w_rel, w_res, requested_bw
            )

    result = st.session_state.result
    
    if not result:
        st.info("Ayarları yapılandırıp 'Rotalamayı Hesapla' butonuna basarak simülasyonu başlatın.")
        # Başlangıçta boş bir ağ göster
        html_path = draw_pyvis_network(graph, [], source, target, theme)
        with open(html_path, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=710)
        os.remove(html_path)
    else:
        # --- SONUÇLARIN GÖSTERİMİ ---
        st.header("📊 Hesaplama Sonuçları")
        
        metrics = result.get("metrics", {})
        path = result.get("path", [])

        if not path:
            st.error(result.get("notes", "Yol bulunamadı!"))
        else:
            # Metrik Kartları
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Toplam Gecikme", f"{metrics.get('total_delay_ms', 0):.2f} ms")
            m_col2.metric("Yol Güvenilirliği", f"{metrics.get('path_reliability', 0)*100:.2f}%")
            m_col3.metric("Kaynak Maliyeti", f"{metrics.get('resource_cost', 0):.2f}")
            
            min_bw = metrics.get('min_bandwidth_on_path', 0)
            meets_bw = metrics.get('meets_requested_bw', False)
            m_col4.metric(
                "Min. Bant Genişliği", f"{min_bw} Mbps",
                help=f"Talep: {requested_bw} Mbps. {'Karşılanıyor.' if meets_bw else 'Karşılanamıyor!'}"
            )

            # Ağ Görselleştirmesi
            st.subheader("🗺️ Ağ Grafiği ve Bulunan Yol")
            with st.spinner("Ağ grafiği oluşturuluyor..."):
                html_path = draw_pyvis_network(graph, path, source, target, theme)
                with open(html_path, 'r', encoding='utf-8') as f:
                    components.html(f.read(), height=710)
                os.remove(html_path)

            # Detay Tabloları ve İndirme Butonları
            st.subheader("📋 Yol Detayları")
            
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.markdown("##### Düğüm Detayları")
                nodes_df = pd.DataFrame(result.get("per_node", []))
                st.dataframe(nodes_df)
                st.download_button(
                    "Düğüm Listesini (CSV) İndir",
                    nodes_df.to_csv(index=False).encode('utf-8'),
                    "yol_dugumleri.csv",
                    "text/csv"
                )

            with d_col2:
                st.markdown("##### Kenar Detayları")
                edges_df = pd.DataFrame(result.get("per_edge", []))
                st.dataframe(edges_df)
                st.download_button(
                    "Kenar Listesini (CSV) İndir",
                    edges_df.to_csv(index=False).encode('utf-8'),
                    "yol_kenarlari.csv",
                    "text/csv"
                )

# --- UYGULAMAYI BAŞLAT ---
if __name__ == "__main__":
    main()
