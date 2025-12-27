import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import time
import os
import numpy as np
from graph_loader import load_graph
from q_learning import QLearningAgent
from metrics import calculate_weighted_cost, calculate_total_delay, calculate_reliability_cost, calculate_resource_cost

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="QoS AI Dashboard", page_icon="⚡", layout="wide")

# --- TASARIM (Dark Tech) ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    [data-testid="stSidebar"] { background-color: #262730; border-right: 1px solid #333; }
    h1, h2, h3, p, label, .stSelectbox, .stSlider, .stNumberInput { color: #FAFAFA !important; font-family: 'Segoe UI', sans-serif; }
    div[data-testid="metric-container"] { background-color: #1E1E1E; border: 1px solid #444; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    div[data-testid="metric-container"]:hover { border-color: #00D2FF; }
    .stButton>button { background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%); color: black; font-weight: bold; border: none; border-radius: 8px; height: 50px; transition: 0.3s; }
    .stButton>button:hover { opacity: 0.9; transform: scale(1.02); color: black; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ QoS Odaklı Akıllı Rotalama (AI)")
st.markdown("Q-Learning algoritması ile **Gecikme, Güvenilirlik ve Bant Genişliği** optimizasyonu.")

# --- VERİ YÜKLEME ---
base_path = os.path.dirname(os.path.abspath(__file__))
node_file = os.path.join(base_path, 'BSM307_317_Guz2025_TermProject_NodeData.csv')
edge_file = os.path.join(base_path, 'BSM307_317_Guz2025_TermProject_EdgeData.csv')

@st.cache_resource
def get_graph():
    return load_graph(node_file, edge_file)

try:
    G = get_graph()
except Exception as e:
    st.error(f"Veri yüklenemedi: {e}")
    st.stop()

# --- SOL MENÜ ---
st.sidebar.header("🎛️ Kontrol Paneli")

st.sidebar.subheader("1. Rota ve Talep")
all_nodes = list(G.nodes())
source = st.sidebar.selectbox("Başlangıç (S)", all_nodes, index=0)
target = st.sidebar.selectbox("Hedef (D)", all_nodes, index=len(all_nodes)-1)

# İSTENEN DEĞİŞİKLİK: 100-1000 arası kullanıcı seçimi (Sadece raporlama için)
bandwidth_demand = st.sidebar.slider("Talep Edilen Hız (Mbps)", min_value=100, max_value=1000, value=500, step=100)

st.sidebar.divider()

st.sidebar.subheader("2. Optimizasyon Ağırlıkları")
w_delay = st.sidebar.slider("Gecikme Önemi", 0.0, 1.0, 0.33)
w_rel = st.sidebar.slider("Güvenilirlik Önemi", 0.0, 1.0, 0.33)
w_res = st.sidebar.slider("Kaynak Önemi", 0.0, 1.0, 0.34)

total = w_delay + w_rel + w_res
if 0.99 <= total <= 1.01:
    st.sidebar.success(f"⚖️ Toplam: {total:.2f}")
else:
    st.sidebar.warning(f"⚠️ Toplam: {total:.2f} (1.0 olmalı)")

st.sidebar.divider()

st.sidebar.subheader("3. Yapay Zeka")
episodes = st.sidebar.slider("Eğitim Turu", 100, 2000, 500)
alpha = st.sidebar.slider("Öğrenme Hızı", 0.01, 1.0, 0.1)

# --- GRAFİK FONKSİYONU ---
def draw_neon_graph(G, path=None):
    pos = nx.spring_layout(G, seed=42)
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        info = f"<b>Node {node}</b><br>Delay: {G.nodes[node].get('processing_delay')}ms"
        node_text.append(info)
        if path and node in path:
            if node == path[0]: node_color.append('#00FF00'); node_size.append(20)
            elif node == path[-1]: node_color.append('#FF0055'); node_size.append(20)
            else: node_color.append('#00D2FF'); node_size.append(12)
        else:
            node_color.append('rgba(255, 255, 255, 0.15)'); node_size.append(6)

    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text, marker=dict(showscale=False, color=node_color, size=node_size, line_width=0))
    traces = [node_trace]

    if path:
        p_x, p_y = [], []
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            p_x.extend([x0, x1, None])
            p_y.extend([y0, y1, None])
        glow_trace = go.Scatter(x=p_x, y=p_y, line=dict(width=8, color='rgba(0, 210, 255, 0.3)'), mode='lines', hoverinfo='none')
        line_trace = go.Scatter(x=p_x, y=p_y, line=dict(width=3, color='#00D2FF'), mode='lines', name='En İyi Yol', hoverinfo='none')
        traces.extend([glow_trace, line_trace])

    layout = go.Layout(showlegend=False, hovermode='closest', margin=dict(b=0,l=0,r=0,t=0), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return go.Figure(data=traces, layout=layout)

# --- ANA EKRAN ---
col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("### 🚀 İşlem Başlat")
    if st.button("EN İYİ ROTAYI BUL", use_container_width=True):
        if source == target:
            st.warning("⚠️ Başlangıç ve Hedef aynı!")
        else:
            with st.spinner(f'Yapay Zeka tüm ağı tarıyor...'):
                # KISITLAMA KALDIRILDI: Sadece grafiği ve ağırlıkları gönderiyoruz
                agent = QLearningAgent(G, source, target, episodes=episodes, alpha=alpha, 
                                     w_delay=w_delay, w_rel=w_rel, w_res=w_res)
                agent.train()
                path = agent.get_best_path()

            if path:
                st.balloons()
                st.success(f"✅ Rota Bulundu! ({len(path)-1} adım)")
                
                cost = calculate_weighted_cost(G, path, w_delay, w_rel, w_res)
                delay_val = calculate_total_delay(G, path)
                
                c1, c2 = st.columns(2)
                c1.metric("Toplam Maliyet", f"{cost:.2f}")
                c2.metric("Gecikme (ms)", f"{delay_val:.2f}")
                
                st.code(str(path), language="python")

                # BANT GENİŞLİĞİ KONTROLÜ (Sadece Raporlama)
                min_bw = float('inf')
                for i in range(len(path)-1):
                    bw = G[path[i]][path[i+1]].get('bandwidth', 0)
                    if bw < min_bw: min_bw = bw
                
                # Kullanıcıya rapor veriyoruz (Engellemek yok)
                if min_bw < bandwidth_demand:
                    st.warning(f"⚠️ UYARI: Bulunan yolun kapasitesi ({min_bw:.0f} Mbps), talep edilen ({bandwidth_demand} Mbps) değerinden düşük!")
                else:
                    st.success(f"✅ Hız Şartı Sağlandı! (Kapasite: {min_bw:.0f} Mbps)")
                
                st.session_state['path'] = path
            else:
                st.error("Yol bulunamadı. Eğitim turunu artırmayı deneyin.")
                st.session_state['path'] = None

with col2:
    path = st.session_state.get('path', None)
    fig = draw_neon_graph(G, path)
    st.plotly_chart(fig, use_container_width=True)
