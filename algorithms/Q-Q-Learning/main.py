
import os
import networkx as nx
from graph_loader import load_graph
from q_learning import QLearningAgent
from metrics import calculate_weighted_cost

def main():
    # Dosya yolları
    base_path = os.path.dirname(os.path.abspath(__file__))
    node_file = os.path.join(base_path, 'BSM307_317_Guz2025_TermProject_NodeData.csv')
    edge_file = os.path.join(base_path, 'BSM307_317_Guz2025_TermProject_EdgeData.csv')

    # 1. Grafiği Yükle
    try:
        G = load_graph(node_file, edge_file)
    except Exception as e:
        print(f"HATA: {e}")
        return

    # 2. Kaynak ve Hedef Seçimi (Örnek olarak)
    # Grafikteki mevcut düğümlerden ilkini ve sonuncusunu seçelim
    all_nodes = list(G.nodes())
    source_node = all_nodes[0]   # İlk düğüm
    target_node = all_nodes[-1]  # Son düğüm (uzak olsun)
    
    print(f"\n--- Q-Learning Başlatılıyor ---")
    print(f"Kaynak: {source_node} -> Hedef: {target_node}")

    # 3. Q-Learning Ajanını Oluştur ve Eğit
    # episodes=1000 ile ajan 1000 kere yolu bulmaya çalışıp öğrenecek
    agent = QLearningAgent(G, source_node, target_node, episodes=1000)
    agent.train()

    # 4. En İyi Yolu Bul
    best_path = agent.get_best_path()

    if best_path:
        print(f"\nBULUNAN EN İYİ YOL: {best_path}")
        cost = calculate_weighted_cost(G, best_path)
        print(f"Bu yolun toplam maliyet skoru: {cost:.4f}")
    else:
        print("\nMaalesef bir yol bulunamadı.")

if __name__ == "__main__":
    main()
