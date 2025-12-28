import pandas as pd
import networkx as nx

# ===============================
# TOPOLOGY YÜKLEME (NODE + EDGE)
# ===============================
def load_topology(
    node_csv="NodeData.csv",
    edge_csv="EdgeData.csv"
):
    node_df = pd.read_csv(node_csv, sep=";", decimal=",")
    edge_df = pd.read_csv(edge_csv, sep=";", decimal=",")

    G = nx.Graph()

    # NODE
    for _, r in node_df.iterrows():
        G.add_node(
            int(r["node_id"]),
            s_ms=float(r["s_ms"]),
            r_node=float(r["r_node"])
        )

    # EDGE
    for _, r in edge_df.iterrows():
        G.add_edge(
            int(r["src"]),
            int(r["dst"]),
            capacity_mbps=float(r["capacity_mbps"]),
            delay_ms=float(r["delay_ms"]),
            r_link=float(r["r_link"])
        )

    print(
    f"Graph yüklendi\n"
    f"Node = {G.number_of_nodes()}\n"
    f"Edge = {G.number_of_edges()}"
)

    # CONNECTIVITY KONTROLÜ
    if not nx.is_connected(G):
        raise ValueError("Graph connected değil!")

    print("Graph CONNECTED (bağlı)")
    return G, node_df, edge_df


# =================================
# SOFT-CONSTRAINT WEIGHT FUNCTION  
# =================================

def capacity_aware_weight(u, v, data, demand_mbps):
    """
    Kapasite yetersizse çok büyük ceza verilir
    """
    if data["capacity_mbps"] < demand_mbps:
        return 1e12  # soft penalty
    return data["delay_ms"]


# =================================
# PATH FINDING 
# =================================

def find_path(G, src, dst, demand_mbps):
    if src not in G or dst not in G:
        raise nx.NodeNotFound("Src veya Dst yok")

    return nx.shortest_path(
        G,
        src,
        dst,
        weight=lambda u, v, d: capacity_aware_weight(u, v, d, demand_mbps)
    )

# =================================
# MAIN (INPUT SADECE BURADA)
# =================================
if __name__ == "__main__":

    try:
        G, node_df, edge_df = load_topology()

        print("\n--- DEMAND INPUT ---")
        src = int(input("Kaynak node (S): "))
        dst = int(input("Hedef node (D): "))
        demand = float(input("Demand (Mbps): "))

       
        path = find_path(G, src, dst, demand)

        print("\nUYGUN YOL BULUNDU")
        print("Yol:", " -> ".join(map(str, path)))

    except nx.NodeNotFound as e:
        print(f"\nNODE HATASI: {e}")

    except ValueError as e:
        print(f"\nTALEP / KAPASİTE HATASI: {e}")

    except Exception as e:
        print(f"\nBEKLENMEYEN HATA: {e}")
