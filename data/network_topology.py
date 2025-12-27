
import pandas as pd
import networkx as nx

# =====================================================
# TOPOLOGY YÜKLEME (NODE + EDGE)
# =====================================================
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

    print(f"Graph yüklendi | Node={G.number_of_nodes()} Edge={G.number_of_edges()}")

    # CONNECTIVITY KONTROLÜ
    if not nx.is_connected(G):
        raise ValueError("Graph connected değil!")

    print("Graph CONNECTED (bağlı)")
    return G, node_df, edge_df


# =====================================================
# KAPASİTE KISITLI EN KISA YOL (HATALI DURUM KONTROLLÜ)
# =====================================================
def route_with_capacity(G, src, dst, demand):

    # NODE VAR MI?
    if src not in G.nodes:
        raise nx.NodeNotFound(f"Kaynak node yok: {src}")

    if dst not in G.nodes:
        raise nx.NodeNotFound(f"Hedef node yok: {dst}")

    # DEMAND GEÇERLİ Mİ?
    if demand <= 0:
        raise ValueError("Demand pozitif olmalıdır.")

    Gd = G.copy()

    for u, v, data in list(Gd.edges(data=True)):
        if data["capacity_mbps"] < demand:
            Gd.remove_edge(u, v)

    if not nx.has_path(Gd, src, dst):
        raise ValueError(
            "Kapasite kısıtı nedeniyle src ile dst arasında yol yok."
        )

    return nx.shortest_path(Gd, src, dst, weight="delay_ms")


# =====================================================
# MAIN (INPUT SADECE BURADA)
# =====================================================
if __name__ == "__main__":

    try:
        G, node_df, edge_df = load_topology()

        print("\n--- DEMAND INPUT ---")
        src = int(input("Kaynak node (S): "))
        dst = int(input("Hedef node (D): "))
        demand = float(input("Demand (Mbps): "))

        path = route_with_capacity(G, src, dst, demand)

        print("\n UYGUN YOL BULUNDU")
        print("Yol:", " -> ".join(map(str, path)))

    except nx.NodeNotFound as e:
        print(f"\n NODE HATASI: {e}")

    except ValueError as e:
        print(f"\n TALEP / KAPASİTE HATASI: {e}")

    except Exception as e:
        print(f"\n BEKLENMEYEN HATA: {e}")
