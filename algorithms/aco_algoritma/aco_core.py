"""
Ant Colony Optimization (ACO) – Çekirdek Algoritma Uygulaması
------------------------------------------------------------

Bu modül, ağ (graph) tabanlı yönlendirme problemleri için Ant Colony Optimization (ACO)
algoritmasının temel bileşenlerini içermektedir.

Amaç:
- Belirli bir başlangıç (start) ve hedef (end) düğümü arasında,
- Bant genişliği (bandwidth) kısıtlarını sağlayan,
- Gecikme (delay) ve feromon yoğunluğunu dikkate alarak
olasılıksal en iyi yolu keşfetmektir.

Algoritma üç temel aşamadan oluşur:
1. Karınca Yürüyüşü (ant_walk):
   - Karıncalar, feromon (τ) ve sezgisel bilgi (η) değerlerine göre
     olasılıksal olarak bir sonraki düğümü seçer.
2. Feromon Güncelleme (update_pheromone):
   - Daha iyi çözümler, ilgili kenarlar üzerinde daha fazla feromon bırakır.
3. Feromon Buharlaşması (evaporate_pheromone):
   - Zamanla feromon değerleri azaltılarak erken yakınsama (premature convergence)
     ve taşma (overflow) problemleri önlenir.

Bu çekirdek yapı; QoS yönlendirme, ağ optimizasyonu ve benzeri problemlerde
genişletilebilir bir temel sunar.
"""

import random


def ant_walk(G, start, end, demand_bw, heuristic_fn,
             alpha=1.0, beta=2.0):

    """
    Tek bir karıncanın, başlangıç düğümünden hedef düğüme kadar
    ACO kurallarına göre yol bulmasını sağlar.

    Parametreler:
    - G           : NetworkX grafiği (kenarlar feromon, gecikme ve bant genişliği içerir)
    - start       : Başlangıç düğümü
    - end         : Hedef düğüm
    - demand_bw   : Gerekli minimum bant genişliği (Mbps)
    - alpha       : Feromon etkisinin ağırlığı (τ^alpha)
    - beta        : Sezgisel bilginin ağırlığı (η^beta)

    Dönüş:
    - path (list) : Bulunan yol (düğüm listesi)
    - None        : Uygun yol bulunamazsa
    """

    # Karıncanın mevcut konumu başlangıç düğümüdür
    current = start

    # İzlenen yol
    path = [current]

    # Döngüleri önlemek için ziyaret edilen düğümler kümesi
    visited = {current}

    # Karınca hedefe ulaşana kadar devam et
    while current != end:
        if len(path) > len(G.nodes):
            return None
        # -------------------------------
        # 1. Bant Genişliği Kısıtı
        # -------------------------------
        # Sadece:
        # - Daha önce ziyaret edilmemiş
        # - İstenen bant genişliğini karşılayan
        # komşular seçilir
        neighbors = [
            n for n in G.neighbors(current)
            if n not in visited and
            G[current][n].get("capacity_mbps", 0) >= demand_bw


        ]

        # Eğer uygun komşu yoksa bu karınca başarısız olur
        if not neighbors:
            return None

        weights = []

        # -------------------------------
        # 2. Olasılık Ağırlıklarının Hesaplanması
        # -------------------------------
        for n in neighbors:
            # Feromon değeri (τ)
            raw_pheromone = G[current][n].get("pheromone", 0.1)

            # Feromon taşmasını (overflow) önlemek için üst sınır
            tau = min(raw_pheromone, 1000) ** alpha

            eta = heuristic_fn(G, current, n) ** beta


            # Ağırlık = Feromon etkisi * Sezgisel bilgi
            w = tau * eta

            # Sayısal kararlılık için minimum değer sınırı
            weights.append(max(w, 1e-6))

        # -------------------------------
        # 3. Olasılıksal Sonraki Düğüm Seçimi
        # -------------------------------
        try:
            # Ağırlıklara göre olasılıksal seçim
            next_node = random.choices(neighbors, weights=weights, k=1)[0]
        except ValueError:
            # Herhangi bir sayısal hata durumunda rastgele seçim
            next_node = random.choice(neighbors)

        # Seçilen düğüm yola eklenir
        path.append(next_node)
        visited.add(next_node)
        current = next_node

    # Hedefe ulaşıldığında bulunan yol döndürülür
    return path


def update_pheromone(G, path, score, Q):
    """
    Bulunan yolun kalitesine göre feromon miktarını günceller.

    Parametreler:
    - G     : NetworkX grafiği
    - path  : Karıncanın bulduğu yol
    - score : Yolun maliyeti (ör. toplam gecikme)
    - Q     : Feromon ekleme katsayısı

    Not:
    - Daha düşük skor (daha iyi yol) daha fazla feromon bırakır.
    """

    # Geçersiz veya anlamsız yollar feromon güncellemez
    if not path or score == 0 or score == float("inf"):
        return

    # Feromon patlamasını önlemek için skor alt sınırı
    addition = Q / max(score, 0.1)

    # Yol üzerindeki her kenar için feromon ekle
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]

        # Mevcut feromona ekleme yapılır
        # Maksimum feromon değeri ile sınırlandırılır
        G[u][v]["pheromone"] = min(
            G[u][v].get("pheromone", 0.1) + addition,
            10000
        )


def evaporate_pheromone(G, rho):
    """
    Graf üzerindeki tüm kenarlarda feromon buharlaşmasını uygular.

    Parametreler:
    - G   : NetworkX grafiği
    - rho : Buharlaşma oranı (0 < rho < 1)

    Amaç:
    - Eski yolların etkisini azaltmak
    - Algoritmanın keşif (exploration) yeteneğini korumak
    """

    for u, v in G.edges():
        current_ph = G[u][v].get("pheromone", 0.1)

        # Buharlaşma sonrası feromon tamamen sıfırlanmaz
        G[u][v]["pheromone"] = max(
            current_ph * (1 - rho),
            0.01
        )
