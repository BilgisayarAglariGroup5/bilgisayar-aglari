# Melek Çakır
# Algoritma sahipleri bu kodu çekerek ağ topografisinden aldığı verilerile beraber hesaplama yapabilecek.
# kendi kodunuza path ile ekenti oluşturmalısınız.



import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import networkx as nx



# Ağırlık sınıfı------------------------------------------

@dataclass(frozen=True)
class Weights:
    """
    Çok amaçlı optimizasyonda kullanılan ağırlıkları tutar.
    Üç metrik için ayrı ağırlık belirlenir.
    """
    w_delay: float          # Gecikme metriğinin ağırlığı
    w_reliability: float   # Güvenilirlik maliyetinin ağırlığı
    w_resource: float      # Kaynak maliyetinin ağırlığı

    def normalized(self) -> "Weights":
        """
        Ağırlıkların toplamını 1 olacak şekilde normalize eder.
        Böylece weighted sum doğru şekilde hesaplanır.
        """
        total = self.w_delay + self.w_reliability + self.w_resource
        if total <= 0:
            raise ValueError("Ağırlıkların toplamı sıfır olamaz.")

        return Weights(
            self.w_delay / total,
            self.w_reliability / total,
            self.w_resource / total,
        )



# Path metrik sonuçları------------------------------------------


@dataclass(frozen=True)
class PathMetrics:
    """
    Bir path için hesaplanan tüm metrik sonuçlarını tutar.
    """

    total_delay_ms: float          # Toplam gecikme (ms)
    reliability_cost: float       # Güvenilirlik maliyeti (-log)
    resource_cost: float          # Kaynak maliyeti
    total_reliability: float      # Gerçek toplam güvenilirlik (çarpım)
    bottleneck_capacity_mbps: float  # Path üzerindeki en düşük bant genişliği
    feasible_for_demand: bool     # Talep bu path üzerinden karşılanabilir mi?

    def as_dict(self) -> Dict[str, float]:
        """
        Metrik sonuçlarını sözlük (dictionary) formatında döndürür.
        """
        return {
            "total_delay_ms": self.total_delay_ms,
            "reliability_cost": self.reliability_cost,
            "resource_cost": self.resource_cost,
            "total_reliability": self.total_reliability,
            "bottleneck_capacity_mbps": self.bottleneck_capacity_mbps,
            "feasible_for_demand": float(self.feasible_for_demand),
        }



# Metrik Hesaplama Motoru------------------------------------------

class MetricsEngine:
    """
    NetworkX tabanlı ağ topolojisi üzerinde çalışan,
    algoritmadan tamamen bağımsız metrik hesaplama sınıfı.
    """

    def __init__(
        self,
        G: nx.Graph,
        *,
        reference_bandwidth_mbps: float = 1000.0,
        eps: float = 1e-12,
    ):
        """
        G: NetworkX graph (node ve edge attribute'larını içerir)
        reference_bandwidth_mbps: Resource cost için referans bant genişliği (1 Gbps)
        eps: log(0) ve bölme hatalarını önlemek için kullanılan küçük sayı
        """
        self.G = G
        self.ref_bw = reference_bandwidth_mbps
        self.eps = eps

    def _edge(self, u: int, v: int) -> Dict:
        """
        Path üzerinde kullanılan (u, v) kenarının bilgilerini döndürür.
        Kenar yoksa path geçersiz kabul edilir.
        """
        if not self.G.has_edge(u, v):
            raise ValueError(f"Graph'ta edge yok: ({u}, {v})")
        return self.G[u][v]

    def compute(
        self,
        path: Sequence[int],
        *,
        demand_mbps: Optional[float] = None,
    ) -> PathMetrics:
        """
        Verilen path için:
        - Toplam gecikme
        - Güvenilirlik maliyeti
        - Kaynak maliyeti
        metriklerini hesaplar.
        """
        if path is None or len(path) < 2:
            raise ValueError("Path en az iki düğümden oluşmalıdır.")

       
        # 1) Toplam Gecikme Hesabı------------------------------------------

        # Tüm bağlantı gecikmelerinin toplamı------------------------------------------
        link_delay_sum = 0.0
        for u, v in zip(path[:-1], path[1:]):
            link_delay_sum += float(self._edge(u, v)["link_delay_ms"])

        # Ara düğümlerin (kaynak ve hedef hariç) işlem gecikmeleri------------------------------------------
        processing_sum = 0.0
        for n in path[1:-1]:
            processing_sum += float(self.G.nodes[n]["processing_delay_ms"])

        total_delay_ms = link_delay_sum + processing_sum

       
        # 2) Güvenilirlik Hesabı------------------------------------------
       
        reliability_cost = 0.0
        total_reliability = 1.0

        # Düğüm güvenilirlikleri------------------------------------------
        for n in path:
            r = max(float(self.G.nodes[n]["node_reliability"]), self.eps)
            reliability_cost += -math.log(r)
            total_reliability *= r

        # Bağlantı güvenilirlikleri------------------------------------------
        for u, v in zip(path[:-1], path[1:]):
            r = max(float(self._edge(u, v)["link_reliability"]), self.eps)
            reliability_cost += -math.log(r)
            total_reliability *= r

        # 3) Kaynak Maliyeti Hesabı------------------------------------------
    
        resource_cost = 0.0
        bottleneck = float("inf")

        for u, v in zip(path[:-1], path[1:]):
            cap = max(float(self._edge(u, v)["capacity_mbps"]), self.eps)
            bottleneck = min(bottleneck, cap)
            resource_cost += self.ref_bw / cap

       
        # 4) Talep (Demand) Kontrolü ------------------------------------------
        
        feasible = True
        if demand_mbps is not None:
            feasible = demand_mbps <= bottleneck

        # Sonuçları PathMetrics nesnesi olarak döndür
        return PathMetrics(
            total_delay_ms=total_delay_ms,
            reliability_cost=reliability_cost,
            resource_cost=resource_cost,
            total_reliability=total_reliability,
            bottleneck_capacity_mbps=bottleneck,
            feasible_for_demand=feasible,
        )

    def weighted_sum(
        self,
        metrics: PathMetrics,
        weights: Weights,
        *,
        infeasible_penalty: float = 1e9,
    ) -> float:
        """
        Üç metriği weighted sum yöntemiyle tek bir skora dönüştürür.
        """
        w = weights.normalized()
        score = (
            w.w_delay * metrics.total_delay_ms
            + w.w_reliability * metrics.reliability_cost
            + w.w_resource * metrics.resource_cost
        )

        # Talep karşılanmıyorsa 
        if not metrics.feasible_for_demand:
            score += infeasible_penalty

        return score
