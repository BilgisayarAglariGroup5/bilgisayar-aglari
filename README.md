# 🚀 QoS Tabanlı Rota Bulma ve Algoritma Karşılaştırma Sistemi

**Bilgisayar Ağları Dersi – Dönem Projesi (2025)**

Bu proje, bir ağ topolojisi üzerinde **QoS (Quality of Service)** kriterlerine göre en uygun rotayı bulan ve farklı algoritmaların performansını **aynı koşullar altında karşılaştıran** bir **masaüstü uygulaması** geliştirmeyi amaçlamaktadır.

Kullanıcı; kaynak–hedef düğümlerini ve QoS ağırlıklarını **masaüstü arayüz** üzerinden belirler. Algoritmalar (**ACO, Genetik Algoritma, Q-Q Learning, Simulated Annealing**) bu girdilere göre en uygun rotayı hesaplar ve sonuçlar **grafiksel ve sayısal olarak** kullanıcıya sunulur.

---

### 👥 Proje Ekibi

| Ad | Soyad |
| :--- | :--- |
| **Arda** | Şengün |
| **Yiğit** | Alakuş |
| **Fatma Zeynep** | Düz |
| **Melek** | Çakır |
| **Metin** | Öztaş |
| **Afif** | Agung |
| **Houmedali** | |
| **Ahmat** | Musa |
| **İrem Gül** | Doğan |

---

## 🎯 Projenin Amacı

Ağ üzerinde **QoS kriterlerini dikkate alarak “en iyi” rotayı bulmak** ve farklı algoritmaların bu problemi nasıl çözdüğünü **karşılaştırmalı olarak analiz etmektir**.

### Kullanılan QoS Kriterleri
- **Gecikme (Delay)**
- **Güvenilirlik (Reliability)** *(karşılaştırmalarda `-log` tabanlı maliyet olarak kullanılır)*
- **Kaynak Kullanımı (Resource Usage)**

Her kriterin ağırlığı kullanıcı tarafından masaüstü arayüz üzerinden dinamik olarak ayarlanabilir.

---

## ⚙️ Kullanılan Algoritmalar

- 🐜 **Ant Colony Optimization (ACO)**
- 🧬 **Genetik Algoritma (GA)**
- 🤖 **Q-Q Learning**
- 🔥 **Simulated Annealing (SA)**

Tüm algoritmalar:
- Aynı ağ grafiği
- Aynı kaynak–hedef (S, D)
- Aynı QoS ağırlıkları

altında çalıştırılarak **adil bir karşılaştırma ortamı** sağlanır.

---

## 🖥️ Masaüstü Uygulama Özellikleri

### 🔹 Tek Algoritma Modu
- Seçilen algoritma için:
  - En iyi rota hesaplanır
  - Ağ grafiği **CustomTkinter arayüzü içinde** renkli olarak gösterilir
  - Yol detayları (delay, bandwidth, reliability) listelenir

### 🔹 Algoritma Karşılaştırma Modu
- ACO, GA, Q-Q Learning ve SA algoritmaları:
  - Aynı koşullarda **N kez (N ≥ 5)** çalıştırılır
- Üretilen çıktılar:
  - **Özet tablo** (ortalama maliyet, standart sapma, en iyi / en kötü sonuçlar, ortalama süre)
  - **Tüm çalıştırmalar tablosu**
  - **Algoritma bazlı en iyi yollar**
- En iyi bulunan yollar grafik üzerinde karşılaştırmalı olarak gösterilir

---

## 📁 Proje Klasör Yapısı

```text
.
├── docs/                # Proje PDF’leri, raporlar ve dokümantasyon
├── algorithms/          # Algoritma kaynak kodları
│   ├── ACO/
│   ├── GA/
│   ├── QLearning/
│   ├── SimulatedAnnealing/
│   └── utils/           # Ortak yardımcı fonksiyonlar
├── metrics/             # QoS metrik hesaplamaları
│   ├── Delay/
│   ├── Reliability/     # Reliability Cost (-log)
│   ├── Resource/
│   └── TotalCost/
├── data/                # Veri dosyaları
│   ├── topology/        # Ağ topolojisi üretimi (seed tabanlı)
│   └── datasets/        # Test ve örnek veri setleri
├── experiments/
 ├── run_experiments.py      # Toplu (batch) deney çalıştırma scripti
 ├── scenarios.csv           # (S, D, B) deney senaryoları
 └── results/
     ├── runs_table.csv      # Tüm tekrarların ham sonuçları
     └── summary_table.csv   # Ortalama, std, en iyi / en kötü sonuçlar
├── scripts/             # Yardımcı otomasyon ve test scriptleri
└── .github/
    └── workflows/       # GitHub Actions ayar dosyaları
```
---

## 🛠 Kullanılan Teknolojiler

* **Python:** Algoritmaların geliştirilmesi ve uygulama mantığı
* **CustomTkinter (CTk):** Modern masaüstü kullanıcı arayüzü
* **Matplotlib (TkAgg backend):** Ağ grafiğinin GUI içinde çizilmesi ve görselleştirilmesi
* **NetworkX:** Ağ topolojisi oluşturma ve grafik modelleme
* **NumPy:** Sayısal hesaplamalar ve yardımcı matematiksel işlemler
* **Pandas:** Topoloji ve veri işleme
* **GitHub:** Versiyon kontrolü ve ekip çalışması

## 🔄 Çalışma Prensibi

**Kullanıcı masaüstü arayüzünden:**
* Kaynak düğümü (S)
* Hedef düğümü (D)
* QoS ağırlıklarını seçer

**Seçilen algoritma(lar) çalıştırılır.**

**Algoritmalar:**
* Ağı işler
* Ağırlıklandırılmış QoS maliyetini hesaplar
* En iyi rotayı belirler

**Sonuçlar:**
* Grafik üzerinde görselleştirilir
* Sayısal tablolar ile kullanıcıya sunulur

## 📄 Lisans

Bu proje **MIT Lisansı** ile lisanslanmıştır.
