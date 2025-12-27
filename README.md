# QoS Tabanlı Rota Bulma Projesi
Bilgisayar Ağları Dersi – Dönem Projesi (2025)

Bu proje, bir ağ topolojisi üzerinde QoS (Quality of Service) kriterlerine göre en uygun rotayı bulan bir sistem geliştirmeyi amaçlar.

Kullanıcı; kaynak–hedef düğümlerini ve QoS ağırlıklarını web arayüzünden seçer. Algoritmalar (GA, ACO vb.) bu verilere göre en iyi rotayı hesaplar.

---

## 👥 Proje Ekibi
- Arda Şengün
- Yiğit Alakuş
- Fatma Zeynep Düz
- Melek Çakır
- Metin Öztaş
- Afif Agung
- Houmedali
- Ahmat Musa
- İrem Gül Doğan

---

## 🎯 Projenin Amacı
Ağ üzerinde QoS kriterlerine göre “en iyi” rotayı bulmak ve bunu kullanıcıya açık, anlaşılır bir şekilde sunmak.

Kriterler:
- Gecikme (Delay)
- Güvenilirlik (Reliability)
- Kaynak Kullanımı (Resource Usage)

Her bir kriter kullanıcı tarafından farklı ağırlıklarla ayarlanabilir.

---

## 📁 Proje Klasör Yapısı

- **docs/**  
  Proje PDF’leri, raporlar ve dokümanlar.

- **web-ui/**  
  Python tabanlı simülasyon ve görselleştirme arayüzü (Streamlit + PyVis).

- **algorithms/**  
  Rota bulma algoritmaları (GA, ACO, vb.) ve yardımcı fonksiyonlar.

- **data/**  
  Seed dosyaları, ağ topolojileri, test verileri.

- **experiments/**  
  Deney sonuçları, log dosyaları, performans kayıtları.

- **scripts/**  
  Otomasyon amaçlı scriptler.

- **.github/workflows/**  
  GitHub Actions ayar dosyaları.

---

## 🛠 Kullanılan Teknolojiler (Planlanan)
- Python → Algoritmalar
- Python(Streamlit + PyVis) → Web Arayüzü
- JSON / GraphML → Ağ verileri
- GitHub → Versiyon kontrolü ve ekip çalışması

---

## 🚀 Çalışma Prensibi
1. Kullanıcı web arayüzünden:
   - Kaynak düğümünü seçer  
   - Hedef düğümünü seçer  
   - QoS ağırlıklarını ayarlar  

2. Arayüz, backend’e bir istek gönderir.

3. Algoritmalar (GA/ACO):
   - Ağı işler  
   - Ağırlıklandırılmış QoS maliyetini hesaplar  
   - En iyi rotayı belirler  

4. Bulunan rota tekrar web arayüzüne gönderilir.

5. Web arayüzü rotayı grafik olarak gösterir.

---

## 📌 Önemli Teslim Tarihleri
- **Kaynak Kod Teslimi:** 31 Aralık 2025 – 23:59  
- **Rapor Teslimi:** 7 Ocak 2026 – 23:59  

---

## 📄 Lisans
Bu proje MIT Lisansı ile lisanslanmıştır.
