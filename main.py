# ui_main.py
# ============================================================
# QoS Multi-Objective Routing Optimization (CustomTkinter + Matplotlib + NetworkX)
# ============================================================
# Bu dosya arayüz (UI) katmanını içerir:
# - Graf üretimi/topoloji (topology.py) üzerinden NetworkX grafiği oluşturur
# - Kullanıcıdan ağırlıkları ve algoritmayı alır
# - Seçilen algoritmayı adapter üzerinden çalıştırır
# - Sonuç yolu graf üzerinde görselleştirir
# - Yol detaylarını (hops/metric) panelde gösterir
#
# Not:
# - Algoritmaların asıl implementasyonları algorithms/* altında beklenir.
# - algorithms.adapter yoksa fallback fonksiyonlar devreye girer.
# ============================================================

import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure  # Pop-up pencerede Figure objesi kullanmak için
import networkx as nx
from tkinter import messagebox
import math
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

from topology import generate_graph, compute_layout, build_hops_for_path

# ------------------------------------------------------------
# Algorithms adapter:
# - list_algorithms(): UI menüsünde hangi algoritmalar görünecek?
# - get_algorithm_meta(name): Parametreleri (label/default/type) UI'de dinamik oluşturmak için
# - run_algorithm(...): Seçilen algoritmayı çalıştırır ve path/metrics döner
# - compare_algorithms(...): Karşılaştırma penceresinde (çoklu run) tüm algoritmaları çalıştırır
#
# Adapter bulunamazsa fallback ile menü görünür ama çalıştırma hata verir.
# ------------------------------------------------------------
try:
    from algorithms.adapter import list_algorithms, get_algorithm_meta, run as run_algorithm, compare as compare_algorithms
except Exception:
    def list_algorithms():
        # Adapter yoksa UI'nin tamamen çökmesini önlemek için sabit liste
        return ["ACO (Ant Colony)", "Genetik (GA)", "Q-Learning", "Simulated Annealing (SA)"]
    def get_algorithm_meta(name):
        # Adapter yoksa parametre meta bilgisi olmayacak
        return None
    def run_algorithm(name, G, src, dst, w_delay, w_rel, w_res, params):
        # Adapter yoksa algoritma çalıştırılamaz
        raise RuntimeError("Algorithms adapter not available")
    def compare_algorithms(*args, **kwargs):
        # Adapter yoksa karşılaştırma çalıştırılamaz
        raise RuntimeError("Algorithms adapter not available")


# ------------------------------------------------------------
# Global görünüm modu (CustomTkinter)
# - Uygulama açılışta Dark ile başlar
# - UI içindeki theme menüsü change_theme ile değiştirir
# ------------------------------------------------------------
ctk.set_appearance_mode("Dark")


class RoutingApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        # ------------------------------------------------------------
        # Ana pencere ayarları
        # ------------------------------------------------------------
        self.title("QoS Multi-Objective Routing Optimization")
        self.geometry("1400x850")  # sidebar + ana grafik alanı için genişlik
        self.minsize(1300, 760)

        # ------------------------------------------------------------
        # Tema renk sözlüğü
        # - Light/Dark geçişinde widget renkleri buradan beslenir
        # ------------------------------------------------------------
        self.themes = {
            "Light": {
                "bg": "#F5F6FA",
                "panel": "#FFFFFF",
                "text": "#1F2937",
                "muted": "#7A8599",
                "border": "#E6E8F0",
                "node_label": "#4B5563",
                "btn": "#8CB6F5",
                "btn_hover": "#6FA3F0",
            },
            "Dark": {
                "bg": "#0E1525",
                "panel": "#121B2E",
                "text": "#E7EAF0",
                "muted": "#A6B0C3",
                "border": "#1C2740",
                "node_label": "#CBD5E1",
                "btn": "#7FB0FF",
                "btn_hover": "#5B98FF",
            }
        }

        # ------------------------------------------------------------
        # Slider renkleri (UI'de görsel ayrım için)
        # ------------------------------------------------------------
        self.SOFT_BLUE = "#8CB6F5"    # hız / gecikme ağırlığı
        self.SOFT_YELLOW = "#F6D08B"  # güvenlik / güvenilirlik ağırlığı
        self.SOFT_PINK = "#F2A1A1"    # kaynak maliyeti ağırlığı

        # ------------------------------------------------------------
        # Graf çizim renkleri (varsayılan)
        # ------------------------------------------------------------
        self.edge_color = "#A7B0C0"
        self.node_color = "#A9C9F7"

        # ------------------------------------------------------------
        # Yol vurgulama renkleri:
        # - path_color: seçilen yol
        # - src_color: kaynak düğüm
        # - dst_color: hedef düğüm
        # - intermediate_color: yol üzerindeki ara düğümler
        # ------------------------------------------------------------
        self.path_color = "#FFC107"
        self.src_color = "#7BDCB5"
        self.dst_color = "#FF8A8A"
        self.intermediate_color = "#0099FF"

        # ------------------------------------------------------------
        # Tema state
        # ------------------------------------------------------------
        self.theme = "Dark"
        self.colors = self.themes[self.theme]

        # ------------------------------------------------------------
        # Graf verisi:
        # - seed/n/p: Erdos-Renyi üretim parametreleri
        # - G: NetworkX grafı
        # - pos: node'ların (x,y) yerleşimi
        # - current_path: ekranda en son çizilen yol (redraw için tutulur)
        # ------------------------------------------------------------
        self.seed = 42
        self.n = 250
        self.p = 0.40
        self.G = generate_graph(self.n, self.p, self.seed)
        self.pos = compute_layout(self.G, seed=self.seed)
        self.current_path = None

        # ------------------------------------------------------------
        # Küre görünümü (orthographic projection) için state
        # - view_mode: "düz" veya "küre"
        # - globe_lon/lat: küreyi sürükleyerek döndürme açıları
        # - globe_R: scroll ile zoom
        # ------------------------------------------------------------
        self.view_mode = "düz"
        self.globe_lon = 0.0
        self.globe_lat = 0.0
        self.globe_R = 1.0
        self._globe_dragging = False
        self._globe_last_xy = None

        # ------------------------------------------------------------
        # Varsayılan kaynak/hedef düğümleri
        # ------------------------------------------------------------
        self.s_node = 5
        self.d_node = 100

        # ------------------------------------------------------------
        # Node seçici pencere state (donmayı önlemek için batch + debounce)
        # ------------------------------------------------------------
        self.node_win = None
        self.node_scroll = None
        self.node_loading_lbl = None
        self.selecting_node = None
        self.filtered_nodes = list(range(self.n))
        self.node_after_id = None          # batch build after id
        self.search_debounce_id = None     # arama debounce after id

        # ------------------------------------------------------------
        # Grid layout:
        # - column 0: sidebar (sabit genişlik)
        # - column 1: main_frame (grafik + detaylar)
        # ------------------------------------------------------------
        self.configure(fg_color=self.colors["bg"])
        self.grid_columnconfigure(0, weight=0, minsize=460)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar frame
        self.sidebar = ctk.CTkFrame(self, width=460, corner_radius=16, fg_color=self.colors["panel"])
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=(20, 10), pady=20)

        # Main content frame
        self.main_frame = ctk.CTkFrame(self, corner_radius=16, fg_color=self.colors["panel"])
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 20), pady=20)
        self.main_frame.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=1)

        # ------------------------------------------------------------
        # UI bileşenlerini kur
        # ------------------------------------------------------------
        self.build_sidebar()
        self.build_plot()
        self.build_metrics()

        # ------------------------------------------------------------
        # Widget'lara tema uygula ve ilk çizimi yap
        # ------------------------------------------------------------
        self.apply_theme_to_widgets()
        self.draw_graph(path=None)

        # Enter tuşu -> Hesapla
        self.bind("<Return>", lambda e: self.on_calculate())

        # X ile kapatınca temiz kapanış (node selector after id iptali vs.)
        self.protocol("WM_DELETE_WINDOW", self.on_app_close)

    # ---------------- APP CLOSE ----------------
    def on_app_close(self):
        # Node selector açıkken after işleri kalmasın diye temizle
        self.cleanup_node_selector()
        self.destroy()

    # ---------------- THEME ----------------
    def change_theme(self, mode):
        # Tema değişince:
        # - colors sözlüğünü güncelle
        # - CustomTkinter appearance mode'u değiştir
        # - Tüm widget'lara yeniden stil uygula
        # - Grafiği mevcut path ile yeniden çiz
        self.theme = mode
        self.colors = self.themes[self.theme]
        ctk.set_appearance_mode(mode)

        self.configure(fg_color=self.colors["bg"])
        self.sidebar.configure(fg_color=self.colors["panel"])
        self.main_frame.configure(fg_color=self.colors["panel"])

        if hasattr(self, "fig"):
            self.fig.patch.set_facecolor(self.colors["panel"])

        self.apply_theme_to_widgets()
        self.draw_graph(path=self.current_path)

        # Node selector açıksa onun temasını da güncelle
        if self.node_win is not None and self.node_win.winfo_exists():
            self.node_win.configure(fg_color=self.colors["panel"])
            self.node_win.lift()
            self.node_win.focus_force()

    def apply_theme_to_widgets(self):
        # Bu fonksiyon var olan widget'lara tema renklerini uygular.
        # Not: Widget'lar bazı durumlarda henüz oluşmamış olabilir, bu yüzden hasattr kontrolü var.

        if hasattr(self, "title_lbl"):
            self.title_lbl.configure(text_color=self.colors["text"])
        if hasattr(self, "section_node_lbl"):
            self.section_node_lbl.configure(text_color=self.colors["text"])
        if hasattr(self, "section_weight_lbl"):
            self.section_weight_lbl.configure(text_color=self.colors["text"])
        if hasattr(self, "hint_lbl"):
            self.hint_lbl.configure(text_color=self.colors["muted"])

        # Demand slider rengi
        if hasattr(self, "demand_mbps_slider"):
            self.demand_mbps_slider.configure(
                progress_color=self.SOFT_BLUE,
                button_color=self.SOFT_BLUE
            )

        # Seçili S/D label'ları
        if hasattr(self, "lbl_s"):
            self.lbl_s.configure(text_color=self.colors["text"])
        if hasattr(self, "lbl_d"):
            self.lbl_d.configure(text_color=self.colors["text"])

        # Tema menüsü
        if hasattr(self, "theme_menu"):
            self.theme_menu.configure(
                fg_color=self.colors["btn"],
                button_color=self.colors["btn"],
                button_hover_color=self.colors["btn_hover"],
                text_color=self.colors["panel"],
            )

        # Ana butonlar (S seç, D seç, hesapla)
        for name in ["btn_select_s", "btn_select_d", "btn_calc"]:
            if hasattr(self, name):
                getattr(self, name).configure(
                    fg_color=self.colors["btn"],
                    hover_color=self.colors["btn_hover"],
                    text_color=self.colors["panel"],
                )

        # Karşılaştır butonu ayrı renkte
        if hasattr(self, "btn_compare"):
            self.btn_compare.configure(
                fg_color="#9B59B6",
                hover_color="#8E44AD",
                text_color="white",
            )

        # Grafiği yenile butonu: "secondary" tarz
        if hasattr(self, "btn_regen"):
            self.btn_regen.configure(
                fg_color=self.colors["panel"],
                text_color=self.colors["text"],
                border_color=self.colors["border"],
                hover_color=self.colors["bg"],
            )

        # Slider label'ları (değerler)
        for name in ["w_delay_lbl", "w_rel_lbl", "w_res_lbl", "demand_mbps_label"]:
            if hasattr(self, name):
                getattr(self, name).configure(text_color=self.colors["muted"])

        # Slider başlıkları
        for name in ["w_delay_title", "w_rel_title", "w_res_title"]:
            if hasattr(self, name):
                getattr(self, name).configure(text_color=self.colors["text"])

        # Normalize checkbox
        if hasattr(self, "normalize_cb"):
            self.normalize_cb.configure(text_color=self.colors["text"])

        # Node kartı
        if hasattr(self, "node_card"):
            self.node_card.configure(fg_color=self.colors["bg"], border_color=self.colors["border"])
        if hasattr(self, "node_card_hint"):
            self.node_card_hint.configure(text_color=self.colors["muted"])

        # Algo menüsü
        if hasattr(self, "algo_menu"):
            self.algo_menu.configure(
                fg_color=self.colors["btn"],
                button_color=self.colors["btn"],
                button_hover_color=self.colors["btn_hover"],
                text_color=self.colors["panel"],
            )

        # Algo param girişleri
        if hasattr(self, "algo_params_frame"):
            for child in self.algo_params_frame.winfo_children():
                try:
                    if isinstance(child, ctk.CTkLabel):
                        child.configure(text_color=self.colors["text"])
                    if isinstance(child, ctk.CTkEntry):
                        child.configure(
                            fg_color=self.colors["panel"],
                            text_color=self.colors["text"],
                            border_color=self.colors["border"]
                        )
                except:
                    pass

        # Detay paneli
        if hasattr(self, "details_container"):
            self.details_container.configure(fg_color=self.colors["panel"], border_color=self.colors["border"])
        if hasattr(self, "details_title"):
            self.details_title.configure(text_color=self.colors["text"])
        if hasattr(self, "copy_btn"):
            self.copy_btn.configure(
                fg_color=self.colors["btn"],
                hover_color=self.colors["btn_hover"],
                text_color=self.colors["panel"]
            )
        if hasattr(self, "details_toggle_btn"):
            self.details_toggle_btn.configure(
                fg_color=self.colors["btn"],
                hover_color=self.colors["btn_hover"],
                text_color=self.colors["panel"]
            )

        if hasattr(self, "sum_lbl"):
            self.sum_lbl.configure(text_color=self.colors["muted"])

    # ---------------- SIDEBAR ----------------
    def build_sidebar(self):
        # Sidebar scroll:
        # - Küçük ekranlarda butonların altta kalıp kaybolmaması için scroll kullanılıyor
        self.sidebar_scroll = ctk.CTkScrollableFrame(
            self.sidebar,
            corner_radius=12,
            scrollbar_button_color=self.colors["border"],
            scrollbar_button_hover_color=self.colors["btn"]
        )
        self.sidebar_scroll.pack(fill="both", expand=True, padx=8, pady=8)

        # Başlık
        self.title_lbl = ctk.CTkLabel(
            self.sidebar_scroll, text="Kontrol Paneli",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=self.colors["text"],
            anchor="w",
            justify="left"
        )
        self.title_lbl.pack(padx=16, pady=(18, 10), anchor="w", fill="x")

        # Tema seçimi
        self.theme_menu = ctk.CTkOptionMenu(
            self.sidebar_scroll, values=["Light", "Dark"],
            command=self.change_theme
        )
        self.theme_menu.set(self.theme)
        self.theme_menu.pack(padx=16, pady=(0, 14), fill="x")

        # Düğüm seçimi bölümü
        self.section_node_lbl = ctk.CTkLabel(
            self.sidebar_scroll, text="Düğüm Seçimi",
            text_color=self.colors["text"],
            font=ctk.CTkFont(size=14, weight="normal"),
            anchor="w",
            justify="left"
        )
        self.section_node_lbl.pack(padx=16, pady=(6, 6), anchor="w", fill="x")

        # Düğüm seçme kartı
        self.node_card = ctk.CTkFrame(
            self.sidebar_scroll, fg_color=self.colors["bg"],
            corner_radius=14, border_width=1, border_color=self.colors["border"]
        )
        self.node_card.pack(padx=16, pady=(0, 12), fill="x")

        self.node_card_hint = ctk.CTkLabel(
            self.node_card,
            text="Kaynak ve hedef düğümleri seçin (S ≠ D).",
            font=ctk.CTkFont(size=12, weight="normal"),
            text_color=self.colors["muted"],
            anchor="w",
            justify="left",
            wraplength=400
        )
        self.node_card_hint.pack(padx=12, pady=(10, 8), anchor="w", fill="x")

        # S seç
        self.btn_select_s = ctk.CTkButton(
            self.node_card, text="Kaynak Düğüm Seç",
            command=lambda: self.open_node_selector("S"),
            corner_radius=12
        )
        self.btn_select_s.pack(padx=12, pady=4, fill="x")

        # D seç
        self.btn_select_d = ctk.CTkButton(
            self.node_card, text="Hedef Düğüm Seç",
            command=lambda: self.open_node_selector("D"),
            corner_radius=12
        )
        self.btn_select_d.pack(padx=12, pady=(0, 6), fill="x")

        # Seçilen S/D göstergeleri
        self.lbl_s = ctk.CTkLabel(
            self.node_card, text=f"Seçilen S: {self.s_node}",
            font=ctk.CTkFont(size=12),
            text_color=self.colors["text"],
            anchor="w",
            justify="left"
        )
        self.lbl_s.pack(padx=12, pady=(0, 2), anchor="w", fill="x")

        self.lbl_d = ctk.CTkLabel(
            self.node_card, text=f"Seçilen D: {self.d_node}",
            font=ctk.CTkFont(size=12),
            text_color=self.colors["text"],
            anchor="w",
            justify="left"
        )
        self.lbl_d.pack(padx=12, pady=(0, 10), anchor="w", fill="x")

        # Ağırlıklar bölümü
        self.section_weight_lbl = ctk.CTkLabel(
            self.sidebar_scroll, text="Optimizasyon Ağırlıkları",
            text_color=self.colors["text"],
            font=ctk.CTkFont(size=14, weight="normal"),
            anchor="w",
            justify="left"
        )
        self.section_weight_lbl.pack(padx=16, pady=(10, 6), anchor="w", fill="x")

        # Üç amaç için slider + anlık değer label'ı
        self.w_delay_title, self.w_delay, self.w_delay_lbl = self.slider_with_value("Hız / Gecikme", 0.33, self.SOFT_BLUE)
        self.w_rel_title, self.w_rel, self.w_rel_lbl = self.slider_with_value("Güvenlik", 0.33, self.SOFT_YELLOW)
        self.w_res_title, self.w_res, self.w_res_lbl = self.slider_with_value("Kaynak Kontrolü", 0.34, self.SOFT_PINK)

        # Ağırlık toplamı göstergesi (normalize açık/kapalı farkını anlatmak için)
        self.sum_lbl = ctk.CTkLabel(
            self.sidebar,
            text="Ağırlık Toplamı: 1.00",
            text_color=self.colors["muted"],
            font=ctk.CTkFont(size=12),
            anchor="w",
            justify="left"
        )
        self.sum_lbl.pack(padx=16, pady=(0, 8), anchor="w", fill="x")

        # Slider değiştikçe:
        # - yanında görünen sayı güncellenir
        # - toplam etiket rengi/uyarısı güncellenir
        self.w_delay.configure(command=lambda v: (self.w_delay_lbl.configure(text=f"{float(v):.2f}"), self.on_weight_change()))
        self.w_rel.configure(command=lambda v: (self.w_rel_lbl.configure(text=f"{float(v):.2f}"), self.on_weight_change()))
        self.w_res.configure(command=lambda v: (self.w_res_lbl.configure(text=f"{float(v):.2f}"), self.on_weight_change()))

        # Algoritma menüsü (adapter'dan gelir)
        algo_list = list_algorithms()

        algo_label = ctk.CTkLabel(
            self.sidebar_scroll, text="Algoritma",
            text_color=self.colors["text"],
            font=ctk.CTkFont(size=12),
            anchor="w",
            justify="left"
        )
        algo_label.pack(padx=16, pady=(8, 4), anchor="w", fill="x")

        self.algo_menu = ctk.CTkOptionMenu(self.sidebar_scroll, values=algo_list, command=self.on_algo_change)
        self.algo_menu.set(algo_list[0])
        self.algo_menu.pack(padx=16, pady=(0, 8), fill="x")

        # Görünüm modu (düz / küre)
        view_label = ctk.CTkLabel(
            self.sidebar_scroll,
            text="Görünüm",
            text_color=self.colors["text"],
            font=ctk.CTkFont(size=12),
            anchor="w",
            justify="left"
        )
        view_label.pack(padx=16, pady=(6, 2), anchor="w", fill="x")
        self.view_menu = ctk.CTkOptionMenu(self.sidebar_scroll, values=["Düz", "Küre"], command=self.on_view_change)
        self.view_menu.set("Düz")
        self.view_menu.pack(padx=16, pady=(0, 8), fill="x")

        # Algoritma parametreleri (meta'dan dinamik üretilir)
        self.algo_params_frame = ctk.CTkFrame(self.sidebar_scroll, fg_color="transparent")
        self.algo_params_frame.pack(fill="x", padx=16, pady=(0, 8))
        self.algo_param_widgets = {}
        self.on_algo_change(self.algo_menu.get())

        # Demand (Mbps): bandwidth talebi (algoritmalara param olarak gider)
        demand_label = ctk.CTkLabel(
            self.sidebar_scroll, text="Demand (Mbps)",
            text_color=self.colors["text"],
            font=ctk.CTkFont(size=12),
            anchor="w",
            justify="left"
        )
        demand_label.pack(padx=16, pady=(8, 4), anchor="w", fill="x")

        self.demand_mbps_slider = ctk.CTkSlider(
            self.sidebar_scroll, from_=0, to=1000,
            progress_color=self.SOFT_BLUE,
            button_color=self.SOFT_BLUE
        )
        self.demand_mbps_slider.set(50.0)
        self.demand_mbps_slider.pack(fill="x", padx=16, pady=(0, 4))

        self.demand_mbps_label = ctk.CTkLabel(
            self.sidebar_scroll, text="50 Mbps",
            text_color=self.colors["muted"],
            font=ctk.CTkFont(size=11),
            anchor="e"
        )
        self.demand_mbps_label.pack(padx=16, pady=(0, 8), anchor="e", fill="x")

        # Slider hareket ettikçe label güncellenir
        self.demand_mbps_slider.configure(
            command=lambda v: self.demand_mbps_label.configure(text=f"{int(float(v))} Mbps")
        )

        # Normalize checkbox:
        # - Açık: (w1,w2,w3) otomatik 1'e normalize edilir
        # - Kapalı: kullanıcı toplamı 1 yapmak zorunda
        self.normalize_var = ctk.BooleanVar(value=True)
        self.normalize_cb = ctk.CTkCheckBox(
            self.sidebar_scroll,
            text="Ağırlıkları Normalize Et (Toplam=1)",
            variable=self.normalize_var,
            text_color=self.colors["text"]
        )
        self.normalize_cb.pack(padx=16, pady=(6, 10), anchor="w", fill="x")

        # Hesapla butonu
        self.btn_calc = ctk.CTkButton(
            self.sidebar_scroll, text="HESAPLA",
            command=self.on_calculate, corner_radius=12
        )
        self.btn_calc.pack(padx=16, pady=(6, 6), fill="x")

        # Karşılaştır butonu (tüm algoritmaları örnek run sayısı ile)
        self.btn_compare = ctk.CTkButton(
            self.sidebar_scroll, text="Algoritmaları Karşılaştır",
            command=self.on_compare_algorithms, corner_radius=12,
            fg_color="#9B59B6", hover_color="#8E44AD"
        )
        self.btn_compare.pack(padx=16, pady=(6, 6), fill="x")

        # Graf yenile: seed artırarak yeni random topoloji üretir
        self.btn_regen = ctk.CTkButton(
            self.sidebar_scroll, text="GRAFİĞİ YENİLE (Seed + 1)",
            fg_color=self.colors["panel"],
            text_color=self.colors["text"],
            border_width=1,
            border_color=self.colors["border"],
            hover_color=self.colors["bg"],
            corner_radius=12,
            command=self.on_regenerate
        )
        self.btn_regen.pack(padx=16, pady=(0, 10), fill="x")

        # Kullanıcı ipucu
        self.hint_lbl = ctk.CTkLabel(
            self.sidebar_scroll,
            text="İpucu: Düğüm listesi donma olmaması için\n parça parça yüklenir."
                 "Grafiğe ÇİFT TIKLAYARAK\n büyük ekranda açabilirsiniz.",
            text_color=self.colors["muted"],
            font=ctk.CTkFont(size=11),
            anchor="w",
            justify="left",
            wraplength=400
        )
        self.hint_lbl.pack(padx=16, pady=(8, 0), anchor="w", fill="x")

        # İlk ağırlık toplamını yazdır
        self.on_weight_change()

    def on_weight_change(self):
        # Slider'lardan ağırlıkları alıp toplamı gösterir.
        # Toplam 1 değilse label'ı amber renk yapıp görsel uyarı verir.
        w1 = float(self.w_delay.get())
        w2 = float(self.w_rel.get())
        w3 = float(self.w_res.get())
        s = w1 + w2 + w3

        self.sum_lbl.configure(text=f"Ağırlık Toplamı: {s:.2f}")
        if abs(s - 1.0) > 1e-3:
            self.sum_lbl.configure(text_color="#B45309")  # uyarı rengi
        else:
            self.sum_lbl.configure(text_color=self.colors["muted"])

    def on_algo_change(self, name):
        # Algoritma değişince:
        # - önce param frame temizlenir
        # - adapter meta'sından param listesi okunur
        # - UI'de label+entry olarak dinamik oluşturulur
        for w in self.algo_params_frame.winfo_children():
            w.destroy()
        self.algo_param_widgets = {}

        meta = get_algorithm_meta(name)

        # Meta yoksa param alanı göstermeyiz
        if not meta or 'params' not in meta:
            self.apply_theme_to_widgets()
            return

        for p in meta['params']:
            lbl = ctk.CTkLabel(
                self.algo_params_frame,
                text=p['label'],
                text_color=self.colors['text'],
                font=ctk.CTkFont(size=11)
            )
            lbl.pack(fill='x', padx=2, pady=(4, 2))

            entry = ctk.CTkEntry(self.algo_params_frame, placeholder_text=str(p.get('default', '')))
            entry.insert(0, str(p.get('default', '')))
            entry.pack(fill='x', padx=2, pady=(0, 4))

            # (entry_widget, tip) şeklinde saklanır; on_calculate() içinde parse edilir
            self.algo_param_widgets[p['name']] = (entry, p.get('type', 'str'))

        self.apply_theme_to_widgets()

    def collect_algo_params(self):
        # Param entry'lerinden değerleri toplayıp tip dönüşümü yapar.
        # Ayrıca demand_mbps değerini tüm algoritmalar için param olarak ekler.
        params = {}
        for name, (widget, typ) in self.algo_param_widgets.items():
            val = widget.get().strip()
            if val == '':
                params[name] = None
                continue
            try:
                if typ == 'int':
                    params[name] = int(val)
                elif typ == 'float':
                    params[name] = float(val)
                else:
                    params[name] = val
            except Exception:
                # Hatalı parse olursa string bırak
                params[name] = val

        demand_mbps = float(self.demand_mbps_slider.get())
        params['demand_mbps'] = demand_mbps
        params['demand_bw'] = demand_mbps  # bazı algoritmalar demand_bw adıyla bekleyebilir

        return params

    def slider_with_value(self, title, default, color):
        # Sidebar'da "başlık + slider + sağda değer label" şeklinde bileşen üretir.
        frame = ctk.CTkFrame(self.sidebar_scroll, fg_color="transparent")
        frame.pack(fill="x", padx=16, pady=(0, 8))

        top = ctk.CTkFrame(frame, fg_color="transparent")
        top.pack(fill="x")

        title_lbl = ctk.CTkLabel(
            top,
            text=title,
            text_color=self.colors["text"],
            font=ctk.CTkFont(size=11),
            anchor="w",
            justify="left",
            wraplength=320
        )
        title_lbl.pack(side="left", fill="x", expand=True, padx=(0, 8))

        val_lbl = ctk.CTkLabel(
            top,
            text=f"{default:.2f}",
            text_color=self.colors["muted"],
            font=ctk.CTkFont(size=12),
            width=50,
            anchor="e"
        )
        val_lbl.pack(side="right", padx=(8, 0))

        slider = ctk.CTkSlider(frame, from_=0, to=1, progress_color=color, button_color=color)
        slider.set(default)
        slider.pack(fill="x", pady=(6, 0))

        return title_lbl, slider, val_lbl

    # ---------------- NODE SELECTOR ----------------
    def open_node_selector(self, mode):
        # Düğüm seçici penceresini açar.
        # Donmayı önlemek için:
        # - arama input'u debounce edilir
        # - düğüm butonları batch (parça parça) oluşturulur
        self.selecting_node = mode

        if self.node_win is not None and self.node_win.winfo_exists():
            self.node_win.deiconify()
            self.node_win.lift()
            self.node_win.focus_force()
            return

        self.node_win = ctk.CTkToplevel(self)
        self.node_win.title("Düğüm Seç")
        self.node_win.geometry("420x520")
        self.node_win.resizable(False, False)
        self.node_win.configure(fg_color=self.colors["panel"])

        self.node_win.transient(self)
        self.node_win.lift()
        self.node_win.focus_force()
        self.node_win.attributes("-topmost", True)
        self.after(80, lambda: self.node_win.attributes("-topmost", False))

        title = "Kaynak (S) seç" if mode == "S" else "Hedef (D) seç"
        ctk.CTkLabel(
            self.node_win, text=title,
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=self.colors["text"]
        ).pack(pady=(12, 6))

        self.search_var = ctk.StringVar(value="")
        search_entry = ctk.CTkEntry(
            self.node_win,
            textvariable=self.search_var,
            placeholder_text="Numara ara (örn: 12)",
        )
        search_entry.pack(fill="x", padx=12, pady=(0, 10))
        search_entry.focus_set()

        self.node_scroll = ctk.CTkScrollableFrame(self.node_win, width=390, height=360)
        self.node_scroll.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.node_loading_lbl = ctk.CTkLabel(
            self.node_win,
            text="Liste hazırlanıyor...",
            text_color=self.colors["muted"],
            font=ctk.CTkFont(size=12)
        )
        self.node_loading_lbl.pack(pady=(0, 10))

        self.node_win.protocol("WM_DELETE_WINDOW", self.cleanup_node_selector)

        # Arama değişince rebuild'i debounce ile tetikle
        self.search_var.trace_add("write", lambda *_: self.schedule_rebuild_node_list())
        self.node_win.update_idletasks()
        self.after(10, self.schedule_rebuild_node_list)

    def schedule_rebuild_node_list(self, delay_ms: int = 120):
        # Debounce: kullanıcı yazarken sürekli rebuild yapmasın
        try:
            if self.search_debounce_id is not None:
                self.after_cancel(self.search_debounce_id)
        except:
            pass
        self.search_debounce_id = None

        if self.node_win is None or (not self.node_win.winfo_exists()):
            return

        self.search_debounce_id = self.after(delay_ms, self.rebuild_node_list)

    def cleanup_node_selector(self):
        # Node selector kapanırken:
        # - after işlerini iptal et
        # - pencereyi destroy et
        try:
            if self.search_debounce_id is not None:
                self.after_cancel(self.search_debounce_id)
        except:
            pass
        self.search_debounce_id = None

        try:
            if self.node_after_id is not None:
                self.after_cancel(self.node_after_id)
        except:
            pass
        self.node_after_id = None

        try:
            if self.node_win is not None and self.node_win.winfo_exists():
                self.node_win.destroy()
        except:
            pass

        self.node_win = None
        self.node_scroll = None
        self.node_loading_lbl = None

    def rebuild_node_list(self):
        # Arama filtresine göre düğüm listesini yeniden kurar (batch ile)
        if self.node_scroll is None:
            return

        self.search_debounce_id = None

        try:
            if self.node_after_id is not None:
                self.after_cancel(self.node_after_id)
        except:
            pass
        self.node_after_id = None

        # Önceki butonları temizle
        for w in self.node_scroll.winfo_children():
            w.destroy()

        # Filtrele:
        # - boşsa tüm düğümler
        # - numeric değilse boş sonuç
        q = (self.search_var.get() or "").strip()
        if q == "":
            self.filtered_nodes = list(range(self.n))
        else:
            if not q.isdigit():
                self.filtered_nodes = []
            else:
                self.filtered_nodes = [i for i in range(self.n) if q in str(i)]

        if self.node_loading_lbl is not None and self.node_loading_lbl.winfo_exists():
            self.node_loading_lbl.configure(text="Liste yükleniyor...")

        self._build_nodes_batched(start_index=0, batch_size=20)

    def _build_nodes_batched(self, start_index: int, batch_size: int = 50):
        # Düğüm butonlarını parça parça oluşturur (UI donmasın)
        if self.node_win is None or (not self.node_win.winfo_exists()) or self.node_scroll is None:
            return

        end_index = min(start_index + batch_size, len(self.filtered_nodes))

        for idx in range(start_index, end_index):
            i = self.filtered_nodes[idx]
            ctk.CTkButton(
                self.node_scroll,
                text=str(i),
                height=34,
                fg_color=self.colors["btn"],
                hover_color=self.colors["btn_hover"],
                text_color=self.colors["panel"],
                command=lambda x=i: self.select_node(x)
            ).pack(fill="x", pady=2)

        if end_index < len(self.filtered_nodes):
            self.node_after_id = self.after(1, lambda: self._build_nodes_batched(end_index, batch_size))
        else:
            self.node_after_id = None
            if self.node_loading_lbl is not None and self.node_loading_lbl.winfo_exists():
                self.node_loading_lbl.configure(
                    text=f"{len(self.filtered_nodes)} düğüm gösteriliyor."
                    if len(self.filtered_nodes) > 0 else
                    "Sonuç yok."
                )

    def select_node(self, node_id: int):
        # Seçilen düğümü (S veya D) set eder ve grafiği yeniden çizer
        if self.selecting_node == "S":
            self.s_node = node_id
            self.lbl_s.configure(text=f"Seçilen S: {self.s_node}")
        else:
            self.d_node = node_id
            self.lbl_d.configure(text=f"Seçilen D: {self.d_node}")

        self.draw_graph(path=self.current_path)
        self.cleanup_node_selector()

    # ---------------- PLOT ----------------
    def build_plot(self):
        # Matplotlib fig/ax oluştur ve Tk canvas içine yerleştir
        self.fig, self.ax = plt.subplots(figsize=(10, 7), dpi=100)
        self.fig.patch.set_facecolor(self.colors["panel"])

        # 'auto' aspect: pencere genişledikçe graf alanı yayılır (kare zorunluluğu yok)
        self.ax.set_aspect('auto')
        self.ax.set_axis_off()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        # Mouse event'leri:
        # - Küre modunda drag/scroll ile dönme/zoom
        # - Düz modda double click ile büyük pop-up açma
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_motion)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)

    def draw_graph(self, path=None):
        # Grafiği yeniden çizer:
        # - düz mod: networkx çizimi
        # - küre mod: orthographic projeksiyon ile çizim
        self.current_path = path

        # Küre modunda ayrı fonksiyon kullanılır
        if getattr(self, 'view_mode', 'düz') == 'küre':
            return self.draw_globe(path=path)

        self.ax.clear()

        # Kenar boşluklarını sıfırla, eksenleri kapat
        self.ax.set_axis_off()
        self.ax.set_aspect('auto')
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # Arkaplan edges
        nx.draw_networkx_edges(
            self.G, self.pos, ax=self.ax,
            width=0.3,
            edge_color=self.edge_color,
            alpha=0.2
        )

        # Tüm düğümler
        nx.draw_networkx_nodes(
            self.G, self.pos, ax=self.ax,
            node_size=30,
            node_color=self.node_color,
            alpha=0.7
        )

        # Tüm etiketler (n=250 olduğundan font küçük)
        nx.draw_networkx_labels(
            self.G, self.pos,
            labels={n: str(n) for n in self.G.nodes()},
            font_size=7,
            font_color=self.colors["node_label"],
            ax=self.ax
        )

        # Kaynak/hedef vurgusu
        nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax, nodelist=[self.s_node], node_size=150, node_color=self.src_color)
        nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax, nodelist=[self.d_node], node_size=150, node_color=self.dst_color)

        # Yol çizimi (varsa)
        if path and len(path) >= 2:
            edges = list(zip(path, path[1:]))

            # Glow (alt katman)
            nx.draw_networkx_edges(
                self.G, self.pos, ax=self.ax,
                edgelist=edges, width=6.0,
                edge_color=self.path_color,
                alpha=0.4
            )

            # Ana çizgi (üst katman)
            nx.draw_networkx_edges(
                self.G, self.pos, ax=self.ax,
                edgelist=edges, width=3.0,
                edge_color=self.path_color,
                alpha=1.0
            )

            # Ara düğümler (S/D hariç) mavi
            inter_nodes = [n for n in path if n != self.s_node and n != self.d_node]
            if inter_nodes:
                nx.draw_networkx_nodes(
                    self.G, self.pos, ax=self.ax,
                    nodelist=inter_nodes,
                    node_size=80,
                    node_color=self.intermediate_color,
                    alpha=1.0,
                    edgecolors="white"
                )

                # Ara düğüm etiketlerini beyaz/kalın yap (okunabilirlik)
                nx.draw_networkx_labels(
                    self.G, self.pos,
                    labels={n: str(n) for n in inter_nodes},
                    font_size=8,
                    font_weight="bold",
                    font_color="white",
                    ax=self.ax
                )

        self.canvas.draw()

    # ---------------- POP-UP GRAPH ----------------
    def open_expanded_graph(self, path=None):
        # Düz görünümde çift tıklama ile:
        # - Yeni bir pencere açılır
        # - Matplotlib toolbar ile zoom/pan aktif olur
        top = ctk.CTkToplevel(self)
        top.title("Detaylı Grafik İnceleme")
        top.geometry("1100x850")
        top.configure(fg_color=self.colors["bg"])
        top.focus_force()

        try:
            top.state("zoomed")
        except:
            pass

        info_frame = ctk.CTkFrame(top, fg_color="transparent")
        info_frame.pack(fill="x", padx=10, pady=5)

        # Not: self.colors içinde "dst_color" anahtarı yok; burada button rengi sabitlenebilir
        ctk.CTkButton(
            info_frame, text="Kapat", command=top.destroy, width=80, fg_color=self.dst_color
        ).pack(side="right", padx=10)


        fig = Figure(figsize=(12, 10), dpi=100)
        fig.patch.set_facecolor(self.colors["panel"])
        ax = fig.add_subplot(111)

        ax.set_axis_off()
        ax.set_aspect('auto')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # Ana graf çizimi
        nx.draw_networkx_edges(self.G, self.pos, ax=ax, width=0.3, edge_color=self.edge_color, alpha=0.2)
        nx.draw_networkx_nodes(self.G, self.pos, ax=ax, node_size=40, node_color=self.node_color, alpha=0.7)
        nx.draw_networkx_labels(self.G, self.pos, labels={n: str(n) for n in self.G.nodes()}, font_size=8, font_color=self.colors["text"], ax=ax)

        nx.draw_networkx_nodes(self.G, self.pos, ax=ax, nodelist=[self.s_node], node_size=200, node_color=self.src_color)
        nx.draw_networkx_nodes(self.G, self.pos, ax=ax, nodelist=[self.d_node], node_size=200, node_color=self.dst_color)

        # Yol vurgusu
        if path and len(path) >= 2:
            edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(self.G, self.pos, ax=ax, edgelist=edges, width=7.0, edge_color=self.path_color, alpha=0.4)
            nx.draw_networkx_edges(self.G, self.pos, ax=ax, edgelist=edges, width=3.5, edge_color=self.path_color, alpha=1.0)

            inter = [n for n in path if n != self.s_node and n != self.d_node]
            if inter:
                nx.draw_networkx_nodes(self.G, self.pos, ax=ax, nodelist=inter, node_size=100, node_color=self.intermediate_color, edgecolors="white", alpha=1.0)

        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(canvas, top)
        toolbar.update()

    # ---------------- VIEW MODE / GLOBE ----------------
    def on_view_change(self, name):
        # Düz/Küre seçimi
        self.view_mode = "küre" if name == "Küre" else "düz"
        self.draw_graph(path=self.current_path)

    def _lonlat_to_ortho(self, lon, lat, lon0, lat0, R):
        # Orthographic projection (derece giriş alır)
        lon_r = math.radians(lon)
        lat_r = math.radians(lat)
        lon0_r = math.radians(lon0)
        lat0_r = math.radians(lat0)
        cos_c = math.sin(lat0_r) * math.sin(lat_r) + math.cos(lat0_r) * math.cos(lat_r) * math.cos(lon_r - lon0_r)
        visible = cos_c > 0
        x = R * math.cos(lat_r) * math.sin(lon_r - lon0_r)
        y = R * (math.cos(lat0_r) * math.sin(lat_r) - math.sin(lat0_r) * math.cos(lat_r) * math.cos(lon_r - lon0_r))
        return x, y, visible

    def _compute_lonlat_from_pos(self):
        # Mevcut 2D layout'u (pos) lon/lat aralığına map eder
        xs = [p[0] for p in self.pos.values()]
        ys = [p[1] for p in self.pos.values()]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        dx = (xmax - xmin) if (xmax - xmin) != 0 else 1.0
        dy = (ymax - ymin) if (ymax - ymin) != 0 else 1.0
        lonlat = {}
        for n, (x, y) in self.pos.items():
            lon = ((x - xmin) / dx) * 360.0 - 180.0
            lat = ((y - ymin) / dy) * 180.0 - 90.0
            lonlat[n] = (lon, lat)
        return lonlat

    def draw_globe(self, path=None):
        # Küre modunda çizim:
        # - Tüm node/edge projeksiyonla 2D düzleme aktarılır
        # - Görünmeyen (arka tarafta kalan) noktalar çizilmez
        self.ax.clear()
        self.ax.set_axis_off()
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        R = 1.0 * self.globe_R

        # Küre arka plan dairesi
        sphere = Circle((0, 0), radius=R, facecolor=self.colors["panel"], edgecolor=self.colors["border"], zorder=0)
        self.ax.add_patch(sphere)

        lonlat = self._compute_lonlat_from_pos()
        pts = {}
        vis = {}
        for n, (lon, lat) in lonlat.items():
            x, y, visible = self._lonlat_to_ortho(lon, lat, self.globe_lon, self.globe_lat, R)
            pts[n] = (x, y)
            vis[n] = visible

        # Edge'ler (iki ucu da görünüyorsa)
        segs = []
        for u, v in self.G.edges():
            if vis.get(u) and vis.get(v):
                segs.append([pts[u], pts[v]])
        if segs:
            lc = LineCollection(segs, colors=self.edge_color, linewidths=0.5, alpha=0.2, zorder=1)
            self.ax.add_collection(lc)

        # Node'lar (görünenler)
        xs = [pts[n][0] for n in self.G.nodes() if vis.get(n)]
        ys = [pts[n][1] for n in self.G.nodes() if vis.get(n)]
        if xs and ys:
            self.ax.scatter(xs, ys, s=20, c=self.node_color, alpha=0.7, zorder=2)

        # Kaynak/hedef vurgusu
        if self.s_node in pts and vis.get(self.s_node):
            x, y = pts[self.s_node]
            self.ax.scatter([x], [y], s=120, c=self.src_color, zorder=3)
        if self.d_node in pts and vis.get(self.d_node):
            x, y = pts[self.d_node]
            self.ax.scatter([x], [y], s=120, c=self.dst_color, zorder=3)

        # Yol vurgusu (görünen segmentler)
        if path and len(path) >= 2:
            p_segs = []
            ix, iy = [], []
            for u, v in zip(path, path[1:]):
                if vis.get(u) and vis.get(v):
                    p_segs.append([pts[u], pts[v]])

            for n in path:
                if vis.get(n) and n not in [self.s_node, self.d_node]:
                    ix.append(pts[n][0])
                    iy.append(pts[n][1])

            if p_segs:
                self.ax.add_collection(LineCollection(p_segs, colors=self.path_color, linewidths=5.0, alpha=0.4, zorder=4))
                self.ax.add_collection(LineCollection(p_segs, colors=self.path_color, linewidths=2.5, alpha=1.0, zorder=5))

            if ix:
                self.ax.scatter(ix, iy, s=60, c=self.intermediate_color, alpha=1.0, zorder=6, edgecolors="white")

        self.ax.set_xlim(-R * 1.1, R * 1.1)
        self.ax.set_ylim(-R * 1.1, R * 1.1)
        self.canvas.draw()

    def _on_mouse_press(self, event):
        # Düz modda: sol çift tık -> expanded graph
        if event.dblclick and event.button == 1:
            if event.inaxes == self.ax and getattr(self, 'view_mode', 'düz') == 'düz':
                self.open_expanded_graph(self.current_path)
                return

        # Küre modunda: sol bas -> sürükleme başlat
        if getattr(self, 'view_mode', 'düz') != 'küre':
            return
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            self._globe_dragging = True
            self._globe_last_xy = (event.x, event.y)

    def _on_mouse_motion(self, event):
        # Küre modunda: mouse drag ile lon/lat güncelle (döndürme)
        if not self._globe_dragging:
            return
        if event.inaxes != self.ax:
            return
        x, y = event.x, event.y
        lx, ly = self._globe_last_xy
        dx = x - lx
        dy = y - ly
        self.globe_lon += dx * 0.3
        self.globe_lat -= dy * 0.18
        self.globe_lat = max(-89.0, min(89.0, self.globe_lat))
        self._globe_last_xy = (x, y)
        self.draw_graph(path=self.current_path)

    def _on_mouse_release(self, event):
        # Küre modunda: drag bitir
        self._globe_dragging = False
        self._globe_last_xy = None

    def _on_scroll(self, event):
        # Küre modunda: mouse wheel ile zoom
        if getattr(self, 'view_mode', 'düz') != 'küre':
            return
        step = getattr(event, 'step', None)
        if step is None:
            if getattr(event, 'button', None) == 'up':
                step = 1
            elif getattr(event, 'button', None) == 'down':
                step = -1
            else:
                step = 1
        if step > 0:
            self.globe_R *= 1.1
        else:
            self.globe_R /= 1.1
        self.globe_R = max(0.3, min(3.0, self.globe_R))
        self.draw_graph(path=self.current_path)

    # ---------------- METRICS / DETAILS PANEL ----------------
    def build_metrics(self):
        # Alt panel: Yol Detayları
        # - Kopyala: Detay metnini clipboard'a alır
        # - Gizle/Göster: paneli collapse/expand yapar

        self.details_container = ctk.CTkFrame(
            self.main_frame, fg_color=self.colors["panel"], corner_radius=12,
            border_width=1, border_color=self.colors["border"]
        )
        self.details_container.grid(row=1, column=0, sticky="nsew", padx=(12, 12), pady=(12, 12))
        self.main_frame.rowconfigure(1, weight=0)

        header_frame = ctk.CTkFrame(self.details_container, fg_color="transparent")
        header_frame.pack(fill="x", padx=8, pady=(8, 4))

        self.details_title = ctk.CTkLabel(
            header_frame, text="Yol Detayları",
            text_color=self.colors["text"],
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.details_title.pack(side="left")

        btn_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        btn_frame.pack(side="right")

        self.copy_btn = ctk.CTkButton(btn_frame, text="Kopyala", command=self.copy_details_to_clipboard, corner_radius=8, width=80)
        self.copy_btn.pack(side="right", padx=(6, 0))

        self.details_toggle_btn = ctk.CTkButton(btn_frame, text="Gizle", command=self.toggle_details, corner_radius=8, width=80)
        self.details_toggle_btn.pack(side="right")

        self.details_scroll = ctk.CTkScrollableFrame(self.details_container, width=520, height=240)
        self.details_scroll.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.details_visible = True

    def populate_path_details(self, result):
        # Sonuç metrics içinden hops çekerek:
        # - üstte özet kutusu
        # - altta her hop/edge detay kartları
        for w in self.details_scroll.winfo_children():
            w.destroy()

        hops = result.metrics.get("hops", [])
        if not hops:
            ctk.CTkLabel(self.details_scroll, text="Yol bulundu ancak detay yok.", text_color=self.colors["muted"]).pack(padx=8, pady=8)
            return

        self.last_result = result
        m = result.metrics

        algo_name = getattr(result, 'algo_name', 'Bilinmeyen Algoritma')
        weights = getattr(result, 'weights', (0.0, 0.0, 0.0))
        w1, w2, w3 = weights

        # Metrik özetleri
        total_delay_ms = m.get('total_delay_ms', 0.0)
        total_reliability_pct = m.get('total_reliability_pct') if m.get('total_reliability_pct') is not None else (m.get('path_reliability', 0.0) * 100.0)
        reliability_cost = m.get('reliability_cost', 0.0)
        resource_cost = m.get('resource_cost', 0.0)
        total_cost = m.get('total_cost', 0.0)

        # Genel özet kutusu
        summary_frame = ctk.CTkFrame(self.details_scroll, fg_color=self.colors["bg"], corner_radius=8, border_width=1, border_color=self.colors["border"])
        summary_frame.pack(fill="x", padx=8, pady=(8, 12))

        summary_title = ctk.CTkLabel(
            summary_frame,
            text="Genel Yol Özeti",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=self.colors["text"]
        )
        summary_title.pack(padx=12, pady=(10, 6), anchor="w")

        summary_lines = [
            f"Algoritma: {algo_name}",
            f"Ağırlıklar: Wdelay={w1:.2f} | Wreliability={w2:.2f} | Wresource={w3:.2f}",
            f"Toplam Gecikme: {total_delay_ms:.2f} ms",
            f"Toplam Güvenilirlik: {total_reliability_pct:.2f}%  (Reliability Cost: {reliability_cost:.4f})",
            f"Kaynak Maliyeti: {resource_cost:.2f}",
            f"Toplam Maliyet: {total_cost:.4f}"
        ]

        for line in summary_lines:
            summary_label = ctk.CTkLabel(
                summary_frame,
                text=line,
                font=ctk.CTkFont(size=11),
                text_color=self.colors["text"],
                anchor="w",
                justify="left"
            )
            summary_label.pack(padx=12, pady=2, anchor="w")

        separator = ctk.CTkFrame(summary_frame, fg_color=self.colors["border"], height=1)
        separator.pack(fill="x", padx=12, pady=(8, 8))

        detail_title = ctk.CTkLabel(
            self.details_scroll,
            text="Yol Detayları (Düğüm / Kenar Bazlı)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=self.colors["text"],
            anchor="w"
        )
        detail_title.pack(padx=8, pady=(0, 6), anchor="w")

        # Her hop için detay kartı
        for idx, hop in enumerate(hops):
            frame = ctk.CTkFrame(self.details_scroll, fg_color=self.colors["bg"], corner_radius=8)
            frame.pack(fill="x", padx=8, pady=6)
            frame.grid_columnconfigure(0, weight=1)
            frame.grid_columnconfigure(1, weight=1)

            if idx == 0:
                # İlk hop: kaynak düğüm detayı
                title = f"{idx}. Node {hop['node']} (Kaynak)"
                ctk.CTkLabel(frame, text=title, text_color=self.colors["text"], font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))
                details_left = ctk.CTkLabel(
                    frame,
                    text=f"proc_delay: {hop['proc_delay']:.2f} ms   |   reliability: {hop['node_reliability']:.6f}",
                    text_color=self.colors["muted"],
                    justify="left"
                )
                details_left.grid(row=1, column=0, sticky="w", padx=8, pady=(0, 8))
                ctk.CTkLabel(frame, text="", text_color=self.colors["muted"]).grid(row=0, column=1, rowspan=2, sticky="e", padx=8)
            else:
                # Diğer hop'lar: edge detayı (from->to) + maliyet bileşenleri
                e = hop['edge']
                c = hop['costs']
                title = f"{idx}. {e['from']} -> {e['to']}"
                ctk.CTkLabel(frame, text=title, text_color=self.colors["text"], font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))

                left_attrs = f"Delay: {e.get('link_delay_ms', e.get('link_delay', 0.0)):.2f} ms   |   BW: {e.get('bandwidth_mbps', e.get('bandwidth', 0.0)):.1f} Mbps   |   Rel: {e.get('link_reliability', 0.0):.4f}"
                ctk.CTkLabel(frame, text=left_attrs, text_color=self.colors["muted"], justify="left").grid(row=1, column=0, sticky="w", padx=8, pady=(0, 8))

                right_txt = (
                    f"node_proc: {hop['proc_delay']:.2f} ms\nnode_rel: {hop['node_reliability']:.6f}\n"
                    f"costs: delay={c['delay_cost']:.2f}, rel={c['rel_cost']:.6f}, res={c['resource_cost']:.6f}\n"
                    f"total: {c['total_cost']:.6f}"
                )
                ctk.CTkLabel(frame, text=right_txt, text_color=self.colors["text"], justify="right").grid(row=0, column=1, rowspan=2, sticky="e", padx=8, pady=8)

        # Panel kapalıysa otomatik aç
        if not self.details_visible:
            self.toggle_details(show=True)

    def copy_details_to_clipboard(self):
        # Son hesaplanan yol detaylarını metin olarak panoya kopyalar
        if not hasattr(self, 'last_result') or not self.last_result:
            messagebox.showwarning("Hata", "Kopyalanacak yol detayı yok.")
            return

        hops = self.last_result.metrics.get('hops', [])
        m = self.last_result.metrics
        lines = [
            f"Toplam Gecikme: {m.get('total_delay_ms', 0.0):.2f} ms | "
            f"Toplam Güvenilirlik: {m.get('total_reliability_pct', 0.0):.2f}% | "
            f"Kaynak: {m.get('resource_cost', 0.0):.2f} | "
            f"Toplam: {m.get('total_cost', 0.0):.2f}",
            ""
        ]

        for idx, hop in enumerate(hops):
            if idx == 0:
                lines.append(f"{idx}. Node {hop['node']} (Kaynak) - proc_delay={hop['proc_delay']:.2f} ms, reliability={hop['node_reliability']:.6f}")
            else:
                e = hop['edge']
                c = hop['costs']
                lines.append(f"{idx}. {e['from']} -> {e['to']} | Delay={e.get('link_delay_ms', e.get('link_delay', 0.0)):.2f} ms | BW={e.get('bandwidth_mbps', e.get('bandwidth', 0.0)):.1f} Mbps | LinkRel={e.get('link_reliability', 0.0):.6f}")
                lines.append(f"    node_proc={hop['proc_delay']:.2f} ms | node_rel={hop['node_reliability']:.6f} | costs: delay={c['delay_cost']:.2f}, rel={c['rel_cost']:.6f}, res={c['resource_cost']:.6f}, total={c['total_cost']:.6f}")

        text = "\n".join(lines)
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
            messagebox.showinfo("Bilgi", "Yol detayları panoya kopyalandı.")
        except Exception as e:
            messagebox.showwarning("Hata", f"Panoya kopyalanamadı: {e}")

    def toggle_details(self, show=None):
        # Detay panelini gizle/göster
        if show is None:
            show = not self.details_visible

        if not show:
            self.details_scroll.pack_forget()
            self.details_toggle_btn.configure(text="Göster")
            self.details_visible = False
        else:
            self.details_scroll.pack(fill="both", expand=True, padx=8, pady=(0, 8))
            self.details_toggle_btn.configure(text="Gizle")
            self.details_visible = True

    # ---------------- ACTIONS ----------------
    def on_regenerate(self):
        # Yeni seed ile grafı yeniden üretir (topoloji değişir)
        self.seed += 1
        self.G = generate_graph(self.n, self.p, self.seed)
        self.pos = compute_layout(self.G, seed=self.seed)
        self.draw_graph(path=None)
        messagebox.showinfo("Bilgi", f"Grafik yenilendi. Seed={self.seed}")

    def on_compare_algorithms(self):
        # Karşılaştırma penceresini açar (tek S/D için çoklu run + özet)
        if self.s_node == self.d_node:
            messagebox.showwarning("Uyarı", "Kaynak ve hedef aynı olamaz.")
            return

        w1 = float(self.w_delay.get())
        w2 = float(self.w_rel.get())
        w3 = float(self.w_res.get())
        s = w1 + w2 + w3

        if s <= 0:
            messagebox.showwarning("Uyarı", "Ağırlıkların toplamı 0 olamaz.")
            return

        # Normalize açıksa ağırlıklar 1'e ölçeklenir
        if self.normalize_var.get():
            w1, w2, w3 = w1 / s, w2 / s, w3 / s

        demand_mbps = float(self.demand_mbps_slider.get())

        ComparisonWindow(
            self,
            self.G,
            self.pos,
            self.s_node,
            self.d_node,
            w1, w2, w3,
            demand_mbps,
            self.colors
        )

    def on_calculate(self):
        # Hesapla:
        # - ağırlıkları (normalize opsiyonuna göre) hazırlar
        # - adapter üzerinden algoritmayı çalıştırır
        # - path'i çizer, detay panelini doldurur
        if self.s_node == self.d_node:
            messagebox.showwarning("Uyarı", "Kaynak ve hedef aynı olamaz.")
            return

        # UI'de "loading" hissi için butonu disable + text değişimi
        self.btn_calc.configure(state="disabled", text="Hesaplanıyor...")
        self.update_idletasks()

        try:
            w1 = float(self.w_delay.get())
            w2 = float(self.w_rel.get())
            w3 = float(self.w_res.get())
            s = w1 + w2 + w3

            if s <= 0:
                messagebox.showwarning("Uyarı", "Ağırlıkların toplamı 0 olamaz.")
                return

            # Normalize açık: slider'ları bile normalize edilmiş değere çekiyoruz
            if self.normalize_var.get():
                w1, w2, w3 = w1 / s, w2 / s, w3 / s
                self.w_delay.set(w1)
                self.w_rel.set(w2)
                self.w_res.set(w3)
                self.on_weight_change()

                # ✅ set() command tetiklemez -> label'ları manuel güncelle
                self.w_delay_lbl.configure(text=f"{w1:.2f}")
                self.w_rel_lbl.configure(text=f"{w2:.2f}")
                self.w_res_lbl.configure(text=f"{w3:.2f}")

            else:
                # Normalize kapalıysa kullanıcı toplamı 1 yapmalı
                if abs(s - 1.0) > 1e-3:
                    messagebox.showwarning("Uyarı", "Normalize kapalıyken ağırlıkların toplamı 1 olmalı.")
                    return

            # Algoritmayı çalıştır
            try:
                algo_name = self.algo_menu.get()
                algo_params = self.collect_algo_params()
                result_dict = run_algorithm(algo_name, self.G, self.s_node, self.d_node, w1, w2, w3, algo_params)
            except Exception as e:
                messagebox.showwarning("Hata", f"Algoritma çalıştırılırken hata: {e}")
                return

            # Yol bulunamadıysa graf temizle
            if not result_dict.get('path'):
                messagebox.showwarning("Uyarı", result_dict.get('notes', 'Yol bulunamadı.'))
                self.draw_graph(path=None)
                return

            path = result_dict['path']
            self.draw_graph(path)

            # Detay/hops hazırlanması:
            # - Algoritma metrics içinde hops yoksa topology.build_hops_for_path ile üret
            m = result_dict.get('metrics', {})
            try:
                if 'hops' not in m or not m.get('hops'):
                    hops = build_hops_for_path(self.G, path, w1, w2, w3, self.s_node, self.d_node)
                    m['hops'] = hops

                # UI populate fonksiyonunun beklediği alanları doldurmak için "fake_result" objesi
                fake_result = type('R', (), {})()
                fake_result.metrics = m
                fake_result.path = path
                fake_result.algo_name = algo_name
                fake_result.weights = (w1, w2, w3)
                self.populate_path_details(fake_result)
            except Exception as e:
                messagebox.showwarning("Hata", f"Detaylar hazırlanırken hata: {e}")

        finally:
            # Butonu eski haline getir
            self.btn_calc.configure(state="normal", text="HESAPLA")
            self.update_idletasks()


class ComparisonWindow(ctk.CTkToplevel):
    """Algoritma karşılaştırma penceresi:
    - Aynı S/D ve ağırlıklar için her algoritmayı birden fazla kez çalıştırır (num_runs)
    - Özet tablo + run bazlı tablo + her algoritmanın en iyi yol görselleştirmesi
    """

    def __init__(self, parent, G, pos, src, dst, w_delay, w_rel, w_res, demand_mbps, colors):
        super().__init__(parent)
        self.title("Algoritma Karşılaştırması")
        self.geometry("1400x900")
        self.configure(fg_color=colors["bg"])

        # Deney bağlamı
        self.G = G
        self.pos = pos
        self.src = src
        self.dst = dst
        self.w_delay = w_delay
        self.w_rel = w_rel
        self.w_res = w_res
        self.demand_mbps = demand_mbps
        self.colors = colors

        # Progress state (thread ile hesap)
        self.is_running = False
        self.progress_var = ctk.DoubleVar(value=0.0)

        # UI kur
        self.build_ui()

        # Pencere açıldıktan kısa süre sonra hesaplamayı başlat
        self.after(100, self.start_comparison)

    def build_ui(self):
        # Üst başlık + context + progress bar
        header = ctk.CTkFrame(self, fg_color=self.colors["panel"], corner_radius=12)
        header.pack(fill="x", padx=20, pady=(20, 10))

        title = ctk.CTkLabel(
            header,
            text="Algoritma Karşılaştırması",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=self.colors["text"]
        )
        title.pack(padx=16, pady=12)

        info_text = (
            f"Kaynak: {self.src} | Hedef: {self.dst} | "
            f"W_delay={self.w_delay:.2f} | W_rel={self.w_rel:.2f} | W_res={self.w_res:.2f} | "
            f"Demand: {self.demand_mbps:.0f} Mbps"
        )
        info_label = ctk.CTkLabel(
            header,
            text=info_text,
            font=ctk.CTkFont(size=12),
            text_color=self.colors["muted"]
        )
        info_label.pack(padx=16, pady=(0, 12))

        self.progress_bar = ctk.CTkProgressBar(header, variable=self.progress_var, width=600)
        self.progress_bar.pack(padx=16, pady=(0, 12))

        self.status_label = ctk.CTkLabel(
            header,
            text="Hazırlanıyor...",
            font=ctk.CTkFont(size=12),
            text_color=self.colors["muted"]
        )
        self.status_label.pack(padx=16, pady=(0, 12))

        # Scrollable içerik (tablo + grafikler)
        main_scroll = ctk.CTkScrollableFrame(self, fg_color=self.colors["bg"])
        main_scroll.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # Ekranın neyi temsil ettiğini açıklayan not
        context_frame = ctk.CTkFrame(main_scroll, fg_color=self.colors["panel"], corner_radius=12)
        context_frame.pack(fill="x", padx=0, pady=(0, 12))

        context_label = ctk.CTkLabel(
            context_frame,
            text="ℹ️ Bu ekran, tek bir (Kaynak, Hedef) çifti için yapılan örnek bir karşılaştırmayı göstermektedir. "
                 "Çoklu (S, D) deneyleri rapor bölümünde sunulmuştur.",
            font=ctk.CTkFont(size=11),
            text_color=self.colors["muted"],
            justify="left",
            wraplength=1200
        )
        context_label.pack(padx=16, pady=12, anchor="w")

        # Özet tablo container
        self.summary_frame = ctk.CTkFrame(main_scroll, fg_color=self.colors["panel"], corner_radius=12)
        self.summary_frame.pack(fill="x", padx=0, pady=(0, 12))

        summary_title = ctk.CTkLabel(
            self.summary_frame,
            text="Özet Tablo (Algoritma Başına)",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=self.colors["text"]
        )
        summary_title.pack(padx=16, pady=(12, 8), anchor="w")

        self.summary_table_frame = ctk.CTkFrame(self.summary_frame, fg_color="transparent")
        self.summary_table_frame.pack(fill="x", padx=16, pady=(0, 12))

        # Run bazlı tablo container
        self.runs_frame = ctk.CTkFrame(main_scroll, fg_color=self.colors["panel"], corner_radius=12)
        self.runs_frame.pack(fill="both", expand=True, padx=0, pady=(0, 12))

        runs_title = ctk.CTkLabel(
            self.runs_frame,
            text="Tüm Çalıştırmalar",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=self.colors["text"]
        )
        runs_title.pack(padx=16, pady=(12, 8), anchor="w")

        self.runs_table_frame = ctk.CTkScrollableFrame(self.runs_frame, height=300)
        self.runs_table_frame.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        # En iyi yollar + görselleştirme sekmeleri
        self.paths_frame = ctk.CTkFrame(main_scroll, fg_color=self.colors["panel"], corner_radius=12)
        self.paths_frame.pack(fill="both", expand=True, padx=0, pady=(0, 12))

        paths_title = ctk.CTkLabel(
            self.paths_frame,
            text="En İyi Yollar Görselleştirmesi (Algoritma Başına)",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=self.colors["text"]
        )
        paths_title.pack(padx=16, pady=(12, 8), anchor="w")

        self.algo_names = ["ACO (Ant Colony)", "Genetik (GA)", "Q-Learning", "Simulated Annealing (SA)"]

        self.path_tabs = ctk.CTkTabview(self.paths_frame)
        self.path_tabs.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        self.tab_frames = {}
        for algo_name in self.algo_names:
            tab = self.path_tabs.add(algo_name)
            self.tab_frames[algo_name] = tab

        self.path_figures = {}

        self.paths_content = ctk.CTkFrame(self.paths_frame, fg_color="transparent")
        self.paths_content.pack(fill="x", padx=16, pady=(0, 12))

    def start_comparison(self):
        # UI donmaması için thread ile çalıştır
        if self.is_running:
            return

        self.is_running = True
        self.status_label.configure(text="Karşılaştırma başlatılıyor...")
        self.progress_var.set(0.0)

        import threading
        thread = threading.Thread(target=self.run_comparison, daemon=True)
        thread.start()

    def run_comparison(self):
        # Background thread:
        # - compare_algorithms adapter fonksiyonunu çağırır
        # - UI güncellemeleri after(0, ...) ile main thread'e bırakılır
        try:
            self.after(0, lambda: self.status_label.configure(text="Algoritmalar çalıştırılıyor... (5 çalıştırma/algoritma)"))
            self.after(0, lambda: self.progress_var.set(0.1))

            # Her algoritmaya demand paramı geçmek için default_params sözlüğü hazırlanır
            default_params = {}
            for algo_name in ["ACO (Ant Colony)", "Genetik (GA)", "Q-Learning", "Simulated Annealing (SA)"]:
                default_params[algo_name] = {
                    'demand_mbps': self.demand_mbps,
                    'demand_bw': self.demand_mbps
                }

            results = compare_algorithms(
                self.G,
                self.src,
                self.dst,
                self.w_delay,
                self.w_rel,
                self.w_res,
                num_runs=5,
                default_params=default_params,
                demand_mbps=self.demand_mbps
            )

            self.after(0, lambda: self.progress_var.set(0.9))
            self.after(0, lambda: self.status_label.configure(text="Sonuçlar hazırlanıyor..."))

            self.after(0, lambda: self.display_results(results))

            self.after(0, lambda: self.progress_var.set(1.0))
            self.after(0, lambda: self.status_label.configure(text="Tamamlandı!"))

        except Exception as e:
            self.after(0, lambda: self.status_label.configure(text=f"Hata: {str(e)}"))
            messagebox.showerror("Hata", f"Karşılaştırma sırasında hata oluştu:\n{str(e)}")
        finally:
            self.is_running = False

    def display_results(self, results):
        # Adapter çıktısını UI tablolarına basar.
        # results beklenen anahtarlar:
        # - "summary_table": algoritma bazlı ortalamalar
        # - "runs_table": her run'ın detayları
        # - "best_paths_by_algo": her algoritmanın en iyi path'i
        summary_table = results["summary_table"]

        headers = ["Algoritma", "Ort. Gecikme (ms)", "Ort. Güvenilirlik Maliyeti", "Ort. Kaynak Maliyeti",
                   "Ort. Toplam Maliyet", "Std. Sapma", "En İyi", "En Kötü", "Ort. Süre (ms)"]
        for col, header in enumerate(headers):
            lbl = ctk.CTkLabel(
                self.summary_table_frame,
                text=header,
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color=self.colors["text"],
                width=130
            )
            lbl.grid(row=0, column=col, padx=3, pady=4, sticky="w")

        for row_idx, row_data in enumerate(summary_table, start=1):
            algo_name = row_data[0]
            avg_delay = row_data[1]
            avg_reliability_cost = row_data[2]
            avg_resource_cost = row_data[3]
            avg_total_cost = row_data[4]
            std_cost = row_data[5]
            best_cost = row_data[6]
            worst_cost = row_data[7]
            avg_runtime = row_data[8]

            algo_lbl = ctk.CTkLabel(
                self.summary_table_frame,
                text=algo_name,
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color=self.colors["btn"],
                width=130
            )
            algo_lbl.grid(row=row_idx, column=0, padx=3, pady=4, sticky="w")

            values = [avg_delay, avg_reliability_cost, avg_resource_cost, avg_total_cost,
                      std_cost, best_cost, worst_cost, avg_runtime]

            for col_idx, value in enumerate(values, start=1):
                if col_idx == 8:
                    text = f"{value:.2f}"
                elif col_idx == 5:
                    text = f"{value:.4f}*" if value == 0.0 else f"{value:.4f}"
                elif col_idx in [1, 2, 3, 4, 6, 7]:
                    text = "∞" if value == float('inf') else f"{value:.4f}"
                else:
                    text = str(value)

                lbl = ctk.CTkLabel(
                    self.summary_table_frame,
                    text=text,
                    font=ctk.CTkFont(size=10),
                    text_color=self.colors["text"],
                    width=130
                )
                lbl.grid(row=row_idx, column=col_idx, padx=3, pady=4, sticky="w")

        std_note = ctk.CTkLabel(
            self.summary_frame,
            text="* Standart sapma 0 ise, algoritma deterministik olarak çalışmaktadır.",
            font=ctk.CTkFont(size=9),
            text_color=self.colors["muted"]
        )
        std_note.pack(padx=16, pady=(0, 12), anchor="w")

        runs_table = results["runs_table"]

        run_headers = ["Algoritma", "Run", "Gecikme", "Güvenilirlik", "Kaynak", "Toplam", "Süre (ms)", "Yol"]
        for col, header in enumerate(run_headers):
            lbl = ctk.CTkLabel(
                self.runs_table_frame,
                text=header,
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color=self.colors["text"],
                width=120
            )
            lbl.grid(row=0, column=col, padx=2, pady=2, sticky="w")

        max_rows = min(50, len(runs_table))
        for row_idx, row_data in enumerate(runs_table[:max_rows], start=1):
            for col_idx, value in enumerate(row_data):
                if col_idx == 6:
                    text = f"{value:.2f}"
                elif col_idx == 7:
                    path_str = " -> ".join(map(str, value[:5]))
                    if len(value) > 5:
                        path_str += "..."
                    text = path_str if value else "Yok"
                elif col_idx in [2, 3, 4, 5]:
                    text = "∞" if value == float('inf') else f"{value:.4f}"
                else:
                    text = str(value)

                lbl = ctk.CTkLabel(
                    self.runs_table_frame,
                    text=text,
                    font=ctk.CTkFont(size=9),
                    text_color=self.colors["text"],
                    width=120
                )
                lbl.grid(row=row_idx, column=col_idx, padx=2, pady=2, sticky="w")

        if len(runs_table) > max_rows:
            info_lbl = ctk.CTkLabel(
                self.runs_table_frame,
                text=f"... ve {len(runs_table) - max_rows} satır daha",
                font=ctk.CTkFont(size=9),
                text_color=self.colors["muted"]
            )
            info_lbl.grid(row=max_rows + 1, column=0, columnspan=8, padx=2, pady=4)

        best_paths = results["best_paths_by_algo"]

        # Metin olarak en iyi path'leri listele
        for algo_name, path in best_paths.items():
            frame = ctk.CTkFrame(self.paths_content, fg_color=self.colors["bg"], corner_radius=8)
            frame.pack(fill="x", padx=4, pady=4)

            algo_lbl = ctk.CTkLabel(
                frame,
                text=f"{algo_name}:",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=self.colors["btn"]
            )
            algo_lbl.pack(side="left", padx=8, pady=8)

            if path:
                path_str = " -> ".join(map(str, path))
                path_lbl = ctk.CTkLabel(
                    frame,
                    text=path_str,
                    font=ctk.CTkFont(size=11),
                    text_color=self.colors["text"],
                    justify="left"
                )
                path_lbl.pack(side="left", padx=8, pady=8, fill="x", expand=True)
            else:
                no_path_lbl = ctk.CTkLabel(
                    frame,
                    text="Yol bulunamadı",
                    font=ctk.CTkFont(size=11),
                    text_color=self.colors["muted"]
                )
                no_path_lbl.pack(side="left", padx=8, pady=8)

        # Sekmeler içinde her algoritma için graf çizimi
        self.draw_path_visualizations(best_paths)

    def draw_path_visualizations(self, best_paths):
        # Her sekmede (algoritma tabında) en iyi yolu graf üzerinde gösterir.
        # Bu çizim, ana ekrandaki stil ile aynı mantığı izler.
        edge_color = "#A7B0C0"
        node_color = "#A9C9F7"
        path_color = "#FFC107"
        src_color = "#7BDCB5"
        dst_color = "#FF8A8A"
        intermediate_color = "#0099FF"

        for algo_name in self.algo_names:
            if algo_name not in self.tab_frames:
                continue

            tab = self.tab_frames[algo_name]
            path = best_paths.get(algo_name, [])

            fig, ax = plt.subplots(figsize=(10, 7), dpi=100)
            fig.patch.set_facecolor(self.colors["panel"])

            ax.set_aspect('auto')
            ax.set_axis_off()
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

            nx.draw_networkx_edges(
                self.G, self.pos, ax=ax,
                width=0.3, edge_color=edge_color, alpha=0.2
            )
            nx.draw_networkx_nodes(
                self.G, self.pos, ax=ax,
                node_size=30, node_color=node_color, alpha=0.7
            )

            nx.draw_networkx_labels(
                self.G, self.pos,
                labels={n: str(n) for n in self.G.nodes()},
                font_size=6,
                font_color=self.colors.get("node_label", "#CBD5E1"),
                ax=ax
            )

            # Kaynak/hedef vurgusu
            nx.draw_networkx_nodes(
                self.G, self.pos, ax=ax,
                nodelist=[self.src], node_size=120, node_color=src_color
            )
            nx.draw_networkx_nodes(
                self.G, self.pos, ax=ax,
                nodelist=[self.dst], node_size=120, node_color=dst_color
            )

            # Yol vurgusu
            if path and len(path) >= 2:
                edges = list(zip(path, path[1:]))

                nx.draw_networkx_edges(
                    self.G, self.pos, ax=ax,
                    edgelist=edges, width=5.0, edge_color=path_color, alpha=0.4
                )
                nx.draw_networkx_edges(
                    self.G, self.pos, ax=ax,
                    edgelist=edges, width=2.5, edge_color=path_color, alpha=1.0
                )

                inter_nodes = [n for n in path if n != self.src and n != self.dst]
                if inter_nodes:
                    nx.draw_networkx_nodes(
                        self.G, self.pos, ax=ax,
                        nodelist=inter_nodes, node_size=60,
                        node_color=intermediate_color, alpha=1.0, edgecolors="white"
                    )
            else:
                # Path yoksa sekmede ortada mesaj
                ax.text(0.5, 0.5, "Yol bulunamadı",
                        transform=ax.transAxes,
                        ha='center', va='center',
                        fontsize=14, color=self.colors["muted"])

            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

            self.path_figures[algo_name] = (fig, ax, canvas)


# ------------------------------------------------------------
# Program entrypoint
# ------------------------------------------------------------
if __name__ == "__main__":
    # RoutingApp ana pencereyi oluştur ve event loop'u başlat
    app = RoutingApp()
    app.mainloop()
