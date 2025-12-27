# ui_main.py
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
from tkinter import messagebox
import math
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

from topology import generate_graph, compute_layout, build_hops_for_path

# Algorithms adapter: import helper functions with safe fallbacks if adapter missing
try:
    from algorithms.adapter import list_algorithms, get_algorithm_meta, run as run_algorithm
except Exception:
    def list_algorithms():
        return ["ACO (Ant Colony)", "Genetik (GA)", "Q-Learning", "Simulated Annealing (SA)"]
    def get_algorithm_meta(name):
        return None
    def run_algorithm(name, G, src, dst, w_delay, w_rel, w_res, params):
        raise RuntimeError("Algorithms adapter not available")


ctk.set_appearance_mode("Dark")


class RoutingApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("QoS Multi-Objective Routing Optimization")
        self.geometry("1300x850")
        self.minsize(1200, 760)

        # ---------- THEME DICT ----------
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

        # Slider colors (soft)
        self.SOFT_BLUE = "#8CB6F5"    # hız
        self.SOFT_YELLOW = "#F6D08B"  # güvenlik
        self.SOFT_PINK = "#F2A1A1"    # kaynak

        # Graph colors
        self.edge_color = "#A7B0C0"
        self.node_color = "#A9C9F7"
        self.path_color = "#F6B26B"
        self.src_color = "#7BDCB5"
        self.dst_color = "#FF8A8A"

        # State
        self.theme = "Dark"
        self.colors = self.themes[self.theme]

        # Graph data
        self.seed = 42
        self.n = 250
        self.p = 0.40
        self.G = generate_graph(self.n, self.p, self.seed)
        self.pos = compute_layout(self.G, seed=self.seed)

        # Globe view state (for stylized globe visualization)
        self.view_mode = "düz"  # "düz" or "küre"
        self.globe_lon = 0.0     # degrees
        self.globe_lat = 0.0     # degrees
        self.globe_R = 1.0       # visual radius scaling
        self._globe_dragging = False
        self._globe_last_xy = None

        # Selected nodes
        self.s_node = 5
        self.d_node = 100

        # Node selector window state
        self.node_win = None
        self.node_scroll = None
        self.node_loading_lbl = None
        self.selecting_node = None
        self.filtered_nodes = list(range(self.n))
        self.node_after_id = None          # batch build after id
        self.search_debounce_id = None     # debounce after id ✅

        # ---------- LAYOUT ----------
        self.configure(fg_color=self.colors["bg"])
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=420, corner_radius=16, fg_color=self.colors["panel"])
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=(20, 10), pady=20)

        self.main_frame = ctk.CTkFrame(self, corner_radius=16, fg_color=self.colors["panel"])
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 20), pady=20)
        self.main_frame.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=1)

        # Build UI
        self.build_sidebar()
        self.build_plot()
        self.build_metrics()

        # Apply theme after all widgets exist
        self.apply_theme_to_widgets()

        self.draw_graph(path=None)

        # Enter -> Calculate
        self.bind("<Return>", lambda e: self.on_calculate())

        # Close safely
        self.protocol("WM_DELETE_WINDOW", self.on_app_close)

    # ---------------- APP CLOSE ----------------
    def on_app_close(self):
        self.cleanup_node_selector()
        self.destroy()

    # ---------------- THEME ----------------
    def change_theme(self, mode):
        self.theme = mode
        self.colors = self.themes[self.theme]
        ctk.set_appearance_mode(mode)

        self.configure(fg_color=self.colors["bg"])
        self.sidebar.configure(fg_color=self.colors["panel"])
        self.main_frame.configure(fg_color=self.colors["panel"])

        if hasattr(self, "fig"):
            self.fig.patch.set_facecolor(self.colors["panel"])

        self.apply_theme_to_widgets()
        self.draw_graph(path=None)

        if self.node_win is not None and self.node_win.winfo_exists():
            self.node_win.configure(fg_color=self.colors["panel"])
            self.node_win.lift()
            self.node_win.focus_force()

    def apply_theme_to_widgets(self):
        if hasattr(self, "title_lbl"):
            self.title_lbl.configure(text_color=self.colors["text"])
        if hasattr(self, "section_node_lbl"):
            self.section_node_lbl.configure(text_color=self.colors["text"])
        if hasattr(self, "section_weight_lbl"):
            self.section_weight_lbl.configure(text_color=self.colors["text"])
        if hasattr(self, "hint_lbl"):
            self.hint_lbl.configure(text_color=self.colors["muted"])

        if hasattr(self, "lbl_s"):
            self.lbl_s.configure(text_color=self.colors["text"])
        if hasattr(self, "lbl_d"):
            self.lbl_d.configure(text_color=self.colors["text"])

        if hasattr(self, "theme_menu"):
            self.theme_menu.configure(
                fg_color=self.colors["btn"],
                button_color=self.colors["btn"],
                button_hover_color=self.colors["btn_hover"],
                text_color=self.colors["panel"],
            )

        # Main buttons
        for name in ["btn_select_s", "btn_select_d", "btn_calc"]:
            if hasattr(self, name):
                getattr(self, name).configure(
                    fg_color=self.colors["btn"],
                    hover_color=self.colors["btn_hover"],
                    text_color=self.colors["panel"],
                )

        if hasattr(self, "btn_regen"):
            self.btn_regen.configure(
                fg_color=self.colors["panel"],
                text_color=self.colors["text"],
                border_color=self.colors["border"],
                hover_color=self.colors["bg"],
            )

        for name in ["w_delay_lbl", "w_rel_lbl", "w_res_lbl"]:
            if hasattr(self, name):
                getattr(self, name).configure(text_color=self.colors["muted"])

        for name in ["w_delay_title", "w_rel_title", "w_res_title"]:
            if hasattr(self, name):
                getattr(self, name).configure(text_color=self.colors["text"])

        if hasattr(self, "metrics"):
            self.metrics.configure(text_color=self.colors["text"])

        if hasattr(self, "normalize_cb"):
            self.normalize_cb.configure(text_color=self.colors["text"])

        if hasattr(self, "node_card"):
            self.node_card.configure(fg_color=self.colors["bg"], border_color=self.colors["border"])
        if hasattr(self, "node_card_hint"):
            self.node_card_hint.configure(text_color=self.colors["muted"])

        # Algo menu theme
        if hasattr(self, "algo_menu"):
            self.algo_menu.configure(
                fg_color=self.colors["btn"],
                button_color=self.colors["btn"],
                button_hover_color=self.colors["btn_hover"],
                text_color=self.colors["panel"],
            )

        # Param entries theme
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

        # Details panel
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
        # Scrollable content area to ensure buttons are reachable on small screens
        self.sidebar_scroll = ctk.CTkScrollableFrame(self.sidebar, corner_radius=12)
        self.sidebar_scroll.pack(fill="both", expand=True, padx=8, pady=8)

        self.title_lbl = ctk.CTkLabel(
            self.sidebar_scroll, text="Kontrol Paneli",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=self.colors["text"]
        )
        self.title_lbl.pack(padx=16, pady=(18, 10), anchor="w")

        self.theme_menu = ctk.CTkOptionMenu(
            self.sidebar_scroll, values=["Light", "Dark"],
            command=self.change_theme
        )
        self.theme_menu.set(self.theme)
        self.theme_menu.pack(padx=16, pady=(0, 14), fill="x")

        self.section_node_lbl = ctk.CTkLabel(
            self.sidebar_scroll, text="Düğüm Seçimi",
            text_color=self.colors["text"],
            font=ctk.CTkFont(size=14, weight="normal")
        )
        self.section_node_lbl.pack(padx=16, pady=(6, 6), anchor="w")

        self.node_card = ctk.CTkFrame(
            self.sidebar_scroll, fg_color=self.colors["bg"],
            corner_radius=14, border_width=1, border_color=self.colors["border"]
        )
        self.node_card.pack(padx=16, pady=(0, 12), fill="x")

        self.node_card_hint = ctk.CTkLabel(
            self.node_card,
            text="Kaynak ve hedef düğümleri seçin (S ≠ D).",
            font=ctk.CTkFont(size=12, weight="normal"),
            text_color=self.colors["muted"]
        )
        self.node_card_hint.pack(padx=12, pady=(10, 8), anchor="w")

        self.btn_select_s = ctk.CTkButton(
            self.node_card, text="Kaynak Düğüm Seç",
            command=lambda: self.open_node_selector("S"),
            corner_radius=12
        )
        self.btn_select_s.pack(padx=12, pady=4, fill="x")

        self.btn_select_d = ctk.CTkButton(
            self.node_card, text="Hedef Düğüm Seç",
            command=lambda: self.open_node_selector("D"),
            corner_radius=12
        )
        self.btn_select_d.pack(padx=12, pady=(0, 6), fill="x")

        self.lbl_s = ctk.CTkLabel(
            self.node_card, text=f"Seçilen S: {self.s_node}",
            font=ctk.CTkFont(size=12),
            text_color=self.colors["text"]
        )
        self.lbl_s.pack(padx=12, pady=(0, 2), anchor="w")

        self.lbl_d = ctk.CTkLabel(
            self.node_card, text=f"Seçilen D: {self.d_node}",
            font=ctk.CTkFont(size=12),
            text_color=self.colors["text"]
        )
        self.lbl_d.pack(padx=12, pady=(0, 10), anchor="w")

        self.section_weight_lbl = ctk.CTkLabel(
            self.sidebar_scroll, text="Optimizasyon Ağırlıkları",
            text_color=self.colors["text"],
            font=ctk.CTkFont(size=14, weight="normal")
        )
        self.section_weight_lbl.pack(padx=16, pady=(10, 6), anchor="w")

        self.w_delay_title, self.w_delay, self.w_delay_lbl = self.slider_with_value("Hız / Gecikme", 0.33, self.SOFT_BLUE)
        self.w_rel_title, self.w_rel, self.w_rel_lbl = self.slider_with_value("Güvenlik", 0.33, self.SOFT_YELLOW)
        self.w_res_title, self.w_res, self.w_res_lbl = self.slider_with_value("Kaynak Kontrolü", 0.34, self.SOFT_PINK)

        # Sum label
        self.sum_lbl = ctk.CTkLabel(
            self.sidebar,
            text="Ağırlık Toplamı: 1.00",
            text_color=self.colors["muted"],
            font=ctk.CTkFont(size=12)
        )
        self.sum_lbl.pack(padx=16, pady=(0, 8), anchor="w")

        # Hook slider changes to update sum label (and keep value labels)
        self.w_delay.configure(command=lambda v: (self.w_delay_lbl.configure(text=f"{float(v):.2f}"), self.on_weight_change()))
        self.w_rel.configure(command=lambda v: (self.w_rel_lbl.configure(text=f"{float(v):.2f}"), self.on_weight_change()))
        self.w_res.configure(command=lambda v: (self.w_res_lbl.configure(text=f"{float(v):.2f}"), self.on_weight_change()))

        # Algoritma seçimi
        algo_list = list_algorithms()

        ctk.CTkLabel(
            self.sidebar_scroll, text="Algoritma",
            text_color=self.colors["text"],
            font=ctk.CTkFont(size=12)
        ).pack(padx=16, pady=(8, 4), anchor="w")

        self.algo_menu = ctk.CTkOptionMenu(self.sidebar_scroll, values=algo_list, command=self.on_algo_change)
        self.algo_menu.set(algo_list[0])
        self.algo_menu.pack(padx=16, pady=(0, 8), fill="x")

        # View mode: Düz / Küre
        ctk.CTkLabel(self.sidebar_scroll, text="Görünüm", text_color=self.colors["text"], font=ctk.CTkFont(size=12)).pack(padx=16, pady=(6, 2), anchor="w")
        self.view_menu = ctk.CTkOptionMenu(self.sidebar_scroll, values=["Düz", "Küre"], command=self.on_view_change)
        self.view_menu.set("Düz")
        self.view_menu.pack(padx=16, pady=(0, 8), fill="x")

        # Algoritma param kontrolleri (dinamik)
        self.algo_params_frame = ctk.CTkFrame(self.sidebar_scroll, fg_color="transparent")
        self.algo_params_frame.pack(fill="x", padx=16, pady=(0, 8))
        self.algo_param_widgets = {}
        self.on_algo_change(self.algo_menu.get())

        self.normalize_var = ctk.BooleanVar(value=True)
        self.normalize_cb = ctk.CTkCheckBox(
            self.sidebar_scroll, text="Ağırlıkları Normalize Et (Toplam=1)",
            variable=self.normalize_var,
            text_color=self.colors["text"]
        )
        self.normalize_cb.pack(padx=16, pady=(6, 10), anchor="w")

        # ✅ HESAPLA button (fixed place)
        self.btn_calc = ctk.CTkButton(
            self.sidebar_scroll, text="HESAPLA",
            command=self.on_calculate, corner_radius=12
        )
        self.btn_calc.pack(padx=16, pady=(6, 6), fill="x")

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

        self.hint_lbl = ctk.CTkLabel(
            self.sidebar_scroll,
            text="İpucu: Düğüm listesi donma olmaması için parça parça yüklenir.\n"
                 "Yol bulunamazsa grafiği yenileyin veya S/D’yi değiştirin.",
            text_color=self.colors["muted"],
            font=ctk.CTkFont(size=12),
            justify="left"
        )
        self.hint_lbl.pack(padx=16, pady=(8, 0), anchor="w")

        # Initial sum update
        self.on_weight_change()

    def on_weight_change(self):
        w1 = float(self.w_delay.get())
        w2 = float(self.w_rel.get())
        w3 = float(self.w_res.get())
        s = w1 + w2 + w3

        self.sum_lbl.configure(text=f"Ağırlık Toplamı: {s:.2f}")
        if abs(s - 1.0) > 1e-3:
            # amber-ish warning
            self.sum_lbl.configure(text_color="#B45309")
        else:
            self.sum_lbl.configure(text_color=self.colors["muted"])

    def on_algo_change(self, name):
        for w in self.algo_params_frame.winfo_children():
            w.destroy()
        self.algo_param_widgets = {}

        meta = get_algorithm_meta(name)

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

            self.algo_param_widgets[p['name']] = (entry, p.get('type', 'str'))

        self.apply_theme_to_widgets()

    def collect_algo_params(self):
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
                params[name] = val
        return params

    def slider_with_value(self, title, default, color):
        frame = ctk.CTkFrame(self.sidebar_scroll, fg_color="transparent")
        frame.pack(fill="x", padx=16, pady=(0, 8))

        top = ctk.CTkFrame(frame, fg_color="transparent")
        top.pack(fill="x")

        title_lbl = ctk.CTkLabel(top, text=title, text_color=self.colors["text"], font=ctk.CTkFont(size=12))
        title_lbl.pack(side="left")

        val_lbl = ctk.CTkLabel(top, text=f"{default:.2f}", text_color=self.colors["muted"], font=ctk.CTkFont(size=12))
        val_lbl.pack(side="right")

        slider = ctk.CTkSlider(frame, from_=0, to=1, progress_color=color, button_color=color)
        slider.set(default)
        slider.pack(fill="x", pady=(6, 0))

        # NOTE: command is overridden in build_sidebar to also update sum label
        return title_lbl, slider, val_lbl

    # ---------------- NODE SELECTOR (FAST OPEN + DEBOUNCE + BATCH) ----------------
    def open_node_selector(self, mode):
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

        self.search_var.trace_add("write", lambda *_: self.schedule_rebuild_node_list())
        self.node_win.update_idletasks()
        self.after(10, self.schedule_rebuild_node_list)

    def schedule_rebuild_node_list(self, delay_ms: int = 120):
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
        if self.node_scroll is None:
            return

        self.search_debounce_id = None

        try:
            if self.node_after_id is not None:
                self.after_cancel(self.node_after_id)
        except:
            pass
        self.node_after_id = None

        for w in self.node_scroll.winfo_children():
            w.destroy()

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
        if self.selecting_node == "S":
            self.s_node = node_id
            self.lbl_s.configure(text=f"Seçilen S: {self.s_node}")
        else:
            self.d_node = node_id
            self.lbl_d.configure(text=f"Seçilen D: {self.d_node}")

        self.draw_graph(path=None)
        self.cleanup_node_selector()

    # ---------------- PLOT ----------------
    def build_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 7), dpi=100)
        self.fig.patch.set_facecolor(self.colors["panel"])
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_axis_off()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 6))

        # Mouse interactions for globe mode
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_motion)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)

    def draw_graph(self, path=None):
        # Globe mode takes over drawing
        if getattr(self, 'view_mode', 'düz') == 'küre':
            return self.draw_globe(path=path)

        self.ax.clear()
        self.ax.set_axis_off()

        nx.draw_networkx_edges(
            self.G, self.pos, ax=self.ax,
            width=0.15, edge_color=self.edge_color, alpha=0.08
        )
        nx.draw_networkx_nodes(
            self.G, self.pos, ax=self.ax,
            node_size=10, node_color=self.node_color, alpha=0.9
        )

        nx.draw_networkx_labels(
            self.G, self.pos,
            labels={n: str(n) for n in self.G.nodes()},
            font_size=6,
            font_color=self.colors["node_label"],
            ax=self.ax
        )

        nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax, nodelist=[self.s_node], node_size=90, node_color=self.src_color)
        nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax, nodelist=[self.d_node], node_size=90, node_color=self.dst_color)

        if path and len(path) >= 2:
            edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(self.G, self.pos, ax=self.ax, edgelist=edges, width=2.8, edge_color=self.path_color, alpha=0.95)
            nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax, nodelist=path, node_size=28, node_color=self.path_color, alpha=0.95)

        self.canvas.draw()

    # ---------------- METRICS ----------------
    def on_view_change(self, name):
        self.view_mode = "küre" if name == "Küre" else "düz"
        self.draw_graph(path=None)

    def _lonlat_to_ortho(self, lon, lat, lon0, lat0, R):
        # Orthographic projection (inputs in degrees)
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
        self.ax.clear()
        self.ax.set_axis_off()
        R = 1.0 * self.globe_R
        # Sphere background
        sphere = Circle((0, 0), radius=R, facecolor=self.colors["panel"], edgecolor=self.colors["border"], zorder=0)
        self.ax.add_patch(sphere)

        lonlat = self._compute_lonlat_from_pos()
        pts = {}
        vis = {}
        for n, (lon, lat) in lonlat.items():
            x, y, visible = self._lonlat_to_ortho(lon, lat, self.globe_lon, self.globe_lat, R)
            pts[n] = (x, y)
            vis[n] = visible

        # Draw edges (only visible segments)
        segs = []
        for u, v in self.G.edges():
            if vis.get(u) and vis.get(v):
                segs.append([pts[u], pts[v]])
        if segs:
            lc = LineCollection(segs, colors=self.edge_color, linewidths=0.5, alpha=0.08, zorder=1)
            self.ax.add_collection(lc)

        # Draw nodes
        xs = [pts[n][0] for n in self.G.nodes() if vis.get(n)]
        ys = [pts[n][1] for n in self.G.nodes() if vis.get(n)]
        if xs and ys:
            self.ax.scatter(xs, ys, s=14, c=self.node_color, zorder=2)

        # Source / dest
        if self.s_node in pts and vis.get(self.s_node):
            x, y = pts[self.s_node]
            self.ax.scatter([x], [y], s=90, c=self.src_color, zorder=3)
        if self.d_node in pts and vis.get(self.d_node):
            x, y = pts[self.d_node]
            self.ax.scatter([x], [y], s=90, c=self.dst_color, zorder=3)

        # Path highlight
        if path and len(path) >= 2:
            p_segs = []
            for u, v in zip(path, path[1:]):
                if vis.get(u) and vis.get(v):
                    p_segs.append([pts[u], pts[v]])
            if p_segs:
                plc = LineCollection(p_segs, colors=self.path_color, linewidths=2.5, alpha=0.95, zorder=4)
                self.ax.add_collection(plc)
                xs = [pts[n][0] for n in path if vis.get(n)]
                ys = [pts[n][1] for n in path if vis.get(n)]
                self.ax.scatter(xs, ys, s=28, c=self.path_color, zorder=4)

        self.ax.set_xlim(-R * 1.1, R * 1.1)
        self.ax.set_ylim(-R * 1.1, R * 1.1)
        self.canvas.draw()

    def _on_mouse_press(self, event):
        if getattr(self, 'view_mode', 'düz') != 'küre':
            return
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            self._globe_dragging = True
            self._globe_last_xy = (event.x, event.y)

    def _on_mouse_motion(self, event):
        if not self._globe_dragging:
            return
        if event.inaxes != self.ax:
            return
        x, y = event.x, event.y
        lx, ly = self._globe_last_xy
        dx = x - lx
        dy = y - ly
        # pixels -> degrees sensitivity
        self.globe_lon += dx * 0.3
        self.globe_lat -= dy * 0.18
        self.globe_lat = max(-89.0, min(89.0, self.globe_lat))
        self._globe_last_xy = (x, y)
        self.draw_graph(path=None)

    def _on_mouse_release(self, event):
        self._globe_dragging = False
        self._globe_last_xy = None

    def _on_scroll(self, event):
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
        self.draw_graph(path=None)

    def build_metrics(self):
        self.metrics = ctk.CTkLabel(
            self.main_frame,
            text="",
            text_color=self.colors["text"],
            font=ctk.CTkFont(size=14)
        )
        self.metrics.grid(row=1, column=0, pady=8, sticky="w")

        self.details_container = ctk.CTkFrame(
            self.main_frame, fg_color=self.colors["panel"], corner_radius=12,
            border_width=1, border_color=self.colors["border"]
        )
        self.details_container.grid(row=2, column=0, sticky="nsew", padx=(12, 12), pady=(6, 12))
        self.main_frame.rowconfigure(2, weight=0)

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
        for w in self.details_scroll.winfo_children():
            w.destroy()

        hops = result.metrics.get("hops", [])
        if not hops:
            ctk.CTkLabel(self.details_scroll, text="Yol bulundu ancak detay yok.", text_color=self.colors["muted"]).pack(padx=8, pady=8)
            return

        self.last_result = result
        m = result.metrics

        total_delay_ms = m.get('total_delay_ms', 0.0)
        total_reliability_pct = m.get('total_reliability_pct') if m.get('total_reliability_pct') is not None else (m.get('path_reliability', 0.0) * 100.0)

        header_text = (
            f"Toplam Gecikme: {total_delay_ms:.2f} ms   |   "
            f"Toplam Güvenilirlik: {total_reliability_pct:.2f}%   |   "
            f"Kaynak: {m.get('resource_cost', 0.0):.2f}   |   "
            f"Toplam: {m.get('total_cost', 0.0):.2f}"
        )
        ctk.CTkLabel(self.details_scroll, text=header_text, text_color=self.colors["text"], justify="left").pack(padx=8, pady=(6, 8), anchor="w")

        for idx, hop in enumerate(hops):
            frame = ctk.CTkFrame(self.details_scroll, fg_color=self.colors["bg"], corner_radius=8)
            frame.pack(fill="x", padx=8, pady=6)
            frame.grid_columnconfigure(0, weight=1)
            frame.grid_columnconfigure(1, weight=1)

            if idx == 0:
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

        if not self.details_visible:
            self.toggle_details(show=True)

    def copy_details_to_clipboard(self):
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
        self.seed += 1
        self.G = generate_graph(self.n, self.p, self.seed)
        self.pos = compute_layout(self.G, seed=self.seed)
        self.draw_graph(path=None)
        messagebox.showinfo("Bilgi", f"Grafik yenilendi. Seed={self.seed}")

    def on_calculate(self):
        if self.s_node == self.d_node:
            messagebox.showwarning("Uyarı", "Kaynak ve hedef aynı olamaz.")
            return

        # Button loading state
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

            if self.normalize_var.get():
                w1, w2, w3 = w1 / s, w2 / s, w3 / s
                self.w_delay.set(w1)
                self.w_rel.set(w2)
                self.w_res.set(w3)
                self.on_weight_change()
            else:
                if abs(s - 1.0) > 1e-3:
                    messagebox.showwarning("Uyarı", "Normalize kapalıyken ağırlıkların toplamı 1 olmalı.")
                    return

            try:
                algo_name = self.algo_menu.get()
                algo_params = self.collect_algo_params()
                result_dict = run_algorithm(algo_name, self.G, self.s_node, self.d_node, w1, w2, w3, algo_params)
            except Exception as e:
                messagebox.showwarning("Hata", f"Algoritma çalıştırılırken hata: {e}")
                return

            if not result_dict.get('path'):
                messagebox.showwarning("Uyarı", result_dict.get('notes', 'Yol bulunamadı.'))
                self.draw_graph(path=None)
                self.metrics.configure(text="")
                return

            path = result_dict['path']
            self.draw_graph(path)

            m = result_dict.get('metrics', {})
            total_delay = m.get('total_delay_ms') or m.get('total_delay') or 0.0
            total_reliability_pct = m.get('total_reliability_pct') or (m.get('path_reliability', 0.0) * 100.0) or 0.0
            resource_cost = m.get('resource_cost', 0.0)
            total_cost = m.get('total_cost', 0.0)

            self.metrics.configure(
                text=(f"[{algo_name}]  Wdelay={w1:.2f}  Wrel={w2:.2f}  Wres={w3:.2f}   |   "
                      f"Gecikme: {total_delay:.2f} ms   |   "
                      f"Güvenilirlik: {total_reliability_pct:.2f}%   |   "
                      f"Kaynak: {resource_cost:.2f}   |   "
                      f"Toplam: {total_cost:.2f}")
            )

            # Details / hops
            try:
                if 'hops' not in m or not m.get('hops'):
                    hops = build_hops_for_path(self.G, path, w1, w2, w3, self.s_node, self.d_node)
                    m['hops'] = hops

                fake_result = type('R', (), {})()
                fake_result.metrics = m
                fake_result.path = path
                self.populate_path_details(fake_result)
            except Exception as e:
                messagebox.showwarning("Hata", f"Detaylar hazırlanırken hata: {e}")

        finally:
            self.btn_calc.configure(state="normal", text="HESAPLA")
            self.update_idletasks()


if __name__ == "__main__":
    app = RoutingApp()
    app.mainloop()
