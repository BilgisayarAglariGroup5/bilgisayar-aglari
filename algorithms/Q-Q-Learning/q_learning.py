    # Yiğit Alakuş
import numpy as np
import random
import networkx as nx
# Melek'in yazdığı metric.py dosyasını buraya bağlıyoruz
from metric import MetricsEngine, Weights

class QLearningAgent:
    def __init__(self, graph, source, target, episodes=500, alpha=0.1, gamma=0.9, 
                 w_delay=0.33, w_rel=0.33, w_res=0.34, min_bandwidth=0):
        """
        Q-Learning Ajanı - Metric.py Entegreli Sürüm
        """
        self.G = graph
        self.source = source
        self.target = target
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.min_bandwidth = min_bandwidth  # <-- Hata veren kısım düzeltildi
        
        # Metric.py motorunu başlatıyoruz (Bağlantı burada kuruluyor)
        self.engine = MetricsEngine(self.G)
        self.weights = Weights(w_delay, w_rel, w_res)
        
        # Keşif Ayarları
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.q_table = {}
        self.episode_rewards = [] 

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def get_valid_neighbors(self, state):
        """Bant genişliği (Hız) kısıtlamasına göre komşuları filtreler"""
        all_neighbors = list(self.G.neighbors(state))
        valid_neighbors = []
        
        for n in all_neighbors:
            edge_data = self.G[state][n]
            # Veri setinde isim farklılıkları olabilir, hepsini kontrol et
            bw = edge_data.get('bandwidth', edge_data.get('capacity_mbps', edge_data.get('bant_genisligi', 0)))
            
            if bw >= self.min_bandwidth:
                valid_neighbors.append(n)
                
        return valid_neighbors

    def choose_action(self, state):
        neighbors = self.get_valid_neighbors(state)
        
        if not neighbors:
            return None # Gidecek yol yok (Tıkandı)

        # Epsilon-Greedy Stratejisi
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(neighbors)
        
        q_values = [self.get_q(state, n) for n in neighbors]
        max_q = max(q_values)
        
        # En iyileri seç (eşitlik varsa rastgele)
        best_actions = [n for n, q in zip(neighbors, q_values) if q == max_q]
        if not best_actions: return random.choice(neighbors)
        return random.choice(best_actions)

    def train(self):
        self.episode_rewards = [] # Grafik verisini sıfırla
        
        for ep in range(self.episodes):
            state = self.source
            path = [state]
            total_reward_in_this_episode = 0
            
            while state != self.target:
                action = self.choose_action(state)
                if action is None:
                    break 
                
                next_state = action
                
                # Döngü Engelleme (Kendi kuyruğunu ısırmasın)
                if next_state in path:
                    reward = -1000
                    self.update_q(state, action, reward, next_state)
                    total_reward_in_this_episode += reward
                    break 
                
                path.append(next_state)
                
                # --- ÖDÜL MEKANİZMASI (METRIC.PY KULLANILIYOR) ---
                if next_state == self.target:
                    # Hedefe ulaştı! Tüm yolun maliyetini metric.py ile hesapla
                    # Bu sayede Melek'in yazdığı formüller ödülü belirler.
                    try:
                        metrics = self.engine.compute(path, demand_mbps=self.min_bandwidth)
                        total_cost = self.engine.weighted_sum(metrics, self.weights)
                        # Maliyet ne kadar azsa, ödül o kadar büyük olsun
                        reward = 10000.0 / (total_cost + 1e-9)
                    except:
                        # Eğer metric.py hesaplarken hata verirse (veri eksikliği vb.)
                        reward = 100.0
                else:
                    # Hedefe daha varmadıysa küçük bir adım cezası ver (yolu uzatmasın)
                    reward = -1 

                self.update_q(state, action, reward, next_state)
                total_reward_in_this_episode += reward
                state = next_state
                
                if len(path) > 250: break # Sonsuz döngü koruması
            
            self.episode_rewards.append(total_reward_in_this_episode)

            # Epsilon azaltma (Zamanla daha az rastgele, daha çok akıllı davran)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def update_q(self, state, action, reward, next_state):
        old_q = self.get_q(state, action)
        
        next_neighbors = self.get_valid_neighbors(next_state)
        if next_neighbors:
            max_future_q = max([self.get_q(next_state, n) for n in next_neighbors])
        else:
            max_future_q = 0
        
        # Q-Learning Formülü
        new_q = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)
        self.q_table[(state, action)] = new_q

    def get_best_path(self):
        """Eğitimden sonra öğrenilen en iyi yolu çıkarır"""
        path = [self.source]
        state = self.source
        visited = {state}
        
        while state != self.target:
            neighbors = self.get_valid_neighbors(state)
            valid_neighbors = [n for n in neighbors if n not in visited]
            
            if not valid_neighbors: return None
            
            # Q tablosuna bakarak en yüksek puanlı komşuyu seç
            best_next = max(valid_neighbors, key=lambda n: self.get_q(state, n))
            
            path.append(best_next)
            visited.add(best_next)
            state = best_next
            
            if len(path) > len(self.G.nodes): return None
            
        return path
