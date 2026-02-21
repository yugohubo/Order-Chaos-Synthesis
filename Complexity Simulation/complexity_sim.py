import numpy as np
import matplotlib.pyplot as plt

# --- 1. SİSTEM PARAMETRELERİ ---
N_PARTICLES = 3      # Toplam parçacık (ajan) sayısı
GRID_SIZE = 50        # Simülasyon alanının büyüklüğü (örn: 50x50)
N_STEPS = 100         # Simülasyonun kaç adım (zaman) ilerleyeceği

# --- 2. SİSTEMİN BAŞLATILMASI ---
def initialize_particles(n, grid_size):
    """N parçacığı grid üzerinde rastgele ama üst üste gelmeyecek şekilde başlatır."""
    positions = set()
    while len(positions) < n:
        pos = (np.random.randint(0, grid_size), 
               np.random.randint(0, grid_size))
        positions.add(pos)
    return np.array(list(positions))

# --- 3. FİZİK KURALLARI (ETKİLEŞİMİN MATEMATİĞİ) ---
def update_system(particles):
    """
    Sistemi bir zaman adımı (t -> t+1) ilerletir.
    Tanımladığımız tüm kuralları (A, B, C) uygular.
    """
    n_particles = len(particles)
    # Her parçacığın net hareket vektörünü (süperpozisyon) saklamak için
    net_movements = np.zeros_like(particles)

    # Her parçacık çiftini (i, j) hesapla
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            P_i = particles[i]
            P_j = particles[j]

            diff = P_j - P_i  # (dx, dy)
            abs_diff = np.abs(diff)
            dist_inf = np.max(abs_diff)  # L-sonsuz (Chebyshev) mesafesi

            move_i = np.zeros(2, dtype=int)
            move_j = np.zeros(2, dtype=int)

            # KURAL A: İTME (Repulsion)
            # "Temas" (d=1) varsa iter
            if dist_inf == 1:
                # sign(diff) bize (1,0), (0,1), (1,1) gibi yön vektörünü verir
                move_i = -np.sign(diff)
                move_j = np.sign(diff)

            # KURAL B+C: ÇEKME & EKSEN BASKINLIĞI
            # "Uzak" (d>=2) ise çeker
            elif dist_inf >= 2:
                d_x, d_y = abs_diff
                sign_diff = np.sign(diff) # (sign(dx), sign(dy))

                # Eksen baskınlığı: X ekseni daha yakınsa (veya Y=0 ise) X'te hareket et
                if d_x > 0 and (d_x < d_y or d_y == 0):
                    move_i[0] = sign_diff[0]  # Pi, Pj'ye X'te yaklaşır
                    move_j[0] = -sign_diff[0] # Pj, Pi'ye X'te yaklaşır
                
                # Eksen baskınlığı: Y ekseni daha yakınsa (veya X=0 ise) Y'de hareket et
                elif d_y > 0 and (d_y < d_x or d_x == 0):
                    move_i[1] = sign_diff[1]
                    move_j[1] = -sign_diff[1]
                
                # Kilitlenme: Eksenler eşitse (d_x == d_y) HAREKET YOK (Konf. 3)
                # Bu 'elif' bloğu kasıtlı olarak boştur.
                elif d_x == d_y:
                    pass 

            # Hesaplanan ikili hareketleri net harekete ekle (Süperpozisyon)
            net_movements[i] += move_i
            net_movements[j] += move_j

    # Tüm parçacıkların pozisyonlarını net hareketlerine göre güncelle
    # Kırpma (clip) ile parçacıkların grid dışına çıkmasını engelle
    new_particles = particles + net_movements
    new_particles = np.clip(new_particles, 0, GRID_SIZE - 1)

    return new_particles

# --- 4. SİMÜLASYONU ÇALIŞTIRMA ---
print(f"{N_PARTICLES} parçacıkla {N_STEPS} adımlık simülasyon başlıyor...")

# Başlangıç pozisyonlarını oluştur
initial_positions = initialize_particles(N_PARTICLES, GRID_SIZE)

# Simülasyon geçmişini kaydet (görselleştirme için)
history = [initial_positions]
current_positions = initial_positions

for step in range(N_STEPS):
    current_positions = update_system(current_positions)
    history.append(current_positions)
    if (step + 1) % 10 == 0:
        print(f"Adım {step + 1}/{N_STEPS} tamamlandı.")

print("Simülasyon tamamlandı. Görselleştiriliyor...")

# --- 5. GÖRSELLEŞTİRME ---
history_array = np.array(history)  # (N_STEPS+1, N_PARTICLES, 2)

plt.figure(figsize=(12, 12))
plt.title(f"N={N_PARTICLES} Parçacıklı Kompleks Sistem Simülasyonu ({N_STEPS} Adım)")
plt.xlabel("X Ekseni")
plt.ylabel("Y Ekseni")
plt.grid(True, linestyle=':', alpha=0.5)
plt.axis('equal') # Eksenleri eşit ölçekle

# Her parçacığın tüm yolunu çiz
for i in range(N_PARTICLES):
    # path shape = (N_STEPS+1, 2)
    path = history_array[:, i, :]
    plt.plot(path[:, 0], path[:, 1], 'o-', 
             markersize=2, alpha=0.3, linewidth=1)

# Başlangıç noktalarını (yeşil) ve Bitiş noktalarını (kırmızı) vurgula
start_points = history_array[0]
end_points = history_array[-1]

plt.scatter(start_points[:, 0], start_points[:, 1], 
            color='green', s=100, zorder=5, label=f'Başlangıç (t=0)')
plt.scatter(end_points[:, 0], end_points[:, 1], 
            color='red', s=100, zorder=5, label=f'Bitiş (t={N_STEPS})')

plt.legend()
plt.show()