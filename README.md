# Asynchronous Multi-Sensor Data Fusion & Extended Kalman Filter (EKF) Engine for UAV Target Tracking

Bu proje; İHA (UAV) ortamında hava hedeflerini yüksek doğrulukla takip etmek, durumsal farkındalık sağlamak ve sensör gecikmelerinden/desenkronizasyonundan bağımsız çalışabilmek amacıyla **saf NumPy ve SciPy** kullanılarak sıfırdan geliştirilmiş bir **Asenkron Genişletilmiş Kalman Filtresi (EKF)** motorudur.

Hiçbir hazır filtreleme kütüphanesi (*FilterPy, PyKalman vb.*) kullanılmamış; tüm durum geçişleri, sürekli-zaman süreç gürültüsü kovaryansları, doğrusal olmayan ölçüm fonksiyonları ve analitik Jacobi matrisleri matematiksel temelleriyle türetilmiştir.

---

## 1. Mimari ve Temel Bileşenler

```
uav-sensor-fusion/
├── src/
│   ├── __init__.py           # Paket dışa aktarımları
│   ├── dynamics.py          # 3B Sabit Hızlı (CV) Durum Vektörü ve Yer Gerçeği Üreteci
│   ├── sensors.py           # Radar (10 Hz) ve Kamera/EO (30 Hz) Sensör Modelleri
│   ├── simulator.py         # heapq Tabanlı Asenkron Öncelik Olay Kuyruğu
│   ├── ekf.py               # Asenkron EKF Motoru (Analitik Jacobianlar & Joseph Formu)
│   └── main.py              # 20s Uçuş Senaryosu, Benchmark, RMSE ve Matplotlib Görselleştirme
├── tests/
│   ├── __init__.py
│   └── test_ekf.py          # Analitik Jacobian vs Sayısal Türev ve Filtre Doğrulama Testleri
├── results/
│   └── ekf_fusion_results.png # 300 DPI Yüksek Çözünürlüklü Sonuç Grafiği
├── requirements.txt         # Gereksinimler
└── README.md
```

---

## 2. Matematiksel Temeller

### A. Durum Modeli (3D Constant Velocity - NED Çerçevesi)
Durum vektörü Kuzey-Doğu-Aşağı (NED) koordinat çerçevesinde 6 boyutludur:
$$\mathbf{x} = \begin{bmatrix} x & v_x & y & v_y & z & v_z \end{bmatrix}^T$$

Zaman adımı $\Delta t = t_k - t_{k-1}$ dinamik olarak iki sensör olayı arasındaki farktan hesaplanır.
Durum geçiş matrisi:
$$\mathbf{F}(\Delta t) = \begin{bmatrix}
1 & \Delta t & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & \Delta t & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & \Delta t \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}$$

Süreç gürültüsü, sürekli-zaman beyaz gürültü ivme (Continuous White Noise Acceleration - CWNA) modeli entegrasyonuyla elde edilir:
$$\mathbf{Q}_{1D}(\Delta t) = q_c \begin{bmatrix} \frac{\Delta t^3}{3} & \frac{\Delta t^2}{2} \\ \frac{\Delta t^2}{2} & \Delta t \end{bmatrix}, \quad \mathbf{Q}(\Delta t) = \text{blkdiag}(\mathbf{Q}_{1D}, \mathbf{Q}_{1D}, \mathbf{Q}_{1D})$$

### B. Radar Ölçüm Modeli (10 Hz)
Hedefin İHA'ya göre bağıl koordinatları: $\Delta x = x - x_{uav}, \ \Delta y = y - y_{uav}, \ \Delta z = z - z_{uav}$.
3B Öklid menzili:
$$R = \sqrt{\Delta x^2 + \Delta y^2 + \Delta z^2}$$
$$z_{\text{radar}} = R + \mathcal{N}(0, \sigma_r^2)$$

Analitik Jacobi matrisi ($1 \times 6$):
$$\mathbf{H}_{\text{radar}} = \begin{bmatrix} \frac{\Delta x}{R} & 0 & \frac{\Delta y}{R} & 0 & \frac{\Delta z}{R} & 0 \end{bmatrix}$$

### C. Kamera / Elektro-Optik (EO) Ölçüm Modeli (30 Hz)
Yatay menzil $\rho = \sqrt{\Delta x^2 + \Delta y^2}$ olmak üzere:
$$\psi = \text{arctan2}(\Delta y, \Delta x) \quad (\text{Azimut})$$
$$\lambda = \arcsin\left(\frac{\Delta z}{R}\right) \quad (\text{Elevasyon})$$
$$\mathbf{z}_{\text{cam}} = \begin{bmatrix} \psi + \mathcal{N}(0, \sigma_\psi^2) \\ \lambda + \mathcal{N}(0, \sigma_\lambda^2) \end{bmatrix}$$

Analitik Jacobi matrisi ($2 \times 6$):
$$\mathbf{H}_{\text{cam}} = \begin{bmatrix}
-\frac{\Delta y}{\rho^2} & 0 & \frac{\Delta x}{\rho^2} & 0 & 0 & 0 \\
-\frac{\Delta x \cdot \Delta z}{\rho R^2} & 0 & -\frac{\Delta y \cdot \Delta z}{\rho R^2} & 0 & \frac{\rho}{R^2} & 0
\end{bmatrix}$$

### D. Asenkron Kalman Güncellemesi ve Joseph Formu
- İnovasyon hesabı: $\mathbf{y} = \mathbf{z} - h(\hat{\mathbf{x}})$. Azimut açısı için $[-\pi, \pi]$ normalizasyonu uygulanır.
- Kalman Kazancı: $\mathbf{K} = \mathbf{P} \mathbf{H}^T (\mathbf{H} \mathbf{P} \mathbf{H}^T + \mathbf{R})^{-1}$
- Kovaryans Güncellemesi: Sayısal kararlılık ve pozitif yarı-tanımlılığı (PSD) garanti etmek için **Joseph Stabilize Formu** kullanılır:
$$\mathbf{P} = (\mathbf{I} - \mathbf{K}\mathbf{H}) \mathbf{P} (\mathbf{I} - \mathbf{K}\mathbf{H})^T + \mathbf{K} \mathbf{R} \mathbf{K}^T$$

---

## 3. Çalıştırma Talimatları

### Testleri Çalıştırma:
Tüm analitik türevler, finite-difference sayısal türevleriyle ve filtre yakınsama testleriyle doğrulanır:
```bash
python tests/test_ekf.py
```

### Simülasyonu ve Benchmark'ı Koşturma:
```bash
python src/main.py
```

---

## 4. Benchmark ve Başarım Sonuçları (20 saniyelik Uçuş Senaryosu)

Simülasyonda hedef $x_0=1200\text{ m}, y_0=-800\text{ m}, z_0=-500\text{ m}$ konumundan $v=[35, 25, -4]\text{ m/s}$ hızla hareket ederken Radar (10 Hz, $\sigma_r=6\text{ m}$) ve Kamera (30 Hz, $\sigma=0.004\text{ rad}$) asenkron zaman damgalarıyla kuyruktan çekilmiştir.

| Metrik | Ham Sensör (Unfused) | EKF Filtresi (Genel) | EKF Kalıcı Durum ($t \ge 3\text{s}$) |
|---|---|---|---|
| **Toplam 3B Konum RMSE** | **11.256 m** | **3.285 m** | **2.349 m (%79.1 İyileşme)** |
| **X Konumu RMSE (Kuzey)** | 5.820 m | 1.354 m | 1.222 m |
| **Y Konumu RMSE (Doğu)** | 6.840 m | 2.061 m | 1.559 m |
| **Z Konumu RMSE (Aşağı)** | 6.780 m | 2.171 m | 1.263 m |
| **Toplam 3B Hız RMSE** | *(Mevcut değil)* | **3.156 m/s** | **1.018 m/s** |

Üretilen grafik `results/ekf_fusion_results.png` dosyasında:
1. **Konum Takibi:** X, Y, Z eksenlerinde Yer Gerçeği, Ham Ölçüm ve EKF Kestirimi.
2. **Hız Yakınsaması:** Doppler/hız ölçümü olmaksızın pozisyon türevlerinden elde edilen hız vektörü kestirimi.
3. **Tutarlılık & $3\sigma$ Sınırları:** EKF konum hataları ve $\pm 3\sigma$ kovaryans sınırları.
