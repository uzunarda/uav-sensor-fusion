# Asynchronous Multi-Sensor Data Fusion & Extended Kalman Filter (EKF) Engine for UAV Target Tracking

Bu proje; İHA (UAV) ortamında hava hedeflerini yüksek doğrulukla takip etmek, durumsal farkındalık sağlamak ve sensör gecikmelerinden/zaman kaymalarından bağımsız çalışabilmek amacıyla **saf NumPy ve SciPy** kullanılarak sıfırdan geliştirilmiş bir **Asenkron Genişletilmiş Kalman Filtresi (EKF)** motoru ve **Canlı Taktik Komuta ve Radar GUI Konsolu**dur.

Hiçbir hazır filtreleme kütüphanesi (*FilterPy, PyKalman vb.*) kullanılmamış; tüm durum geçişleri, sürekli-zaman süreç gürültüsü kovaryans integralleri, doğrusal olmayan ölçüm fonksiyonları, analitik Jacobi matrisleri, Joseph-form kovaryans stabilizasyonu ve **Sıra Dışı Ölçüm (Out-Of-Sequence Measurement - OOSM) Rollback / Buffer-Replay mimarisi** matematiksel temelleriyle sıfırdan türetilmiştir.

---

## 1. Mimari ve Dizin Yapısı

```
uav-sensor-fusion/
├── AGENTS.md                  # Sistem mimarisi, matematiksel modeller ve AI ajan kuralları
├── SKILLS.md                  # Adım adım operasyonel geliştirme, test ve sensör ekleme yetenekleri
├── README.md                  # Proje ana dökümantasyonu ve performans kıyaslama tablosu
├── requirements.txt           # Bağımlılıklar (NumPy, SciPy, Matplotlib, PyQt6, PyQtGraph, PyOpenGL)
├── start.bat                  # Tek tıkla doğrudan Canlı Taktik Konsolu başlatan Windows betiği
├── start.ps1                  # Güvenli kodlamalı (BOM/ASCII uyumlu) PowerShell başlatıcı
├── .agents/skills/            # Antigravity ve modern ajanlar için otomatik yetenek paketi
│   └── uav-sensor-fusion/
│       └── SKILL.md
├── src/
│   ├── __init__.py            # Paket dışa aktarımları
│   ├── dynamics.py           # 3B Sabit Hızlı (CV) Kinematik Model ve Yer Gerçeği Üreteci
│   ├── sensors.py            # Radar (10 Hz) ve Kamera/EO (30 Hz) Sensör Modelleri (Gecikme & Bias)
│   ├── simulator.py          # heapq Tabanlı Asenkron Öncelik Olay Kuyruğu (t_receive sıralı)
│   ├── ekf.py                # Asenkron EKF Motoru (OOSM Rollback/Replay & Joseph Formu)
│   ├── main.py               # Aşama 2 Kıyaslama Benchmark'ı (Mod A, Mod B, Mod C Karşılaştırması)
│   └── gui/                  # Faz 2.5: Canlı Taktik Komuta ve Çok Sensörlü Füzyon Konsolu
│       ├── __init__.py
│       ├── app.py            # Ana QMainWindow Taktik Kokpit ve 4 Sekmeli Komuta Merkezi
│       ├── worker.py         # 60 FPS Canlı Oynatıcı, Hız Çarpanları ve Anlık Mod Değiştirici
│       ├── views/            # 4 Ana Taktik Sekme Görünümü
│       │   ├── live_radar_view.py   # Sekme 1: Canlı Radar, 3B Uzay Görünümü ve Telemetri
│       │   ├── benchmark_view.py    # Sekme 2: Mod A/B/C Karşılaştırmalı Metrik ve Grafik Paneli
│       │   ├── test_runner_view.py  # Sekme 3: Dahili Birim/GUI Test Koşturucu ve Canlı Log Konsolu
│       │   └── config_view.py       # Sekme 4: Sensör Gecikmeleri ve Filtre Ayarları Konfigüratörü
│       └── widgets/          # Özel Taktik Arayüz Bileşenleri
│           ├── radar_display.py     # 2B PPI Scope Radar (Range Rings, Dinamik FOV, 2σ Elipsi)
│           ├── gl_display.py        # 3B OpenGL Koordinat Eksenleri, İrtifa İzdüşümü ve LOS Vektörü
│           ├── telemetry_panel.py   # HUD Dijital Sayaçlar (Hız, İrtifa, Hata) ve Flaş OOSM Rozeti
│           └── control_bar.py       # Oynat/Durdur, Hız (0.5x-Azami), Mod Seçici ve Zaman Çubuğu
├── tests/
│   ├── __init__.py
│   ├── test_ekf.py           # Analitik Jacobian (Sonlu Farklar), OOSM Rollback ve Filtre Testleri
│   └── test_gui.py           # GUI Başlatma, Frame Dispatch ve Offscreen Arayüz Testleri
└── results/
    ├── ekf_oosm_comparison.png  # 4 Panelli A/B/C Kıyaslama ve OOSM Dağılım Grafiği
    └── ekf_fusion_results.png   # Yüksek Çözünürlüklü 300 DPI Füzyon Yörünge Grafiği
```

---

## 2. Matematiksel Temeller ve Algoritmik Mimarisi

### A. Koordinat Sistemi ve Durum Vektörü
Tüm dinamikler ve filtre tahminleri **3B North-East-Down (NED)** referans koordinat çerçevesinde gerçekleştirilir:

$$\mathbf{x} = \begin{bmatrix} x & \dot{x} & y & \dot{y} & z & \dot{z} \end{bmatrix}^T \in \mathbb{R}^{6 \times 1}$$

- $x, \dot{x}$: Kuzey (North) ekseninde konum ($m$) ve hız ($m/s$)
- $y, \dot{y}$: Doğu (East) ekseninde konum ($m$) ve hız ($m/s$)
- $z, \dot{z}$: Aşağı (Down) ekseninde konum ($m$) ve düşey hız ($m/s$, NED kuralı: aşağıya doğru pozitif)
- *Görselleştirme İrtifası (Altitude):* $-z$ değeridir.

### B. Sürekli Beyaz Gürültülü İvme Modeli (CWNA Dynamics)
Durum geçiş matrisi $\mathbf{F}(\Delta t)$ ve ayrık süreç gürültüsü kovaryansı $\mathbf{Q}(\Delta t)$, $q_c$ sürekli ivme spektral yoğunluğunun integraliyle 3 eksen için blok-diyagonal olarak kurulur:

$$\mathbf{F}_{1D}(\Delta t) = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix}, \quad \mathbf{Q}_{1D}(\Delta t) = q_c \begin{bmatrix} \frac{\Delta t^3}{3} & \frac{\Delta t^2}{2} \\ \frac{\Delta t^2}{2} & \Delta t \end{bmatrix}$$

$$\mathbf{F}(\Delta t) = \text{blkdiag}(\mathbf{F}_{1D}, \mathbf{F}_{1D}, \mathbf{F}_{1D}), \quad \mathbf{Q}(\Delta t) = \text{blkdiag}(\mathbf{Q}_{1D}, \mathbf{Q}_{1D}, \mathbf{Q}_{1D})$$

### C. Doğrusal Olmayan Ölçüm Modelleri ve Analitik Jacobiler

Sensörler İHA'ya göre bağıl hedef konumunu ($\Delta x = x - x_{uav}$, $\Delta y = y - y_{uav}$, $\Delta z = z - z_{uav}$) gözlemler:

#### 1. Radar Sensörü (10 Hz, Menzil / Range)

Mesafe ölçüm fonksiyonu ($r$ Öklid menzili):

$$h_{radar}(\mathbf{x}) = r = \sqrt{\Delta x^2 + \Delta y^2 + \Delta z^2}$$

Analitik Jacobi matrisi ($\mathbf{H}_{radar} \in \mathbb{R}^{1 \times 6}$):

$$\mathbf{H}_{radar} = \begin{bmatrix} \frac{\Delta x}{r} & 0 & \frac{\Delta y}{r} & 0 & \frac{\Delta z}{r} & 0 \end{bmatrix}$$

#### 2. Kamera / EO Sensörü (30 Hz, Azimut & Elevasyon)

Yatay menzil $\rho = \sqrt{\Delta x^2 + \Delta y^2}$ olmak üzere açısal ölçüm fonksiyonu:

$$h_{cam}(\mathbf{x}) = \begin{bmatrix} \psi \\ \lambda \end{bmatrix} = \begin{bmatrix} \text{atan2}(\Delta y, \Delta x) \\ \text{atan2}(-\Delta z, \rho) \end{bmatrix}$$

Analitik Jacobi matrisi ($\mathbf{H}_{cam} \in \mathbb{R}^{2 \times 6}$):

$$\mathbf{H}_{cam} = \begin{bmatrix} -\frac{\Delta y}{\rho^2} & 0 & \frac{\Delta x}{\rho^2} & 0 & 0 & 0 \\ \frac{\Delta x \Delta z}{r^2 \rho} & 0 & \frac{\Delta y \Delta z}{r^2 \rho} & 0 & -\frac{\rho}{r^2} & 0 \end{bmatrix}$$

*Açı Sarmalama (Angle Unwrapping):* Azimut kalıntısı $\tilde{y}_\psi = z_\psi - \hat{\psi}$ açısı $-\pi$ ile $+\pi$ aralığına sarılır:

$$\tilde{y}_\psi = (\tilde{y}_\psi + \pi) \pmod{2\pi} - \pi$$

### D. Joseph-Formu Kovaryans Güncellemesi

Filtrenin sayısal yuvarlama hatalarından dolayı kovaryans matrisinin simetrisini veya pozitif yarı-tanımlılığını kaybetmesini önlemek için standart güncelleme yerine **Joseph Formu** uygulanır:

$$\mathbf{P}_k = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1} (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k)^T + \mathbf{K}_k \mathbf{R}_k \mathbf{K}_k^T$$

Ayrıca her adımdan sonra simetri zorlanır: $\mathbf{P} = \frac{1}{2}(\mathbf{P} + \mathbf{P}^T)$.

### E. Asenkron Öncelik Kuyruğu ve OOSM Rollback/Replay Motoru
- **Asenkron Kuyruk (`simulator.py`):** Sensörler bağımsız frekanslara (10 Hz vs 30 Hz), iletim gecikmelerine ($\tau$) ve donanım jitter'ına sahiptir. Olaylar işlemciye varış anına ($t_{receive} = t_{capture} + \tau + \epsilon$) göre `heapq` ile sıralanır.
- **Sıra Dışı Ölçüm (OOSM):** Kamera işlem süresi radardan uzun olduğunda $t_{capture} < t_{filter}$ durumu oluşur.
- **Rollback / Replay:** Filtre geçmişe ait bir `FilterSnapshot` listesi tutar. Sıra dışı ölçüm algılandığında en yakın geçmiş duruma geri sarılır (rollback), gecikmiş ölçüm araya eklenir ve güncel zamana kadar adımlar yeniden koşturulur (replay).

---

## 3. Canlı Taktik Komuta ve Radar Konsolu (GUI - 4 Ana Sekme)

Modern askeri durumsal farkındalık konsolu görünümünde, **PyQt6** ve **PyQtGraph** kullanılarak 60 FPS pürüzsüz donanım hızlandırmalı arayüz sunar:

### 🎯 Sekme 1: Canlı Taktik Radar & 3B Uzay Görünümü (`live_radar_view.py`)
- **2B PPI Scope Radar (Sol Geniş Panel):**
  - Merkezde İHA platform ikonu.
  - $500\text{ m}, 1000\text{ m}, 1500\text{ m}$ menzil halkaları ve azimut ızgarası.
  - Hedefe kilitlenen dinamik kamera görüş açısı (FOV konisi).
  - Katmanlar: 🔴 Ham gürültülü ölçümler, 🔵 Yer Gerçeği (Ground Truth), 🟢 EKF Füzyon kestirimi, 🟡 $2\sigma$ Belirsizlik Kovaryans Elipsi.
- **3B OpenGL Uzay Görünümü (Sağ Üst Panel):**
  - **3B Koordinat Eksenleri:** 🔴 +Y Kuzey (1800m), 🔵 +X Doğu (1800m), 🟢 +Z İrtifa (900m) eksen çizgileri ve mesafe çentikleri.
  - **Hedef İrtifa İzdüşüm Çizgisi (Altitude Drop Line):** Hedefin 3B noktasından zemin ızgarasına ($Z=0$) inen dikey kılavuz çizgisi ve zemin izdüşüm ayak izi.
  - İHA ile hedef arasındaki dinamik 3B yaklaşma vektörü (LOS Vector).
  - Kompakt, 26px yüksekliğinde tek satırlı şık HUD lejantı.
- **HUD Dijital Telemetri Paneli (Sağ Alt Panel):**
  - Ferah 9 dijital sayaç: Toplam Hız $\|\hat{\mathbf{V}}\|$, Eksenel Hızlar ($V_x, V_y, V_z$), Hedef İrtifası, Bağıl Menzil, Anlık 3B Konum/Hız Hatası ve Son Sensör Bilgisi.
  - **OOSM Flaş Rozeti:** Sıra dışı ölçüm işlendiğinde 150 ms yanan parlak amber rozet.
- **Oynatma ve Mod Kontrol Çubuğu (En Alt):**
  - Oynat, Duraklat, Sıfırla butonları; `0.5x`, `1.0x`, `2.0x`, `Azami` hız çarpanları.
  - Kesintisiz mod değiştirici: `Mod A (İdeal)`, `Mod B (Naive)`, `Mod C (Rollback/OOSM Destekli)`.
  - Zaman kaydırıcı (Timeline Scrubbing) ile 0.0s - 20.0s arasında anında zıplama.

### 📊 Sekme 2: Aşama 2 Analitik Kıyaslama Benchmark'ı (`benchmark_view.py`)
- Tek tıkla arka plan iş parçacığında (`QThread`) Mod A, Mod B ve Mod C simülasyonunu koşturma.
- Kapsamlı RMSE metrik kıyaslama tablosu (Kuzey, Doğu, İrtifa, Toplam Hız ve OOSM sayıları).
- Anlık etkileşimli hata ve OOSM gecikme dağılım grafikleri.

### 🧪 Sekme 3: Sistem Tanılama ve Test Koşturucu (`test_runner_view.py`)
- GUI içinden tek tıkla birim ve analitik Jacobian testlerini koşturma.
- Yeşil/Kırmızı durum rozetleri, çalışma süreleri ve renkli terminal log akışı.

### ⚙️ Sekme 4: Sensör Gecikmeleri ve Filtre Konfigüratörü (`config_view.py`)
- Radar ve Kamera frekansları, ortalama gecikme süreleri ($\tau$), jitter ve saat kaymalarını anlık özelleştirme.
- EKF süreç gürültüsü yoğunluğu ($q_c$) kalibrasyonu ve tek tıkla simülasyonu yeniden başlatma.

---

## 4. Aşama 2: Performans Kıyaslama Benchmark Sonuçları

Aşağıdaki metrikler 20 saniyelik manevralı hedef uçuş senaryosunda $10\text{ Hz}$ Radar ve $30\text{ Hz}$ Kamera verileriyle elde edilmiştir:

| Metrik / Parametre | Mod A (İdeal) | Mod B (Naive / Gecikmeli) | Mod C (Rollback / OOSM Telafili) |
|---|---|---|---|
| **Kalıcı Durum 3B Konum RMSE** | **1.954 m** | **3.464 m** | **2.681 m (%22.6 İyileşme)** |
| - Eksen X (Kuzey) RMSE | 1.304 m | 1.870 m | 1.474 m |
| - Eksen Y (Doğu) RMSE | 0.971 m | 2.381 m | 1.498 m |
| - Eksen Z (İrtifa) RMSE | 1.085 m | 1.684 m | 1.664 m |
| **Toplam 3B Konum RMSE** | 3.119 m | 4.241 m | 3.572 m |
| **Kalıcı Durum 3B Hız RMSE** | 0.918 m/s | 1.111 m/s | 1.108 m/s |
| **Toplam 3B Hız RMSE** | 3.859 m/s | 3.668 m/s | 3.701 m/s |
| **İşlenen OOSM Sayısı** | 0 (Gecikmesiz) | 0 (Yönetilemez) | **127 Olay** |
| **Atılan (Dropped) OOSM** | 0 | 0 | **0 Olay (%100 Başarı)** |

---

## 5. Kurulum ve Çalıştırma Talimatları

### Bağımlılıkların Kurulumu:
```bash
pip install -r requirements.txt
```

### Tek Tıkla Doğrudan Başlatıcı:
- **Windows Gezgini'nden Çift Tıklayarak:**
  ```cmd
  start.bat
  ```
  *(Hiçbir seçim ekranı beklemeden bağımlılıkları denetler ve doğrudan PyQt6 Taktik Konsolunu açar).*
- **PowerShell ile Doğrudan Başlatma:**
  ```powershell
  .\start.ps1
  ```
  *(İsteğe bağlı terminal parametreleri: `.\start.ps1 -Mode gui`, `.\start.ps1 -Mode bench`, `.\start.ps1 -Mode test`)*

### Doğrudan Python ile Çalıştırma:
- **Canlı Taktik Komuta ve Radar Konsolunu Başlatma:**
  ```bash
  python -m src.gui.app
  ```
- **Aşama 2 Kıyaslama Benchmark'ını ve Grafikleri Üretme:**
  ```bash
  python src/main.py
  ```
- **Tüm Birim ve GUI Testlerini Koşturma:**
  ```bash
  python -m unittest discover tests
  ```

---

## 6. AI Ajanları ve İşbirlikçi Geliştirme Standartları

Bu proje, insan mühendislerin yanı sıra yapay zeka ajanlarıyla (Antigravity, Cursor, Windsurf, Claude Code, GitHub Copilot) kusursuz işbirliği yapabilecek şekilde yapılandırılmıştır:

- **[AGENTS.md](file:///c:/Users/Arda/Desktop/uav-sensor-fusion/AGENTS.md):** Sistemin matematiksel değişmezleri (NED çerçevesi, float64 hassasiyeti, Joseph formu, dış kütüphane yasağı) ve ajan kuralları.
- **[SKILLS.md](file:///c:/Users/Arda/Desktop/uav-sensor-fusion/SKILLS.md):** Test çalıştırma, benchmark koşturma, yeni sensör ekleme ve filtre kalibrasyonu operasyonel runbook'ları.
- **[.agents/skills/uav-sensor-fusion/SKILL.md](file:///c:/Users/Arda/Desktop/uav-sensor-fusion/.agents/skills/uav-sensor-fusion/SKILL.md):** Antigravity ajan motoru için yerel keşif ve otomatik yetenek tanımlaması.
