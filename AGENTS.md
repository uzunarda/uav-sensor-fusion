# AGENTS.md - Multi-Agent Collaborative Development Guidelines & System Architecture

Bu doküman, **İHA Hava Sahası Farkındalık ve Asenkron Çok Sensörlü Genişletilmiş Kalman Filtresi (EKF)** projesinde çalışacak tüm AI ajanları (Antigravity, Cursor, Windsurf, Claude Code, GitHub Copilot vb.) ve geliştiriciler için projenin kurallarını, mimari kararlarını, matematiksel temellerini ve kodlama standartlarını tanımlar.

---

## 1. Proje Özeti ve Temel Misyon

Bu proje; İHA (UAV) ortamında hava hedeflerini yüksek hassasiyetle izlemek, durumsal farkındalık sağlamak ve sensör gecikmelerinden/zaman kaymalarından bağımsız çalışabilmek amacıyla **saf NumPy ve SciPy** kullanılarak sıfırdan geliştirilmiş bir **Asenkron Genişletilmiş Kalman Filtresi (EKF)** motoru ve **Canlı Taktik Radar GUI Konsolu**dur.

### ⚠️ KESİN KURAL: Dış Filtreleme Kütüphanesi Yasağı
Projeye **kesinlikle** `filterpy`, `pykalman`, `scikit-kinematics` gibi hazır Kalman/durum kestirim kütüphaneleri eklenemez.
- Tüm durum geçişleri, Jacobi türevleri, gürültü kovaryans integralleri, Joseph-form kovaryans güncellemeleri ve OOSM (Out-of-Sequence Measurement) tampon mekanizmaları sıfırdan el ile formüle edilmelidir.

---

## 2. Koordinat Sistemi ve Durum Vektörü Standartları

Tüm fiziksel dinamikler ve filtre hesaplamaları **3B North-East-Down (NED)** referans koordinat çerçevesinde gerçekleştirilir.

### Durum Vektörü ($\mathbf{x} \in \mathbb{R}^{6 \times 1}$):
$$\mathbf{x} = \begin{bmatrix} x & \dot{x} & y & \dot{y} & z & \dot{z} \end{bmatrix}^T$$

- $x$: Kuzey (North) ekseninde konum (metre)
- $\dot{x}$ (`vx`): Kuzey hızı ($m/s$)
- $y$: Doğu (East) ekseninde konum (metre)
- $\dot{y}$ (`vy`): Doğu hızı ($m/s$)
- $z$: Aşağı (Down) ekseninde konum (metre, NED kuralı: aşağıya doğru pozitif)
- $\dot{z}$ (`vz`): Aşağı hız ($m/s$)

> **Görselleştirme Notu:** Taktik Radar GUI ve 3B görünümde irtifa (Altitude), NED çerçevesindeki $-z$ değeridir.

---

## 3. Matematiksel Model ve Algoritmik Prensipler

### 3.1. Sürekli Beyaz Gürültülü İvme Modeli (CWNA Dynamics)
Durum geçiş matrisi $F(\Delta t)$ ve ayrık süreç gürültüsü kovaryansı $Q(\Delta t)$ 3 eksen için blok-diyagonal olarak kurulur:

$$F_{1D}(\Delta t) = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix}, \quad Q_{1D}(\Delta t) = q_c \begin{bmatrix} \frac{\Delta t^3}{3} & \frac{\Delta t^2}{2} \\ \frac{\Delta t^2}{2} & \Delta t \end{bmatrix}$$

$$F(\Delta t) = \text{blkdiag}(F_{1D}, F_{1D}, F_{1D}), \quad Q(\Delta t) = \text{blkdiag}(Q_{1D}, Q_{1D}, Q_{1D})$$

Burada $q_c$ hedef manevralarını temsil eden sürekli zaman ivme spektral yoğunluğudur ($m^2/s^3$).

### 3.2. Doğrusal Olmayan Ölçüm Modelleri ve Analitik Jacobiler

Sensörler İHA'ya bağıl hedef konumunu ($\Delta x = x_{tgt} - x_{uav}$, $\Delta y = y_{tgt} - y_{uav}$, $\Delta z = z_{tgt} - z_{uav}$) ölçer:

#### Radar Sensörü (10 Hz, Menzil / Range)

$$h_{radar}(\mathbf{x}) = r = \sqrt{\Delta x^2 + \Delta y^2 + \Delta z^2}$$

$$H_{radar} = \frac{\partial h}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\Delta x}{r} & 0 & \frac{\Delta y}{r} & 0 & \frac{\Delta z}{r} & 0 \end{bmatrix}$$

#### Kamera / EO Sensörü (30 Hz, Azimut & Elevasyon)

Yatay menzil $\rho = \sqrt{\Delta x^2 + \Delta y^2}$ ve 3B menzil $r = \sqrt{\rho^2 + \Delta z^2}$ olmak üzere:

$$h_{cam}(\mathbf{x}) = \begin{bmatrix} \psi \\ \lambda \end{bmatrix} = \begin{bmatrix} \text{atan2}(\Delta y, \Delta x) \\ \text{atan2}(-\Delta z, \rho) \end{bmatrix}$$

$$H_{cam} = \begin{bmatrix} -\frac{\Delta y}{\rho^2} & 0 & \frac{\Delta x}{\rho^2} & 0 & 0 & 0 \\ \frac{\Delta x \Delta z}{r^2 \rho} & 0 & \frac{\Delta y \Delta z}{r^2 \rho} & 0 & -\frac{\rho}{r^2} & 0 \end{bmatrix}$$

**Açı Sarmalama (Angle Unwrapping):** Azimut kalıntısı $\tilde{y}_\psi$ hesaplanırken açı $-\pi$ ile $+\pi$ arasına sarılmalıdır:

```python
y[0, 0] = (y[0, 0] + np.pi) % (2.0 * np.pi) - np.pi
```

### 3.3. Joseph-Formu Kovaryans Güncellemesi

Filtre matrisinin her zaman simetrik ve pozitif yarı-tanımlı (positive semi-definite) kalmasını garantilemek için standart güncelleme yerine **Joseph Formu** uygulanır:

$$P_k = (I - K_k H_k) P_{k|k-1} (I - K_k H_k)^T + K_k R_k K_k^T$$

Ayrıca her adımdan sonra simetri zorlanır: $P = \frac{1}{2}(P + P^T)$.

### 3.4. Asenkron Öncelik Kuyruğu ve OOSM Rollback/Replay Motoru
- **Asenkron Kuyruk (`simulator.py`):** Sensör ölçümleri bağımsız frekanslara, iletim gecikmelerine ($\tau$) ve Gaussian jitter'a sahiptir. Olaylar işlemciye varış anına ($t_{receive} = t_{capture} + \tau + \epsilon$) göre `heapq` ile sıralanır.
- **Sıra Dışı Ölçüm (OOSM):** Gecikmeli bir ölçüm geldiğinde $t_{capture} < t_{filter}$ durumu oluşur.
- **Rollback / Replay Mimarisi (`ekf.py`):**
  1. Filtre her tahmin ve güncellemede bir `FilterSnapshot(t, x, P, measurement, uav_pos)` kaydeder.
  2. Gecikmiş ölçüm tespit edildiğinde ($t_{capture} < t_{current}$), tamponda $t_{snap} \le t_{capture}$ olan en son güvenli duruma geri sarılır (rollback).
  3. Yeni ölçüm kronolojik sırasına göre araya eklenir.
  4. Geçmişten güncel zamana kadar tüm adımlar sırasıyla yeniden koşturulur (replay).

---

## 4. Dizin ve Dosya Yapısı

```
uav-sensor-fusion/
├── AGENTS.md                  # Bu rehber: Sistem mimarisi ve ajan kuralları
├── SKILLS.md                  # Ajan çalışma yetenekleri ve operasyonel senaryolar
├── README.md                  # Proje ana dökümantasyonu ve kıyaslama tablosu
├── requirements.txt           # Bağımlılıklar (NumPy, SciPy, PyQt6, PyQtGraph, PyOpenGL)
├── start.ps1 / start.bat      # Tek tıkla interaktif başlatıcı betikler
├── .agents/skills/            # Otomatik keşfedilen modüler ajan yetenekleri
│   └── uav-sensor-fusion/
│       └── SKILL.md
├── src/
│   ├── dynamics.py            # Hedef dinamikleri ve analitik yörünge üreteci
│   ├── sensors.py             # Radar & Kamera sensör modelleri (gecikme, jitter, bias)
│   ├── simulator.py           # Asenkron heapq tabanlı sensör olay simülatörü
│   ├── ekf.py                 # EKF motoru, analitik Jacobiler, Joseph formu, OOSM buffer
│   ├── main.py                # Aşama 2 Kıyaslama Benchmark'ı (Mod A, Mod B, Mod C)
│   └── gui/                   # Faz 2.5: Canlı Taktik Radar GUI Konsolu (PyQt6 + PyQtGraph)
│       ├── app.py             # QMainWindow Komuta Merkezi ana uygulaması
│       ├── worker.py          # 60 FPS Simülasyon iş parçacığı & veri hazırlayıcı
│       ├── views/             # 4 Ana Sekme Görünümü
│       │   ├── live_radar_view.py    # 2x2 MFD: Canlı radar, 3B görünüm, sentetik veri akışı & telemetri
│       │   ├── benchmark_view.py     # Mod A/B/C grafik ve metrik kıyaslama sekmesi
│       │   ├── test_runner_view.py   # Dahili GUI test koşturucu ve log sekmesi
│       │   └── config_view.py        # Sensör gecikmeleri ve filtre ayarları sekmesi
│       └── widgets/           # Özel UI bileşenleri
│           ├── radar_display.py       # [MOD-01A] 2B PPI Scope radar ekranı (FOV, Covariance Ellipse)
│           ├── gl_display.py          # [MOD-01B] 3B OpenGL yörünge ve LOS yaklaşma vektörü
│           ├── sensor_stream_panel.py # [MOD-01C] Canlı sentetik sensör paket akışı ve GT/Ham/EKF kıyaslama
│           ├── telemetry_panel.py     # [MOD-01D] Telemetri HUD & OOSM durum konsolu
│           └── control_bar.py         # Oynatıcı, hız çarpanı, mod seçici ve zaman çubuğu
├── tests/
│   ├── test_ekf.py            # Analitik Jacobian ve OOSM matematiksel doğrulama testleri
│   └── test_gui.py            # GUI başlatma, frame dispatch ve offscreen testleri
└── results/
    ├── ekf_oosm_comparison.png       # 4 Panelli benchmark kıyaslama grafiği
    └── ekf_fusion_results.png        # Yüksek çözünürlüklü füzyon grafiği
```

---

## 5. Ajanlar İçin Kodlama Standartları ve İlkeler

Tüm ajanlar projede değişiklik veya ekleme yaparken aşağıdaki kurallara **harfiyen** uymalıdır:

1. **Python 3.9+ Tip Belirteçleri (Type Annotations):**
   - Tüm yeni fonksiyon ve sınıflarda tam tip belirteçleri kullanılmalıdır (`from __future__ import annotations`).
   - Fonksiyon argümanları ve dönüş tipleri açıkça belirtilmelidir.
2. **Sayısal Hassasiyet ve Boyut Uyumu:**
   - Tüm durum vektörleri `shape=(6, 1)` ve matrisler `dtype=np.float64` olmalıdır.
   - 1B dizi beklenilen yerlerde (`.flatten()`) veya skalerlerde açık dönüşüm yapılmalıdır.
3. **Headless / Offscreen Test Güvenliği:**
   - GUI testlerinde veya otomatik CI koşturmalarında `os.environ["QT_QPA_PLATFORM"] = "offscreen"` ayarı mutlaka korunmalıdır. Ajan test çalıştırırken grafik ortamı yoksa crash olmamalıdır.
4. **Analitik ve Sayısal Jacobian Doğrulaması:**
   - Yeni bir sensör veya doğrusal olmayan ölçüm fonksiyonu eklendiğinde `tests/test_ekf.py` içerisine merkezi sonlu farklar (central finite difference) ile analitik Jacobian'ı kıyaslayan bir test eklenmelidir.
5. **Simülasyon Modu Bütünlüğü:**
   - Projede 3 mod kıyaslaması esastır:
     - **Mod A:** İdeal (Sıfır gecikme).
     - **Mod B:** Naive (Gecikmeler var ama filtre habersiz; ölçüm varış anında işlenir).
     - **Mod C:** OOSM Rollback/Replay destekli gecikme telafili EKF.
   - Yeni bir algoritma eklendiğinde bu mod mimarisini bozmamalıdır.
6. **Mevcut Yorum ve Dökümantasyonun Korunması:**
   - Var olan fonksiyon docstring'leri ve matematiksel açıklamalar silinmemeli, geliştirilmelidir.
7. **Taktik Arayüz ve Tasarım Standartları (GUI):**
   - Arayüzde kesinlikle emoji kullanılmamalı; `[MOD-01A]`, `[KIN-01]` gibi teknik modül kodları, keskin köşeli (`0px` radius) koyu karbon-arduvaz paneller ve monospaced (`Consolas`) tipografi korunmalıdır.
   - Kurum veya marka adları (`ASELSAN`, `C4ISR` vb.) arayüz başlıklarında kullanılmamalıdır.

---

## 6. Sık Kullanılan Komutlar (Hızlı Başvuru)

- **Tüm Birim ve GUI Testlerini Çalıştırma:**
  ```powershell
  python -m unittest discover tests
  ```
- **Kıyaslama Benchmark'ını ve Grafikleri Üretme:**
  ```powershell
  python src/main.py
  ```
- **Canlı Taktik Radar GUI Konsolunu Başlatma:**
  ```powershell
  python -m src.gui.app
  ```
- **Tek Tıkla Menülü Başlatıcı:**
  ```powershell
  .\start.ps1
  ```
