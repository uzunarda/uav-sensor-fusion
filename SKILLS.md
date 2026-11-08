# SKILLS.md - Operational Playbooks & Engineering Workflows for AI Agents

Bu doküman, projede görev alacak AI ajanlarının ve yazılımcıların sık tekrarlanan operasyonel, algoritmik ve arayüz geliştirme görevlerini sıfır hata ile yürütebilmeleri için hazırlanmış adım adım uygulanabilir yetenek (skill) rehberidir.

---

## Yetenek Haritası (Skills Index)

| Yetenek Adı | Amaç | İlgili Dosyalar |
|---|---|---|
| **1. `run-tests`** | Birim ve GUI testlerini koşturma, analitik Jacobian doğrulaması | `tests/test_ekf.py`, `tests/test_gui.py` |
| **2. `run-benchmark`** | Mod A/B/C kıyaslama simülasyonunu koşturma ve RMSE metriklerini çıkarma | `src/main.py`, `results/` |
| **3. `run-gui`** | Canlı taktik radar konsolunu çalıştırma ve offscreen GUI testi yapma | `src/gui/app.py`, `src/gui/worker.py` |
| **4. `add-sensor`** | Yeni bir sensör modeli ekleme (ör. Doppler Radar, Altimetre, LiDAR) | `src/sensors.py`, `src/ekf.py`, `tests/test_ekf.py` |
| **5. `tune-filter`** | Süreç gürültüsü $q_c$ ve kovaryans matrislerini kalibre etme | `src/ekf.py`, `src/main.py` |
| **6. `debug-oosm`** | Sıra Dışı Ölçüm (OOSM) Rollback / Replay mekanizmasını denetleme | `src/ekf.py`, `src/simulator.py` |

---

## 1. Yetenek: `run-tests` (Test Koşturma ve Jacobian Doğrulama)

### Amaç:
Kod tabanında yapılan herhangi bir değişiklik sonrasında tüm matematiksel ve arayüz testlerinin yeşil kaldığını doğrulamak.

### Çalıştırma Komutları:
```powershell
# Tüm testleri keşfet ve koştur
python -m unittest discover tests

# Sadece EKF ve matematiksel testleri koştur
python -m unittest tests.test_ekf

# Sadece GUI yaşam döngüsü testlerini koştur (offscreen modunda)
python -m unittest tests.test_gui
```

### Kritik İpuçları:
- GUI testleri başlarken `os.environ["QT_QPA_PLATFORM"] = "offscreen"` atanmalıdır.
- Analitik Jacobian testleri (`test_radar_jacobian_numerical`, `test_camera_jacobian_numerical`) merkezi sonlu farklar ($eps = 10^{-6}$) ile doğrulanır:
  $$\frac{\partial h_i}{\partial x_j} \approx \frac{h_i(\mathbf{x} + \epsilon \mathbf{e}_j) - h_i(\mathbf{x} - \epsilon \mathbf{e}_j)}{2\epsilon}$$
- Tolerans: `np.testing.assert_allclose(H_analytic, H_numeric, atol=1e-5)` sağlanmalıdır.

---

## 2. Yetenek: `run-benchmark` (Başarım Kıyaslama ve Raporlama)

### Amaç:
Mod A (İdeal), Mod B (Naive) ve Mod C (OOSM Destekli) karşılaştırmalı simülasyonunu çalıştırarak RMSE değerlerini ve `results/` altındaki görsel raporları üretmek.

### Çalıştırma Komutu:
```powershell
python src/main.py
```

### Doğrulama Adımları:
1. Konsolda şu başlıkların eksiksiz yazıldığından emin olun:
   - `Mod A (Ideal / Zero Latency)`
   - `Mod B (Naive / Latency Ignored)`
   - `Mod C (Rollback / OOSM Delay Compensated)`
2. `results/ekf_oosm_comparison.png` dosyasının güncellendiğini doğrulayın.
3. **Kalıcı Durum Konum RMSE Kriteri:** Mod C'nin Mod B'ye kıyasla en az **%20** iyileşme sağladığını teyit edin (Beklenen: Mod B $\approx 3.46\text{ m}$, Mod C $\approx 2.68\text{ m}$).
4. İşlenen OOSM sayısının $100+$ olduğunu ve `Dropped OOSM = 0` olduğunu gözlemleyin.

---

## 3. Yetenek: `run-gui` (Canlı Taktik Radar Konsolu)

### Amaç:
60 FPS PyQt6 + PyQtGraph taktik radar konsolunu başlatmak veya GUI geliştirmeleri yapmak.

### Çalıştırma Komutları:
```powershell
# GUI konsolunu başlatma
python -m src.gui.app

# Veya başlatıcı betiği ile:
.\start.ps1 -Mode gui
```

### Mimari Detaylar:
- **`SimulationWorker` (`src/gui/worker.py`):**
  - Başlangıçta simülasyonu 3 mod için de (Mod A, Mod B, Mod C) önceden çözer (`SimulationFrame` nesneleri).
  - 60 FPS QTimer (`16 ms`) ile zaman çizelgesini ilerletir.
  - Mod değiştirildiğinde simülasyon sıfırlanmaz, mevcut $t$ anından yeni modun önbelleğine kesintisiz geçer.
- **Bileşen Hiyerarşisi (`src/gui/widgets/`):**
  - `radar_display.py`: `pg.PlotWidget` üzerinde PPI Scope, menzil halkaları ($500\text{ m}, 1000\text{ m}, 1500\text{ m}$), dinamik kamera FOV konisi ve 2D kovaryans elipsi ($P_{xx}, P_{yy}, P_{xy}$).
  - `gl_display.py`: `GLViewWidget` ile 3B yörünge, zemin ızgarası ve İHA-Hedef LOS (görüş hattı) çizgisi.
  - `telemetry_panel.py`: HUD metrikleri ve 150 ms OOSM ikaz flaşı.
  - `control_bar.py`: Oynat/Durdur/Sıfırla, Hız çarpanı, Mod seçici radio butonlar, Zaman kaydırıcı.

---

## 4. Yetenek: `add-sensor` (Yeni Sensör Modeli Ekleme Adımları)

Yeni bir sensör türü (örneğin Doppler Hız Radarı, Barometrik İrtifa Ölçer veya LiDAR) entegre edilirken aşağıdaki adımlar eksiksiz izlenmelidir:

### Adım 1: Sensör Sınıfını Tanımlayın (`src/sensors.py`)
`BaseSensor` sınıfından türetin:
```python
class AltimeterSensor(BaseSensor):
    def __init__(self, frequency_hz=20.0, std_dev=1.5, **kwargs):
        super().__init__(frequency_hz=frequency_hz, **kwargs)
        self.std_dev = float(std_dev)

    def measure(self, timestamp: float, target_pos: np.ndarray, uav_pos: Optional[np.ndarray] = None) -> SensorMeasurement:
        t_capture = float(timestamp)
        # NED koordinatında irtifa: -z
        z_true = -target_pos[2]
        noise = float(self.rng.normal(0.0, self.std_dev))
        z_meas = np.array([z_true + noise], dtype=np.float64)
        R = np.array([[self.std_dev ** 2]], dtype=np.float64)

        t_receive = self.compute_receive_time(t_capture)
        return SensorMeasurement(
            t_capture=t_capture,
            sensor_type="ALTIMETER",
            z=z_meas,
            R_cov=R,
            t_receive=t_receive,
            uav_pos=uav_pos,
        )
```

### Adım 2: EKF Ölçüm Fonksiyonu ve Analitik Jacobi'yi Yazın (`src/ekf.py`)
```python
def h_altimeter(self, x: np.ndarray, uav_pos: Optional[np.ndarray] = None) -> np.ndarray:
    # Durum vektörü: x = [x, vx, y, vy, z, vz]^T
    # Altimetre ölçümü: -z (NED'de irtifa)
    return np.array([[-x[4, 0]]], dtype=np.float64)

def jacobian_altimeter(self, x: np.ndarray, uav_pos: Optional[np.ndarray] = None) -> np.ndarray:
    # dh/dz = -1, diğer durumlar 0
    H = np.zeros((1, 6), dtype=np.float64)
    H[0, 4] = -1.0
    return H
```

### Adım 3: EKF Güncelleme Metoduna Dallanma Ekleyin (`src/ekf.py`)
`update_measurement` fonksiyonunda `measurement.sensor_type == "ALTIMETER"` durumunu ekleyerek `h_altimeter` ve `jacobian_altimeter` çağrılarını bağlayın.

### Adım 4: Birim Testini Yazın (`tests/test_ekf.py`)
Sonlu farklar ile analitik Jacobian'ın uyumunu test edin.

---

## 5. Yetenek: `tune-filter` (Kovaryans ve Gürültü Kalibrasyonu)

Filtre başarımını optimize etmek veya hedef manevralarına göre ayarlamak için:

1. **Süreç Gürültüsü Spektral Yoğunluğu ($q_c$):**
   - Parametre: `process_noise_spectral_density` (`src/ekf.py`).
   - $q_c$ artırılırsa: Filtre hedef manevralarına daha hızlı tepki verir, ancak ölçüm gürültülerine karşı hassaslaşır (tahmin dalgalanır).
   - $q_c$ azaltılırsa: Filtre sabit hız varsayımına daha katı bağlanır, pürüzsüzleşir; ancak manevralarda gecikmeli takip oluşur.
   - Tipik İHA hedefi için önerilen aralık: $q_c \in [0.5, 2.5] \text{ m}^2/\text{s}^3$.

2. **Ölçüm Gürültüsü Kovaryansı ($R$):**
   - `RadarSensor`: Range standart sapması $\sigma_r = 5.0\text{ m} \implies R = 25.0$.
   - `CameraSensor`: Azimut/Elevasyon sapması $\sigma_\psi = \sigma_\lambda = 0.005\text{ rad} \implies R = \text{diag}([2.5 \times 10^{-5}, 2.5 \times 10^{-5}])$.

3. **Kovaryans Elipsi ($2\sigma$ Uncertainty Ellipse):**
   - GUI'deki elips, $P$'nin $x$ ve $y$ alt bloğunun ($2 \times 2$) özdeğer (eigenvalues) ve özvektörlerinden (eigenvectors) türetilir.
   - Elipsin aşırı büyümesi filtrenin ıraksadığına (divergence), aşırı küçülüp hatalı kalması ise aşırı güvene (overconfidence) işaret eder.

---

## 6. Yetenek: `debug-oosm` (OOSM Tampon Denetimi)

OOSM Rollback mekanizmasında hata ayıklarken kontrol edilecek unsurlar:

1. **Tampon Yaşı Sınırı (`max_buffer_dt`):**
   - Varsayılan: `1.0 saniye`.
   - Eğer bir ölçümün gecikmesi $\Delta t_{delay} > \text{max\_buffer\_dt}$ ise bu ölçüm bayat (stale) kabul edilip atılır (`dropped_oosm_count += 1`).
2. **Kuyruk Sıralaması:**
   - `SensorMeasurement.__lt__`: Olaylar kesinlikle `t_receive` zamanına göre sıralanmalıdır:
     ```python
     if abs(self.t_receive - other.t_receive) < 1e-12:
         return self.t_capture < other.t_capture
     return self.t_receive < other.t_receive
     ```
3. **Rollback Döngüsünde Durum Bütünlüğü:**
   - Rollback yapıldığında `history` tamponundaki eski tahminler ve kovaryanslar temizlenmeli, sadece geri sarılan noktaya kadar olan kısım korunup sonrasındaki tüm olaylar sırasıyla tekrar işletilmelidir.
