# ==================================================
# PROJECT CONFIGURATION - Paint by Numbers
# ==================================================
# Tüm sabit değerler burada tanımlanır.
# Magic number kullanmak yerine bu dosyadan import edilir.

# --- Görüntü Ayarları ---
IMAGE_PATH = "data/aslan resmi.jpg"
OUTPUT_DIR = "outputs/"

# --- K-Means Kümeleme ---
K_CLUSTERS = 16
RANDOM_STATE = 42

# --- Preprocessing (Görüntü Yumuşatma) ---
# Bilateral Filter parametreleri
# Kenarları koruyarak gürültüyü azaltır
BILATERAL_D = 22              # Filtre çapı (piksel komşuluğu)
BILATERAL_SIGMA_COLOR = 50   # Renk uzayında sigma (benzer renk toleransı)
BILATERAL_SIGMA_SPACE = 50   # Koordinat uzayında sigma (mesafe toleransı)

# --- Bölge Temizleme (Dinamik Threshold) ---
# Sabit piksel yerine görüntü çözünürlüğüne oransal çalışır.
# Gerçek piksel değeri main.py'da hesaplanır: total_pixels * oran
MIN_REGION_RATIO = 0.002     # Bu orandan küçük bölgeler komşu renge katılır
CONTRAST_THRESHOLD = 50.0    # LAB mesafesi bu değerin üstündeyse küçük bölge korunur
                             # Sadece gerçekten belirgin detaylar korunur (göz vb.)

# --- Kontur ve Çizgi Ayarları ---
LINE_COLOR = (210, 210, 210)    # açık gri (tam beyaz değil, gözüksün diye)
LINE_THICKNESS = 1            # Çizgi kalınlığı (piksel)

# --- Numara Yerleştirme ---
FONT_COLOR = (210, 210, 210)    # Numara rengi
FONT_THICKNESS = 1            # Numara kalınlığı
MIN_LABEL_RATIO = 0.004      # Bu orandan küçük alanlara numara yazılmaz
