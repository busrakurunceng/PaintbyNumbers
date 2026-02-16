# ==================================================
# PROJECT CONFIGURATION - Paint by Numbers
# ==================================================
# Tüm sabit değerler burada tanımlanır.
# Magic number kullanmak yerine bu dosyadan import edilir.

# --- Görüntü Ayarları ---
IMAGE_PATH = "data/sample.jpg"
OUTPUT_DIR = "outputs/"

# --- K-Means Kümeleme ---
K_CLUSTERS = 16
RANDOM_STATE = 42

# --- Preprocessing (Görüntü Yumuşatma) ---
# Bilateral Filter parametreleri
# Kenarları koruyarak gürültüyü azaltır
BILATERAL_D = 9              # Filtre çapı (piksel komşuluğu)
BILATERAL_SIGMA_COLOR = 75   # Renk uzayında sigma (benzer renk toleransı)
BILATERAL_SIGMA_SPACE = 75   # Koordinat uzayında sigma (mesafe toleransı)

# --- Bölge Temizleme ---
# Bu pikselden küçük bölgeler komşu renge katılır
MIN_REGION_AREA = 500

# --- Kontur ve Çizgi Ayarları ---
LINE_COLOR = (80, 80, 80)    # Koyu gri (tam siyah değil, boya kapatsın diye)
LINE_THICKNESS = 2            # Çizgi kalınlığı (piksel)

# --- Numara Yerleştirme ---
FONT_COLOR = (60, 60, 60)    # Numara rengi
FONT_THICKNESS = 1            # Numara kalınlığı
MIN_AREA_FOR_LABEL = 800     # Bu pikselden küçük alanlara numara yazılmaz
