# ==================================================
# PROJECT CONFIGURATION - Paint by Numbers
# ==================================================
# Tüm sabit değerler burada tanımlanır.
# Magic number kullanmak yerine bu dosyadan import edilir.

# --- Mod: "general" = manzara/hayvan (aslan vb.), "portrait" = yüz/couple ---
MODE = "portrait"

# --- Görüntü Ayarları ---
IMAGE_PATH = "data/couple.jpg"
OUTPUT_DIR = "outputs/"

# --- K-Means Kümeleme ---
RANDOM_STATE = 42
if MODE == "portrait":
    K_CLUSTERS = 22
else:
    K_CLUSTERS = 16

# --- Preprocessing (Görüntü Yumuşatma) ---
# Bilateral Filter parametreleri
# Kenarları koruyarak gürültüyü azaltır
BILATERAL_D = 12              # Filtre çapı (küçük = daha fazla detay korunur)
BILATERAL_SIGMA_COLOR = 50   # Renk uzayında sigma (benzer renk toleransı)
BILATERAL_SIGMA_SPACE = 50   # Koordinat uzayında sigma (mesafe toleransı)

# --- Bölge Temizleme (Çift Eşik Sistemi) ---
# Detay bölgelerinde (göz, burun, yüz) düşük threshold,
# arka plan bölgelerinde yüksek threshold uygulanır.
if MODE == "portrait":
    MIN_REGION_RATIO_DETAIL = 0.0002
    MIN_REGION_RATIO_BACKGROUND = 0.002
    DETAIL_EDGE_THRESHOLD = 0.25
    CONTRAST_THRESHOLD = 55.0
else:
    MIN_REGION_RATIO_DETAIL = 0.0005
    MIN_REGION_RATIO_BACKGROUND = 0.003
    DETAIL_EDGE_THRESHOLD = 0.3
    CONTRAST_THRESHOLD = 60.0

# --- Kontur ve Çizgi Ayarları ---
LINE_COLOR = (210, 210, 210)    # açık gri (tam beyaz değil, gözüksün diye)
LINE_THICKNESS = 1            # Çizgi kalınlığı (piksel)

# --- Numara Yerleştirme ---
FONT_COLOR = (210, 210, 210)    # Numara rengi
FONT_THICKNESS = 1            # Numara kalınlığı
MIN_LABEL_RATIO = 0.0005     # Bu orandan küçük alanlara numara yazılmaz
