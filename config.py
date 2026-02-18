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
BILATERAL_D = 12              # Filtre çapı (küçük = daha fazla detay korunur)
BILATERAL_SIGMA_COLOR = 50   # Renk uzayında sigma (benzer renk toleransı)
BILATERAL_SIGMA_SPACE = 50   # Koordinat uzayında sigma (mesafe toleransı)

# --- Bölge Temizleme (Çift Eşik Sistemi) ---
# Detay bölgelerinde (göz, burun, yüz) düşük threshold,
# arka plan bölgelerinde yüksek threshold uygulanır.
MIN_REGION_RATIO_DETAIL = 0.0005      # Detay bölgelerinde: küçük alanlar korunur
MIN_REGION_RATIO_BACKGROUND = 0.003   # Arka planda: agresif temizlik
DETAIL_EDGE_THRESHOLD = 0.3           # Bu yoğunluğun üstü "detay" sayılır
CONTRAST_THRESHOLD = 60.0             # LAB mesafesi bu değerin üstündeyse bölge korunur

# --- Kontur ve Çizgi Ayarları ---
LINE_COLOR = (210, 210, 210)    # açık gri (tam beyaz değil, gözüksün diye)
LINE_THICKNESS = 1            # Çizgi kalınlığı (piksel)

# --- Numara Yerleştirme ---
FONT_COLOR = (210, 210, 210)    # Numara rengi
FONT_THICKNESS = 1            # Numara kalınlığı
MIN_LABEL_RATIO = 0.0005     # Bu orandan küçük alanlara numara yazılmaz
