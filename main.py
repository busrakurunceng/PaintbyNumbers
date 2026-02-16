# ==================================================
# MAIN - Paint by Numbers Pipeline
# ==================================================
# Şu an mevcut olan modülleri sırayla çalıştırır.
# Yeni modüller eklendikçe buraya adım adım eklenecek.

from config import (
    IMAGE_PATH, OUTPUT_DIR, K_CLUSTERS, RANDOM_STATE,
    BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE,
)

from src.image_io import load_image, save_image
from src.preprocessing import apply_bilateral_filter
from src.pixel_analysis import get_image_info, extract_pixels
from src.clustering import apply_kmeans, get_dominant_colors
from src.segmentation import segment_image, create_label_map
from src.color_categorization import categorize_centers


def main():
    """Paint by Numbers Pipeline - Ana fonksiyon."""

    print("=" * 55)
    print("  PAINT BY NUMBERS PIPELINE")
    print("=" * 55)

    # 1. Görüntü yükleme
    print("\n[ADIM 1] Görüntü yükleniyor...")
    image = load_image(IMAGE_PATH)
    get_image_info(image)

    # 2. Preprocessing (yumuşatma)
    print("\n[ADIM 2] Görüntü yumuşatılıyor...")
    smoothed = apply_bilateral_filter(
        image, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE
    )
    save_image(smoothed, "smoothed.png", OUTPUT_DIR)

    # 3. Piksel çıkarma (yumuşatılmış görüntüden)
    print("\n[ADIM 3] Piksel matrisi hazırlanıyor...")
    pixels = extract_pixels(smoothed)

    # 4. K-Means kümeleme
    print("\n[ADIM 4] K-Means kümeleme başlıyor...")
    labels, centers = apply_kmeans(pixels, K_CLUSTERS, RANDOM_STATE)
    dominant_colors = get_dominant_colors(centers, labels)

    # 5. Segmentasyon
    print("\n[ADIM 5] Segmentasyon yapılıyor...")
    segmented = segment_image(labels, centers, smoothed.shape)
    label_map = create_label_map(labels, smoothed.shape)
    save_image(segmented, "segmented.png", OUTPUT_DIR)

    # 6. Renk kategorizasyonu
    print("\n[ADIM 6] Renkler isimlendiriliyor...")
    color_names = categorize_centers(centers)

    # --- Buradan sonrası adım adım eklenecek ---
    # ADIM 7: Bölge tespiti (connected components)
    # ADIM 8: Kontur çıkarma
    # ADIM 9: Numara yerleştirme
    # ADIM 10: Tuval + legend render

    print("\n" + "=" * 55)
    print("  MEVCUT PIPELINE TAMAMLANDI!")
    print(f"  Çıktılar: {OUTPUT_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    main()
