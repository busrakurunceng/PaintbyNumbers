# ==================================================
# MAIN - Paint by Numbers Pipeline
# ==================================================

from config import IMAGE_PATH, OUTPUT_DIR, K_CLUSTERS, RANDOM_STATE

from src.image_io import load_image, save_image
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

    # 2. Piksel çıkarma
    print("\n[ADIM 2] Piksel matrisi hazırlanıyor...")
    pixels = extract_pixels(image)

    # 3. K-Means kümeleme
    print("\n[ADIM 3] K-Means kümeleme başlıyor...")
    labels, centers = apply_kmeans(pixels, K_CLUSTERS, RANDOM_STATE)
    dominant_colors = get_dominant_colors(centers, labels)

    # 4. Segmentasyon
    print("\n[ADIM 4] Segmentasyon yapılıyor...")
    segmented = segment_image(labels, centers, image.shape)
    label_map = create_label_map(labels, image.shape)
    save_image(segmented, "segmented.png", OUTPUT_DIR)

    # 5. Renk kategorizasyonu
    print("\n[ADIM 5] Renkler isimlendiriliyor...")
    color_names = categorize_centers(centers)

    # --- Buradan sonrası adım adım eklenecek ---
    # ADIM 6: Preprocessing (bilateral filter)
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
