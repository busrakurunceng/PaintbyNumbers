# ==================================================
# MAIN - Paint by Numbers Pipeline
# ==================================================
# Şu an mevcut olan modülleri sırayla çalıştırır.
# Yeni modüller eklendikçe buraya adım adım eklenecek.

import numpy as np

from config import (
    IMAGE_PATH, OUTPUT_DIR, K_CLUSTERS, RANDOM_STATE,
    BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE,
    MIN_REGION_RATIO_DETAIL, MIN_REGION_RATIO_BACKGROUND,
    DETAIL_EDGE_THRESHOLD, CONTRAST_THRESHOLD, MIN_LABEL_RATIO,
    LINE_COLOR, LINE_THICKNESS,
    FONT_COLOR, FONT_THICKNESS,
)

from src.image_io import load_image, save_image
from src.preprocessing import apply_bilateral_filter
from src.pixel_analysis import get_image_info, extract_pixels
from src.clustering import apply_kmeans, get_dominant_colors
from src.segmentation import segment_image, create_label_map
from src.color_categorization import categorize_centers
from src.region_detection import detect_regions, build_detail_map, remove_small_regions
from src.contour_extraction import extract_contours, draw_contours_on_canvas
from src.number_placement import calculate_centroids, place_numbers
from src.rendering import render_reference_image, render_color_legend
from src.visualization import save_comparison


def main():
    """Paint by Numbers Pipeline - Ana fonksiyon."""

    print("=" * 55)
    print("  PAINT BY NUMBERS PIPELINE")
    print("=" * 55)

    # 1. Görüntü yükleme
    print("\n[ADIM 1] Görüntü yükleniyor...")
    image = load_image(IMAGE_PATH)
    info = get_image_info(image)

    # Dinamik threshold hesaplama (piksel sayısına oransal)
    total_pixels = info["total_pixels"]
    min_area_detail = int(total_pixels * MIN_REGION_RATIO_DETAIL)
    min_area_background = int(total_pixels * MIN_REGION_RATIO_BACKGROUND)
    min_area_for_label = int(total_pixels * MIN_LABEL_RATIO)
    print(f"     Eşikler: detay={min_area_detail}px, arka plan={min_area_background}px, label={min_area_for_label}px")

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

    # 7. Detay haritası + bölge tespiti + gürültü temizleme
    print("\n[ADIM 7] Detay haritası ve bölge tespiti...")
    detail_map = build_detail_map(smoothed)
    region_map = detect_regions(label_map, K_CLUSTERS)
    region_map, label_map = remove_small_regions(
        region_map, label_map, centers, detail_map,
        min_area_detail, min_area_background,
        CONTRAST_THRESHOLD, DETAIL_EDGE_THRESHOLD,
    )

    # Temizlenmiş segmented görüntüyü kaydet
    cleaned_segmented = segment_image(label_map.ravel(), centers, smoothed.shape)
    save_image(cleaned_segmented, "cleaned_segmented.png", OUTPUT_DIR)

    # Önce/sonra karşılaştırması
    region_count = len(np.unique(region_map)) - 1
    save_comparison(
        segmented, cleaned_segmented,
        "Temizleme Öncesi", f"Temizleme Sonrası ({region_count} bölge)",
        "comparison_cleanup.png", OUTPUT_DIR,
    )

    # 8. Kontur çıkarma
    print("\n[ADIM 8] Konturlar çıkarılıyor...")
    contour_list = extract_contours(region_map)
    canvas = draw_contours_on_canvas(smoothed.shape, contour_list, LINE_COLOR, LINE_THICKNESS)
    save_image(canvas, "canvas_outline.png", OUTPUT_DIR)

    # 9. Numara yerleştirme
    print("\n[ADIM 9] Numaralar yerleştiriliyor...")
    centroids = calculate_centroids(region_map, label_map, min_area_for_label)
    numbered_canvas = place_numbers(canvas, centroids, FONT_COLOR, FONT_THICKNESS)
    save_image(numbered_canvas, "canvas_numbered.png", OUTPUT_DIR)

    # 10. Referans resim + renk legendı
    print("\n[ADIM 10] Referans resim ve renk legendı oluşturuluyor...")
    reference = render_reference_image(
        cleaned_segmented, contour_list, centroids,
        LINE_COLOR, 1, FONT_COLOR, FONT_THICKNESS,
    )
    save_image(reference, "reference.png", OUTPUT_DIR)
    render_color_legend(centers, color_names, OUTPUT_DIR)

    print("\n" + "=" * 55)
    print("  PIPELINE TAMAMLANDI!")
    print(f"  Çıktılar: {OUTPUT_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    main()
