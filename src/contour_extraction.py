# ==================================================
# CONTOUR EXTRACTION - Sınır Çizgileri Çıkarma
# ==================================================
# Temizlenmiş bölge haritasındaki her bölgenin dış hatlarını bulur.
# Boyama kitabındaki siyah çizgiler bunlardır.

import cv2
import numpy as np


def extract_contours(region_map: np.ndarray) -> list:
    """Her bölgenin konturlarını çıkarır.

    region_map'teki her benzersiz bölge ID'si için
    dış hat (contour) hesaplar.

    Args:
        region_map: Benzersiz bölge ID'leri (H, W).

    Returns:
        contour_list: Her eleman (region_id, contour) tuple'ı.
    """
    contour_list = []
    unique_regions = np.unique(region_map)

    for region_id in unique_regions:
        if region_id == 0:
            continue

        mask = (region_map == region_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            contour_list.append((region_id, contour))

    print(f"[OK] Kontur çıkarma tamamlandı.")
    print(f"     Toplam kontur: {len(contour_list)}")

    return contour_list


def draw_contours_on_canvas(
    shape: tuple,
    contour_list: list,
    line_color: tuple = (80, 80, 80),
    line_thickness: int = 2,
) -> np.ndarray:
    """Beyaz tuval üzerine konturları çizer.

    Args:
        shape: Görüntü boyutu (H, W, 3).
        contour_list: extract_contours() çıktısı.
        line_color: Çizgi rengi (R, G, B).
        line_thickness: Çizgi kalınlığı (piksel).

    Returns:
        canvas: Beyaz zemin + siyah sınır çizgileri (H, W, 3) - uint8.
    """
    canvas = np.ones(shape, dtype=np.uint8) * 255

    contours_only = [contour for _, contour in contour_list]
    cv2.drawContours(canvas, contours_only, -1, line_color, line_thickness)

    print(f"[OK] Tuval çizildi.")
    print(f"     Çizgi rengi: {line_color}, kalınlık: {line_thickness}px")

    return canvas
