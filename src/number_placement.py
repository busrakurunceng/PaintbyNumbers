# ==================================================
# NUMBER PLACEMENT - Numara Yerleştirme
# ==================================================
# Her bölgenin ağırlık merkezine (centroid) o bölgenin
# renk numarasını yazar. Çok küçük alanlara numara yazılmaz.
# Font boyutu alan büyüklüğüne göre dinamik ayarlanır.

import cv2
import numpy as np


def calculate_centroids(
    region_map: np.ndarray,
    label_map: np.ndarray,
    min_area: int = 800,
) -> list:
    """Her bölgenin merkez koordinatını ve renk ID'sini hesaplar.

    cv2.moments() ile ağırlık merkezi bulunur.
    Çok küçük alanlar atlanır.

    Args:
        region_map: Benzersiz bölge ID'leri (H, W).
        label_map: Renk küme etiketleri (H, W).
        min_area: Bu pikselden küçük alanlara numara konmaz.

    Returns:
        centroids: Her eleman {"region_id", "color_id", "cx", "cy", "area"} dict'i.
    """
    centroids = []
    unique_regions = np.unique(region_map)

    for region_id in unique_regions:
        if region_id == 0:
            continue

        mask = (region_map == region_id).astype(np.uint8)
        area = int(np.sum(mask))

        if area < min_area:
            continue

        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            continue

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        color_id = int(label_map[region_map == region_id][0])

        centroids.append({
            "region_id": region_id,
            "color_id": color_id,
            "cx": cx,
            "cy": cy,
            "area": area,
        })

    print(f"[OK] Centroid hesaplandı.")
    print(f"     Numaralanacak bölge: {len(centroids)}")

    return centroids


def place_numbers(
    canvas: np.ndarray,
    centroids: list,
    font_color: tuple = (60, 60, 60),
    font_thickness: int = 1,
) -> np.ndarray:
    """Tuval üzerine renk numaralarını yazar.

    Font boyutu alanın büyüklüğüne göre dinamik ayarlanır.
    Numara mümkün olduğunca merkezde kalır.

    Args:
        canvas: Beyaz zemin + kontur çizgileri (H, W, 3).
        centroids: calculate_centroids() çıktısı.
        font_color: Numara rengi (R, G, B).
        font_thickness: Numara kalınlığı.

    Returns:
        numbered_canvas: Numaralar eklenmiş tuval (H, W, 3).
    """
    result = canvas.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for c in centroids:
        area = c["area"]
        text = str(c["color_id"])

        # Alan büyüklüğüne göre font boyutu
        if area > 10000:
            font_scale = 0.5
        elif area > 3000:
            font_scale = 0.4
        else:
            font_scale = 0.3

        # Metnin boyutunu hesapla, merkeze oturt
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        tx = c["cx"] - tw // 2
        ty = c["cy"] + th // 2

        cv2.putText(result, text, (tx, ty), font, font_scale,
                    font_color, font_thickness, cv2.LINE_AA)

    print(f"[OK] Numaralar yerleştirildi.")
    print(f"     Yazılan numara: {len(centroids)}")

    return result
