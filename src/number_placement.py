# ==================================================
# NUMBER PLACEMENT - Numara Yerleştirme
# ==================================================
# Distance transform ile her bölgenin en geniş iç noktasını
# bulur, dinamik font ölçekleme ile numarayı yerleştirir.
# Bounding box yerine inscribed circle yöntemi kullanılır.

import cv2
import numpy as np


def calculate_centroids(
    region_map: np.ndarray,
    label_map: np.ndarray,
    min_area: int = 800,
) -> list:
    """Her bölgenin en geniş iç noktasını ve iç çember yarıçapını hesaplar.

    Distance transform ile bölge içindeki en uzak noktayı (kenarlardan)
    bulur. Bu nokta, numaranın en rahat sığacağı yerdir.

    Args:
        region_map: Benzersiz bölge ID'leri (H, W).
        label_map: Renk küme etiketleri (H, W).
        min_area: Bu pikselden küçük alanlara numara konmaz.

    Returns:
        centroids: Her eleman {"region_id", "color_id", "cx", "cy", "area", "radius"} dict'i.
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

        # Distance transform: her pikselin en yakın kenara mesafesi
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, max_radius, _, max_loc = cv2.minMaxLoc(dist)

        if max_radius < 1:
            continue

        cx, cy = max_loc  # En geniş iç noktanın koordinatları
        color_id = int(label_map[region_map == region_id][0])

        centroids.append({
            "region_id": region_id,
            "color_id": color_id,
            "cx": cx,
            "cy": cy,
            "area": area,
            "radius": float(max_radius),
        })

    print(f"[OK] Centroid hesaplandı (distance transform).")
    print(f"     Numaralanacak bölge: {len(centroids)}")

    return centroids


def _fit_font_scale(text, font, radius, thickness):
    """Metnin iç çembere sığacağı font ölçeğini hesaplar.

    radius'a oransal scale hesaplar, ardından metnin
    gerçekten sığıp sığmadığını doğrular. Sığmıyorsa
    küçülterek tekrar dener.

    Args:
        text: Yazılacak metin.
        font: OpenCV font tipi.
        radius: İç çember yarıçapı (piksel).
        thickness: Font kalınlığı.

    Returns:
        Uygun font_scale veya None (sığmıyorsa).
    """
    MAX_SCALE = 0.45
    MIN_SCALE = 0.15

    scale = min(MAX_SCALE, radius / 30)
    scale = max(scale, MIN_SCALE)

    diameter = radius * 2 * 0.85

    # Sığana kadar küçült
    while scale >= MIN_SCALE:
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        if tw < diameter and th < diameter:
            return scale
        scale -= 0.02

    return None


def place_numbers(
    canvas: np.ndarray,
    centroids: list,
    font_color: tuple = (60, 60, 60),
    font_thickness: int = 1,
) -> np.ndarray:
    """Tuval üzerine renk numaralarını yazar.

    Distance transform'dan gelen iç çember yarıçapına göre
    dinamik font ölçekleme yapar. Sığmayan numaralar yazılmaz.

    Args:
        canvas: Beyaz zemin + kontur çizgileri (H, W, 3).
        centroids: calculate_centroids() çıktısı.
        font_color: Numara rengi (R, G, B).
        font_thickness: Numara kalınlığı.

    Returns:
        numbered_canvas: Numaralar eklenmiş tuval (H, W, 3).
    """
    result = canvas.copy()
    h, w = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    placed = 0
    skipped = 0

    for c in centroids:
        text = str(c["color_id"])
        radius = c["radius"]

        font_scale = _fit_font_scale(text, font, radius, font_thickness)

        if font_scale is None:
            skipped += 1
            continue

        (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        tx = c["cx"] - tw // 2
        ty = c["cy"] + th // 2

        # Görüntü sınırı kontrolü: taşanları içeri çek veya atla
        if tx < 0:
            tx = 1
        if tx + tw > w:
            tx = w - tw - 1
        if ty - th < 0:
            ty = th + 1
        if ty > h:
            ty = h - 1

        # Hala sınır dışındaysa atla
        if tx < 0 or ty - th < 0 or tx + tw > w or ty > h:
            skipped += 1
            continue

        cv2.putText(result, text, (tx, ty), font, font_scale,
                    font_color, font_thickness, cv2.LINE_AA)
        placed += 1

    print(f"[OK] Numaralar yerleştirildi.")
    print(f"     Yazılan: {placed}, Sığmayan (atlandı): {skipped}")

    return result
