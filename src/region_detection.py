# ==================================================
# REGION DETECTION - Bölge Tespiti
# ==================================================
# Detay haritası bazlı çift eşik sistemi:
# - Detay bölgelerinde (göz, burun, yüz) düşük threshold
# - Arka plan bölgelerinde yüksek threshold
# Edge yoğunluğuna göre otomatik karar verir.

import cv2
import numpy as np


def detect_regions(label_map: np.ndarray, k: int) -> np.ndarray:
    """Her renk kümesini ayrı bağlı bölgelere ayırır.

    Args:
        label_map: 2D etiket haritası (H, W).
        k: Toplam küme sayısı.

    Returns:
        region_map: 2D harita (H, W) - her piksel benzersiz bölge ID'si taşır.
    """
    region_map = np.zeros_like(label_map, dtype=np.int32)
    region_counter = 0

    for label_id in range(k):
        mask = (label_map == label_id).astype(np.uint8)
        num_regions, component_map = cv2.connectedComponents(mask)

        for region_id in range(1, num_regions):
            region_counter += 1
            region_map[component_map == region_id] = region_counter

    print(f"[OK] Bölge tespiti tamamlandı.")
    print(f"     {k} renk kümesi -> {region_counter} ayrı bölge")

    return region_map


def build_detail_map(image: np.ndarray) -> np.ndarray:
    """Edge yoğunluğuna dayalı detay haritası oluşturur.

    Canny edge detection + Gaussian blur ile her pikselin
    ne kadar "detay bölgesinde" olduğunu 0-1 arasında verir.
    Yüksek değer = çok detay (göz, burun, yele geçişleri).

    Args:
        image: RGB formatında numpy dizisi (H, W, 3).

    Returns:
        detail_map: (H, W) float32, 0.0-1.0 arasında normalize.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Edge'leri bulanıklaştırarak "detay yoğunluğu" haritası oluştur
    detail_map = cv2.GaussianBlur(edges.astype(np.float32), (31, 31), 0)

    # 0-1 arasına normalize et
    max_val = detail_map.max()
    if max_val > 0:
        detail_map /= max_val

    print(f"[OK] Detay haritası oluşturuldu.")
    print(f"     Detay yoğunluğu: min={detail_map.min():.2f}, max={detail_map.max():.2f}")

    return detail_map


def _lab_distance(centers: np.ndarray, label_a: int, label_b: int) -> float:
    """İki renk kümesi arasındaki LAB mesafesini hesaplar."""
    rgb_a = np.uint8([[centers[label_a].astype(int)]])
    rgb_b = np.uint8([[centers[label_b].astype(int)]])

    lab_a = cv2.cvtColor(rgb_a, cv2.COLOR_RGB2LAB)[0][0].astype(np.float32)
    lab_b = cv2.cvtColor(rgb_b, cv2.COLOR_RGB2LAB)[0][0].astype(np.float32)

    return float(np.linalg.norm(lab_a - lab_b))


def remove_small_regions(
    region_map: np.ndarray,
    label_map: np.ndarray,
    centers: np.ndarray,
    detail_map: np.ndarray,
    min_area_detail: int,
    min_area_background: int,
    contrast_threshold: float = 40.0,
    detail_threshold: float = 0.3,
) -> tuple:
    """Çift eşik sistemiyle küçük bölgeleri temizler.

    Detay yoğunluğu yüksek bölgelerde düşük threshold,
    arka plan bölgelerinde yüksek threshold uygulanır.
    Ek olarak: çok koyu + küçük alanlar (göz bebekleri) korunur.

    Args:
        region_map: Benzersiz bölge ID'leri (H, W).
        label_map: Renk küme etiketleri (H, W).
        centers: Küme merkezleri (K, 3) - RGB.
        detail_map: Detay yoğunluğu haritası (H, W), 0-1.
        min_area_detail: Detay bölgelerinde minimum alan.
        min_area_background: Arka plan bölgelerinde minimum alan.
        contrast_threshold: LAB mesafesi bu değerin üstündeyse bölge korunur.
        detail_threshold: Bu yoğunluğun üstü "detay bölgesi" sayılır.

    Returns:
        cleaned_region_map, cleaned_label_map
    """
    cleaned_region = region_map.copy()
    cleaned_label = label_map.copy()

    unique_regions = np.unique(cleaned_region)
    removed_count = 0
    preserved_contrast = 0
    preserved_dark = 0

    for region_id in unique_regions:
        if region_id == 0:
            continue

        region_mask = (cleaned_region == region_id)
        area = np.sum(region_mask)

        # Bölgenin detay yoğunluğunu hesapla
        region_detail = detail_map[region_mask].mean()
        is_detail_zone = region_detail > detail_threshold

        # Çift eşik: detay bölgesinde düşük, arka planda yüksek
        min_area = min_area_detail if is_detail_zone else min_area_background

        if area >= min_area:
            continue

        # Koyu detay koruması: çok koyu + küçük = göz bebekleri vb.
        region_label = int(cleaned_label[region_mask][0])
        rgb = centers[region_label]
        mean_intensity = rgb.mean()

        if mean_intensity < 50 and area > 20:
            preserved_dark += 1
            continue

        # Komşu bul
        dilated = cv2.dilate(
            region_mask.astype(np.uint8),
            np.ones((3, 3), np.uint8),
            iterations=1,
        )
        neighbor_mask = (dilated == 1) & (~region_mask)

        if np.sum(neighbor_mask) == 0:
            continue

        neighbor_ids = cleaned_region[neighbor_mask]
        neighbor_ids = neighbor_ids[neighbor_ids != 0]

        if len(neighbor_ids) == 0:
            continue

        dominant_neighbor = np.bincount(neighbor_ids).argmax()

        # Kontrast kontrolü
        neighbor_label = int(cleaned_label[cleaned_region == dominant_neighbor][0])
        distance = _lab_distance(centers, region_label, neighbor_label)

        if distance > contrast_threshold:
            preserved_contrast += 1
            continue

        # Küçük bölgeyi komşuya kat
        cleaned_region[region_mask] = dominant_neighbor
        cleaned_label[region_mask] = neighbor_label
        removed_count += 1

    remaining = len(np.unique(cleaned_region)) - 1
    print(f"[OK] Gürültü temizleme tamamlandı.")
    print(f"     Silinen: {removed_count}")
    print(f"     Korunan (kontrast): {preserved_contrast}")
    print(f"     Korunan (koyu detay): {preserved_dark}")
    print(f"     Kalan bölge sayısı: {remaining}")

    return cleaned_region, cleaned_label
