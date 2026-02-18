# ==================================================
# REGION DETECTION - Bölge Tespiti
# ==================================================
# label_map'te aynı renk etiketi farklı konumlarda olabilir.
# Örn: "Mavi" hem gökyüzünde hem gölde varsa bunlar ayrı bölgeler.
# Connected component analysis ile her bağlı alanı ayırır,
# küçük gürültü bölgelerini komşu renge katar.

import cv2
import numpy as np


def detect_regions(label_map: np.ndarray, k: int) -> np.ndarray:
    """Her renk kümesini ayrı bağlı bölgelere ayırır.

    Aynı renk etiketine sahip ama fiziksel olarak ayrı
    pikselleri farklı bölge ID'leri ile işaretler.

    Args:
        label_map: 2D etiket haritası (H, W) - her değer bir küme ID'si.
        k: Toplam küme sayısı.

    Returns:
        region_map: 2D harita (H, W) - her piksel benzersiz bölge ID'si taşır.
    """
    region_map = np.zeros_like(label_map, dtype=np.int32)
    region_counter = 0

    for label_id in range(k):
        mask = (label_map == label_id).astype(np.uint8)
        num_regions, component_map = cv2.connectedComponents(mask)

        # 0 = arka plan, 1+ = bağlı bölgeler
        for region_id in range(1, num_regions):
            region_counter += 1
            region_map[component_map == region_id] = region_counter

    print(f"[OK] Bölge tespiti tamamlandı.")
    print(f"     {k} renk kümesi -> {region_counter} ayrı bölge")

    return region_map


def _lab_distance(centers: np.ndarray, label_a: int, label_b: int) -> float:
    """İki renk kümesi arasındaki LAB mesafesini hesaplar.

    LAB uzayında Euclidean mesafe, insan gözünün algıladığı
    renk farkına yakındır. Yüksek mesafe = belirgin kontrast.

    Args:
        centers: Küme merkezleri (K, 3) - RGB.
        label_a: Birinci küme ID'si.
        label_b: İkinci küme ID'si.

    Returns:
        İki renk arasındaki LAB mesafesi (float).
    """
    rgb_a = np.uint8([[centers[label_a].astype(int)]])
    rgb_b = np.uint8([[centers[label_b].astype(int)]])

    lab_a = cv2.cvtColor(rgb_a, cv2.COLOR_RGB2LAB)[0][0].astype(np.float32)
    lab_b = cv2.cvtColor(rgb_b, cv2.COLOR_RGB2LAB)[0][0].astype(np.float32)

    return float(np.linalg.norm(lab_a - lab_b))


def remove_small_regions(
    region_map: np.ndarray,
    label_map: np.ndarray,
    min_area: int,
    centers: np.ndarray,
    contrast_threshold: float = 30.0,
) -> tuple:
    """Küçük bölgeleri en büyük komşu bölgeye katar.

    Kontrastı yüksek küçük bölgeler korunur (göz, burun gibi).
    Bir bölge küçük olsa bile, komşusuyla renk farkı
    contrast_threshold'u aşıyorsa silinmez.

    Args:
        region_map: Benzersiz bölge ID'leri (H, W).
        label_map: Renk küme etiketleri (H, W).
        min_area: Bu pikselden küçük bölgeler birleştirilir.
        centers: Küme merkezleri (K, 3) - RGB.
        contrast_threshold: LAB mesafesi bu değerin üstündeyse
                            küçük bölge korunur (varsayılan: 30.0).

    Returns:
        cleaned_region_map: Temizlenmiş bölge haritası (H, W).
        cleaned_label_map: Güncellenmiş renk etiket haritası (H, W).
    """
    cleaned_region = region_map.copy()
    cleaned_label = label_map.copy()

    unique_regions = np.unique(cleaned_region)
    removed_count = 0
    preserved_count = 0

    for region_id in unique_regions:
        if region_id == 0:
            continue

        region_mask = (cleaned_region == region_id)
        area = np.sum(region_mask)

        if area >= min_area:
            continue

        # Bölgenin 1 piksel genişletilmiş komşuluğunu bul
        dilated = cv2.dilate(
            region_mask.astype(np.uint8),
            np.ones((3, 3), np.uint8),
            iterations=1,
        )
        neighbor_mask = (dilated == 1) & (~region_mask)

        if np.sum(neighbor_mask) == 0:
            continue

        # Komşular arasında en sık görülen bölge ID'sini bul
        neighbor_ids = cleaned_region[neighbor_mask]
        neighbor_ids = neighbor_ids[neighbor_ids != 0]

        if len(neighbor_ids) == 0:
            continue

        dominant_neighbor = np.bincount(neighbor_ids).argmax()

        # Kontrast kontrolü: bölgenin rengi ile komşunun rengi arasındaki fark
        region_label = int(cleaned_label[region_mask][0])
        neighbor_label = int(cleaned_label[cleaned_region == dominant_neighbor][0])

        distance = _lab_distance(centers, region_label, neighbor_label)

        if distance > contrast_threshold:
            preserved_count += 1
            continue

        # Küçük bölgeyi komşuya kat
        cleaned_region[region_mask] = dominant_neighbor
        cleaned_label[region_mask] = neighbor_label
        removed_count += 1

    remaining = len(np.unique(cleaned_region)) - 1  # 0 hariç
    print(f"[OK] Gürültü temizleme tamamlandı.")
    print(f"     Silinen küçük bölge: {removed_count}")
    print(f"     Kontrast nedeniyle korunan: {preserved_count}")
    print(f"     Kalan bölge sayısı: {remaining}")

    return cleaned_region, cleaned_label
