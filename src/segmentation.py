# ==================================================
# SEGMENTATION - Görüntü Bölütleme
# ==================================================

import numpy as np


def segment_image(
    labels: np.ndarray, centers: np.ndarray, original_shape: tuple
) -> np.ndarray:
    """K-Means sonuçlarını kullanarak segmented görüntü oluşturur.

    Her pikseli, ait olduğu kümenin merkez rengine boyar.
    Böylece orijinal görüntü sadece K adet renkten oluşan
    posterize bir görüntüye dönüşür.

    Args:
        labels: Her pikselin küme etiketi (N,).
        centers: Küme merkezleri (K, 3) - RGB değerleri.
        original_shape: Orijinal görüntü boyutu (H, W, 3).

    Returns:
        Segmented görüntü (H, W, 3) - uint8.
    """
    # Her pikseli kendi küme merkezinin rengiyle doldur
    segmented_flat = centers[labels]

    # Düz diziyi orijinal görüntü boyutuna geri çevir
    segmented = segmented_flat.reshape(original_shape)

    # float -> uint8 dönüşümü (görüntü formatı)
    segmented = segmented.astype(np.uint8)

    print(f"[OK] Segmentasyon tamamlandi.")
    print(f"     Boyut: {segmented.shape}")
    print(f"     Benzersiz renk sayisi: {len(centers)}")

    return segmented


def create_label_map(labels: np.ndarray, original_shape: tuple) -> np.ndarray:
    """Her pikselin küme etiketini gösteren 2D harita oluşturur.

    Bu harita, hangi pikselin hangi bölgeye ait olduğunu
    gösterir. İleride numaralandırma için kullanılacak.

    Args:
        labels: Her pikselin küme etiketi (N,).
        original_shape: Orijinal görüntü boyutu (H, W, 3).

    Returns:
        2D etiket haritası (H, W) - her değer bir küme ID'si.
    """
    height, width = original_shape[0], original_shape[1]

    label_map = labels.reshape(height, width)

    print(f"[OK] Etiket haritasi olusturuldu: {label_map.shape}")
    print(f"     Etiket araligi: [{label_map.min()}, {label_map.max()}]")

    return label_map
