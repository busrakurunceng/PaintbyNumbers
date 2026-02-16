# ==================================================
# PIXEL ANALYSIS - Piksel Matris Analizi
# ==================================================
# vision-color-pipeline projesinden alınmıştır.
# Görüntüyü K-Means'in beklediği 2D formata dönüştürür.

import numpy as np


def get_image_info(image: np.ndarray) -> dict:
    """Görüntünün temel bilgilerini döndürür.

    Args:
        image: RGB formatında numpy dizisi (H, W, 3).

    Returns:
        Görüntü bilgilerini içeren sözlük.
    """
    height, width, channels = image.shape

    info = {
        "height": height,
        "width": width,
        "channels": channels,
        "total_pixels": height * width,
        "dtype": str(image.dtype),
        "min_value": int(image.min()),
        "max_value": int(image.max()),
    }

    print(f"     Boyut: {width}x{height}, Toplam piksel: {info['total_pixels']:,}")

    return info


def extract_pixels(image: np.ndarray) -> np.ndarray:
    """Görüntüyü 2D piksel matrisine dönüştürür.

    (H, W, 3) boyutundaki görüntüyü (H*W, 3) boyutuna
    düzleştirir. Bu format K-Means gibi algoritmaların
    beklediği girdi formatıdır.

    Args:
        image: RGB formatında numpy dizisi (H, W, 3).

    Returns:
        (H*W, 3) boyutunda 2D numpy dizisi (float32).
    """
    height, width, channels = image.shape

    pixels = image.reshape(-1, channels)
    pixels = pixels.astype(np.float32)

    print(f"[OK] Piksel matrisi: ({height}x{width}) -> ({pixels.shape[0]}, {pixels.shape[1]})")

    return pixels
