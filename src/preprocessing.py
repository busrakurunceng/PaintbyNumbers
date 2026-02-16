# ==================================================
# PREPROCESSING - Görüntü Sadeleştirme
# ==================================================
# K-Means'ten ÖNCE görüntüyü yumuşatır.
# Bilateral Filter kenarları koruyarak gürültüyü azaltır.
# Gaussian Blur kenarları da bulanıklaştırır, bu yüzden uygun değildir.

import cv2
import numpy as np


def apply_bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
    iterations: int = 2,
) -> np.ndarray:
    """Bilateral filtre ile görüntüyü yumuşatır.

    Kenarları koruyarak düz bölgelerdeki gürültüyü
    ve mikro detayları temizler. Birden fazla iterasyon
    daha agresif yumuşatma sağlar.

    Args:
        image: RGB formatında numpy dizisi (H, W, 3).
        d: Filtre çapı. Büyük değer = daha geniş komşuluk.
        sigma_color: Renk uzayında sigma.
        sigma_space: Koordinat uzayında sigma.
        iterations: Filtrenin kaç kez uygulanacağı.

    Returns:
        Yumuşatılmış görüntü (H, W, 3) - uint8.
    """
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    filtered = image_bgr.copy()
    for i in range(iterations):
        filtered = cv2.bilateralFilter(filtered, d, sigma_color, sigma_space)

    result = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)

    print(f"[OK] Bilateral filtre uygulandı.")
    print(f"     d={d}, sigma_color={sigma_color}, sigma_space={sigma_space}")
    print(f"     İterasyon: {iterations}")

    return result
