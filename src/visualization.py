# ==================================================
# VISUALIZATION - Görselleştirme
# ==================================================
# Pipeline adımlarını yan yana karşılaştırma görselleri üretir.

import os

import numpy as np
import matplotlib.pyplot as plt


def save_comparison(
    image_left: np.ndarray,
    image_right: np.ndarray,
    title_left: str,
    title_right: str,
    filename: str,
    output_dir: str,
) -> str:
    """İki görüntüyü yan yana koyup tek dosya olarak kaydeder.

    Args:
        image_left: Sol taraftaki görüntü (RGB).
        image_right: Sağ taraftaki görüntü (RGB).
        title_left: Sol başlık.
        title_right: Sağ başlık.
        filename: Kaydedilecek dosya adı.
        output_dir: Çıktı klasörü.

    Returns:
        Kaydedilen dosyanın tam yolu.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.imshow(image_left)
    ax1.set_title(title_left, fontsize=12, fontweight="bold")
    ax1.axis("off")

    ax2.imshow(image_right)
    ax2.set_title(title_right, fontsize=12, fontweight="bold")
    ax2.axis("off")

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150)
    plt.close()

    print(f"[OK] Karşılaştırma kaydedildi: {filepath}")
    return filepath
