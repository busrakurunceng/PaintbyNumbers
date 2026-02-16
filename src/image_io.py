# ==================================================
# IMAGE I/O - Görüntü Yükleme ve Kaydetme
# ==================================================
# vision-color-pipeline projesinden alınmıştır.
# BGR↔RGB dönüşümü ile OpenCV ve matplotlib uyumu sağlanır.

import os

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Görüntüyü diskten yükler ve RGB formatında döndürür.

    OpenCV varsayılan olarak BGR formatında okur.
    Bu fonksiyon otomatik olarak RGB'ye çevirir,
    böylece matplotlib ile uyumlu hale gelir.

    Args:
        path: Görüntü dosyasının yolu.

    Returns:
        RGB formatında numpy dizisi (H, W, 3).

    Raises:
        FileNotFoundError: Dosya bulunamazsa.
        ValueError: Görüntü okunamazsa.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Görüntü bulunamadı: {path}")

    image = cv2.imread(path)

    if image is None:
        raise ValueError(f"Görüntü okunamadı: {path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"[OK] Görüntü yüklendi: {path}")
    print(f"     Boyut: {image_rgb.shape[1]}x{image_rgb.shape[0]} piksel")

    return image_rgb


def save_image(image: np.ndarray, filename: str, output_dir: str) -> str:
    """Görüntüyü belirtilen klasöre kaydeder.

    RGB formatındaki görüntüyü OpenCV'nin beklediği BGR'ye
    çevirip diske yazar.

    Args:
        image: RGB formatında numpy dizisi.
        filename: Kaydedilecek dosya adı (örn: "result.png").
        output_dir: Çıktı klasörünün yolu.

    Returns:
        Kaydedilen dosyanın tam yolu.
    """
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, filename)

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(full_path, image_bgr)

    print(f"[OK] Görüntü kaydedildi: {full_path}")
    return full_path
