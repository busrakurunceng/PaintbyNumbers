# ==================================================
# CLUSTERING - K-Means Renk Kümeleme
# ==================================================

from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans


def apply_kmeans(
    pixels: np.ndarray, k: int, random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Pikselleri K-Means algoritması ile K gruba ayırır.

    Her piksel, en yakın küme merkezine atanır.
    Sonuçta 16 milyon renk yerine sadece K adet renk kalır.

    Args:
        pixels: (N, 3) boyutunda float32 piksel matrisi.
        k: Küme sayısı (kaç farklı renk istiyoruz).
        random_state: Tekrarlanabilirlik için seed değeri.

    Returns:
        labels: Her pikselin ait olduğu küme indeksi (N,).
        centers: Küme merkezleri, yani K adet RGB değeri (K, 3).
    """
    print(f"[..] K-Means başlatılıyor (K={k})...")

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    kmeans.fit(pixels)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    print(f"[OK] K-Means tamamlandı.")
    print(f"     Küme sayısı: {k}")
    print(f"     Etiketlenen piksel: {len(labels):,}")

    return labels, centers


def get_dominant_colors(
    centers: np.ndarray, labels: np.ndarray
) -> List[dict]:
    """Her kümenin RGB değerini ve görüntüdeki yüzdesini hesaplar.

    Dominant renkler yüzdeye göre büyükten küçüğe sıralanır.
    Bu sayede hangi rengin görüntüde ne kadar yer kapladığını görürüz.

    Args:
        centers: Küme merkezleri (K, 3).
        labels: Her pikselin küme etiketi (N,).

    Returns:
        Her renk için sözlük listesi:
        [{"color_id": 0, "rgb": [R, G, B], "percentage": 35.2}, ...]
    """
    total_pixels = len(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)

    dominant_colors = []
    for label, count in zip(unique_labels, counts):
        rgb = centers[label].astype(int).tolist()
        percentage = round((count / total_pixels) * 100, 2)

        dominant_colors.append({
            "color_id": int(label),
            "rgb": rgb,
            "percentage": percentage,
        })

    # Yüzdeye göre büyükten küçüğe sırala
    dominant_colors.sort(key=lambda x: x["percentage"], reverse=True)

    print("=" * 45)
    print("DOMİNANT RENKLER")
    print("=" * 45)
    for i, color in enumerate(dominant_colors):
        r, g, b = color["rgb"]
        print(f"  #{i + 1}  RGB({r:3d}, {g:3d}, {b:3d})  ->  %{color['percentage']}")
    print("=" * 45)

    return dominant_colors
