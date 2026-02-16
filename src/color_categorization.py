# ==================================================
# COLOR CATEGORIZATION - Renk Isimlendirme
# ==================================================
# RGB degerlerini insan tarafindan anlasilir
# renk isimlerine donusturur.
#
# Yontem: Bilinen renk sozlugundeki en yakin rengi
# LAB uzayinda Euclidean mesafe ile bulur.
# LAB uzayi insan gozu algisindan tasarlandigindan
# mesafe hesabi perceputal (algisal) olarak dogrudur.

from typing import List

import cv2
import numpy as np


# Bilinen renk sozlugu: isim -> (R, G, B)
# Temel renkler + ara tonlar + acik/koyu varyantlar
COLOR_DICTIONARY = {
    # Achromatic
    "Siyah":        (0, 0, 0),
    "Koyu Gri":     (64, 64, 64),
    "Gri":          (128, 128, 128),
    "Acik Gri":     (192, 192, 192),
    "Beyaz":        (255, 255, 255),

    # Kirmizi ailesi
    "Koyu Kirmizi": (139, 0, 0),
    "Kirmizi":      (255, 0, 0),
    "Acik Kirmizi": (255, 102, 102),

    # Turuncu ailesi
    "Koyu Turuncu": (200, 100, 0),
    "Turuncu":      (255, 140, 0),
    "Acik Turuncu": (255, 179, 102),

    # Sari ailesi
    "Koyu Sari":    (204, 170, 0),
    "Sari":         (255, 220, 0),
    "Acik Sari":    (255, 255, 153),

    # Yesil ailesi
    "Koyu Yesil":   (0, 100, 0),
    "Yesil":        (0, 180, 0),
    "Acik Yesil":   (144, 238, 144),

    # Mavi ailesi
    "Koyu Mavi":    (0, 0, 139),
    "Mavi":         (0, 100, 255),
    "Acik Mavi":    (135, 206, 250),
    "Lacivert":     (0, 0, 80),

    # Mor ailesi
    "Koyu Mor":     (75, 0, 130),
    "Mor":          (148, 0, 211),
    "Lila":         (200, 162, 200),

    # Pembe ailesi
    "Pembe":        (255, 105, 180),
    "Acik Pembe":   (255, 182, 193),

    # Sicak ara tonlar
    "Kahverengi":   (139, 69, 19),
    "Koyu Kahve":   (80, 40, 10),
    "Bej":          (210, 180, 140),
    "Krem":         (255, 253, 208),
    "Somon":        (250, 128, 114),
    "Mercan":       (255, 127, 80),
    "Bordo":        (128, 0, 0),
    "Altin":        (255, 193, 37),
    "Amber":        (255, 191, 0),

    # Soguk ara tonlar
    "Camgobegi":    (0, 255, 255),
    "Turkuaz":      (0, 206, 209),
    "Leylak":       (150, 120, 182),
    "Arduvaz":      (112, 128, 144),
    "Zeytin":       (128, 128, 0),
}


def _rgb_to_lab(r: int, g: int, b: int) -> np.ndarray:
    """Tek bir RGB degerini LAB uzayina cevirir.

    LAB uzayi insan gozu algisina gore tasarlanmistir.
    Bu uzaydaki Euclidean mesafe, iki renk arasindaki
    algisal farki (Delta-E) yaklasik olarak verir.

    Args:
        r: Kirmizi kanal (0-255).
        g: Yesil kanal (0-255).
        b: Mavi kanal (0-255).

    Returns:
        LAB degerleri (3,) numpy dizisi.
    """
    pixel = np.uint8([[[r, g, b]]])
    lab_pixel = cv2.cvtColor(pixel, cv2.COLOR_RGB2LAB)
    return lab_pixel[0][0].astype(np.float32)


def classify_color(r: int, g: int, b: int) -> str:
    """Bir RGB degerini en yakin bilinen renk ismine donusturur.

    COLOR_DICTIONARY'deki tum renkleri LAB uzayina cevirir,
    girdi rengini de LAB'a cevirir ve Euclidean mesafe ile
    en yakin rengi bulur.

    Args:
        r: Kirmizi kanal (0-255).
        g: Yesil kanal (0-255).
        b: Mavi kanal (0-255).

    Returns:
        En yakin bilinen rengin ismi.
    """
    input_lab = _rgb_to_lab(r, g, b)

    min_distance = float("inf")
    closest_name = "Bilinmeyen"

    for name, (cr, cg, cb) in COLOR_DICTIONARY.items():
        candidate_lab = _rgb_to_lab(cr, cg, cb)
        distance = np.linalg.norm(input_lab - candidate_lab)

        if distance < min_distance:
            min_distance = distance
            closest_name = name

    return closest_name


def categorize_centers(centers: np.ndarray) -> List[dict]:
    """Tum kume merkezlerini isimlendirip listeler.

    Her kume merkezi icin RGB ve LAB tabanli en yakin
    renk ismini bir sozluk olarak dondurur.

    Args:
        centers: Kume merkezleri (K, 3) - RGB degerleri.

    Returns:
        Her renk icin sozluk listesi:
        [{"color_id": 0, "rgb": (R,G,B), "name": "..."}, ...]
    """
    results = []

    print("=" * 50)
    print("RENK KATEGORIZASYONU (LAB mesafe)")
    print("=" * 50)

    for i, center in enumerate(centers):
        r, g, b = int(center[0]), int(center[1]), int(center[2])
        name = classify_color(r, g, b)

        results.append({
            "color_id": i,
            "rgb": (r, g, b),
            "name": name,
        })

        print(f"  Renk {i}: RGB({r:3d}, {g:3d}, {b:3d}) | {name}")

    print("=" * 50)

    return results
