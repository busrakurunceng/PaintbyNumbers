# ==================================================
# RENDERING - Çıktı Üretimi
# ==================================================
# Son kullanıcıya verilecek dosyaları oluşturur:
# 1. Referans resim: Bölgeler kendi renkleriyle dolu + numaralar
# 2. Renk legendı: Hangi numaranın hangi renk olduğunu gösteren tablo

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def render_reference_image(
    cleaned_segmented: np.ndarray,
    contour_list: list,
    centroids: list,
    line_color: tuple = (80, 80, 80),
    line_thickness: int = 1,
    font_color: tuple = (60, 60, 60),
    font_thickness: int = 1,
) -> np.ndarray:
    """Renkli referans resim oluşturur.

    Posterize edilmiş görüntü üzerine kontur çizgileri
    ve numara etiketleri eklenir. Kullanıcı boyarken
    hangi alanın hangi renk olduğunu buradan kontrol eder.

    Args:
        cleaned_segmented: Temizlenmiş posterize görüntü (H, W, 3).
        contour_list: extract_contours() çıktısı.
        centroids: calculate_centroids() çıktısı.
        line_color: Çizgi rengi (R, G, B).
        line_thickness: Çizgi kalınlığı.
        font_color: Numara rengi (R, G, B).
        font_thickness: Numara kalınlığı.

    Returns:
        reference: Renkli zemin + çizgiler + numaralar (H, W, 3).
    """
    reference = cleaned_segmented.copy()

    contours_only = [contour for _, contour in contour_list]
    cv2.drawContours(reference, contours_only, -1, line_color, line_thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for c in centroids:
        text = str(c["color_id"])

        if c["area"] > 10000:
            font_scale = 0.5
        elif c["area"] > 5000:
            font_scale = 0.4
        else:
            font_scale = 0.3

        (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

        if tw > c["bbox_w"] * 0.8 or th > c["bbox_h"] * 0.8:
            continue

        tx = c["cx"] - tw // 2
        ty = c["cy"] + th // 2

        cv2.putText(reference, text, (tx, ty), font, font_scale,
                    font_color, font_thickness, cv2.LINE_AA)

    print(f"[OK] Referans resim oluşturuldu.")
    return reference


def render_color_legend(
    centers: np.ndarray,
    color_names: list,
    output_dir: str,
) -> str:
    """Renk legendı oluşturur ve kaydeder.

    Her renk numarası için RGB kutusu ve isim gösterilir.
    "1 = Koyu Mavi", "2 = Bej" gibi bir tablo.

    Args:
        centers: Küme merkezleri (K, 3).
        color_names: categorize_centers() çıktısı.
        output_dir: Çıktı klasörü.

    Returns:
        Kaydedilen dosyanın tam yolu.
    """
    k = len(centers)
    rows = (k + 3) // 4  # 4 sütunluk grid
    cols = min(k, 4)

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 1.5 * rows))
    fig.suptitle("Renk Legendı - Hangi Numara Hangi Renk?",
                 fontsize=14, fontweight="bold")

    name_map = {c["color_id"]: c["name"] for c in color_names}

    if rows == 1:
        axes = [axes] if cols == 1 else [axes]
    axes_flat = np.array(axes).flatten()

    for i in range(len(axes_flat)):
        ax = axes_flat[i]

        if i < k:
            rgb = centers[i].astype(int)
            name = name_map.get(i, "?")

            color_block = np.array([[rgb]], dtype=np.uint8)
            ax.imshow(color_block, aspect="auto")

            ax.set_title(f"#{i}", fontsize=11, fontweight="bold")
            ax.set_xlabel(f"{name}\n({rgb[0]}, {rgb[1]}, {rgb[2]})", fontsize=8)
        else:
            ax.set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "color_legend.png")
    plt.savefig(filepath, dpi=150)
    plt.close()

    print(f"[OK] Renk legendı kaydedildi: {filepath}")
    return filepath
