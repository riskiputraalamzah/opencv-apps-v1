# aplikasi_opencv_flask/utils_opencv.py
import cv2
import numpy as np
import os
from datetime import datetime

# --- FUNGSI PROSES_FOTO_PROFIL DAN PROSES_WATERMARK ANDA ---
# Pastikan fungsi-fungsi ini ada di sini jika Anda menggunakannya dari app.py


def proses_foto_profil(img_bytes, target_size=(500, 500), grayscale=False, blur=False, blur_kernel_size=5):
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_original is None:
            return None, "Gagal membaca format gambar. Pastikan file adalah JPG, PNG, dll."
        tinggi, lebar, channels = img_original.shape
        sisi_persegi = min(tinggi, lebar)
        y_mulai = (tinggi - sisi_persegi) // 2
        x_mulai = (lebar - sisi_persegi) // 2
        img_cropped = img_original[y_mulai: y_mulai +
                                   sisi_persegi, x_mulai: x_mulai + sisi_persegi]
        img_resized = cv2.resize(
            img_cropped, target_size, interpolation=cv2.INTER_AREA)
        gambar_diproses = img_resized.copy()
        if grayscale:
            gambar_diproses = cv2.cvtColor(gambar_diproses, cv2.COLOR_BGR2GRAY)
            # Jika di-grayscale, pastikan shape sesuai untuk blur jika blur juga aktif
            if blur and len(gambar_diproses.shape) == 2:  # Sekarang hanya 2D
                # Blur untuk grayscale
                kernel_val = int(blur_kernel_size)
                if kernel_val % 2 == 0:
                    kernel_val += 1
                kernel = (kernel_val, kernel_val)
                gambar_diproses = cv2.GaussianBlur(gambar_diproses, kernel, 0)
        elif blur:  # Jika tidak grayscale tapi blur aktif
            kernel_val = int(blur_kernel_size)
            if kernel_val % 2 == 0:
                kernel_val += 1
            kernel = (kernel_val, kernel_val)
            gambar_diproses = cv2.GaussianBlur(gambar_diproses, kernel, 0)
        return gambar_diproses, None
    except Exception as e:
        print(f"Error di proses_foto_profil: {e}")
        return None, f"Terjadi kesalahan internal saat memproses gambar: {str(e)}"


def proses_watermark(main_img_bytes, watermark_img_bytes, position="bottom-right", scale_percent=20, opacity=0.7):
    try:
        nparr_main = np.frombuffer(main_img_bytes, np.uint8)
        img_main = cv2.imdecode(nparr_main, cv2.IMREAD_COLOR)
        nparr_wm = np.frombuffer(watermark_img_bytes, np.uint8)
        img_watermark_original = cv2.imdecode(nparr_wm, cv2.IMREAD_UNCHANGED)

        if img_main is None:
            return None, "Gagal membaca gambar utama."
        if img_watermark_original is None:
            return None, "Gagal membaca gambar watermark."

        wm_h_orig, wm_w_orig = img_watermark_original.shape[:2]
        main_h, main_w = img_main.shape[:2]
        new_wm_w = int(main_w * (scale_percent / 100.0))
        new_wm_h = int(wm_h_orig * (new_wm_w / wm_w_orig)
                       ) if wm_w_orig > 0 else 0

        if new_wm_w <= 0 or new_wm_h <= 0:
            return None, "Ukuran watermark setelah skala tidak valid."
        img_watermark_resized = cv2.resize(
            img_watermark_original, (new_wm_w, new_wm_h), interpolation=cv2.INTER_AREA)

        has_alpha = img_watermark_resized.shape[2] == 4
        if has_alpha:
            b, g, r, alpha_channel = cv2.split(img_watermark_resized)
            watermark_rgb = cv2.merge((b, g, r))
            mask = alpha_channel
        else:
            watermark_rgb = img_watermark_resized
            gray_wm = cv2.cvtColor(watermark_rgb, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_wm, 5, 255, cv2.THRESH_BINARY)

        h_wm, w_wm = watermark_rgb.shape[:2]
        margin = 10

        pos_map = {
            "bottom-right": (main_h - h_wm - margin, main_h - margin, main_w - w_wm - margin, main_w - margin),
            "top-left": (margin, margin + h_wm, margin, margin + w_wm),
            "top-right": (margin, margin + h_wm, main_w - w_wm - margin, main_w - margin),
            "bottom-left": (main_h - h_wm - margin, main_h - margin, margin, margin + w_wm),
            "center": ((main_h - h_wm) // 2, (main_h + h_wm) // 2, (main_w - w_wm) // 2, (main_w + w_wm) // 2)
        }
        y1, y2, x1, x2 = pos_map.get(position, pos_map["bottom-right"])
        y1, y2, x1, x2 = max(0, y1), min(
            main_h, y2), max(0, x1), min(main_w, x2)

        roi_h, roi_w = y2 - y1, x2 - x1
        # Ambil dimensi saat ini
        current_h_wm, current_w_wm = watermark_rgb.shape[:2]

        if roi_h < current_h_wm or roi_w < current_w_wm:
            # Crop watermark jika area ROI lebih kecil dari watermark yang sudah diresize
            watermark_rgb = watermark_rgb[0:roi_h, 0:roi_w]
            mask = mask[0:roi_h, 0:roi_w]

        if watermark_rgb.shape[0] == 0 or watermark_rgb.shape[1] == 0:
            return None, "Ukuran watermark menjadi tidak valid setelah penyesuaian ROI."

        # Sesuaikan ROI dengan ukuran watermark akhir
        roi = img_main[y1:y1+watermark_rgb.shape[0],
                       x1:x1+watermark_rgb.shape[1]]

        if roi.shape[:2] != watermark_rgb.shape[:2]:
            return None, f"Dimensi ROI ({roi.shape[:2]}) dan watermark ({watermark_rgb.shape[:2]}) tidak cocok final. ROI target: {y1}:{y1+watermark_rgb.shape[0]}, {x1}:{x1+watermark_rgb.shape[1]}"

        img_main_final = img_main.copy()  # Bekerja pada salinan untuk blending yang aman

        if has_alpha:  # Blending dengan alpha channel asli
            alpha_norm = mask.astype(float) / 255.0 * \
                opacity  # Gabungkan opacity form
            if len(alpha_norm.shape) == 2:
                # Ulang alpha untuk 3 channel
                alpha_norm = np.stack([alpha_norm]*3, axis=-1)

            blended_roi = roi * (1 - alpha_norm) + watermark_rgb * alpha_norm
            img_main_final[y1:y1+watermark_rgb.shape[0], x1:x1 +
                           watermark_rgb.shape[1]] = blended_roi.astype(np.uint8)
        else:  # Blending tanpa alpha channel asli, gunakan addWeighted dengan mask
            mask_inv = cv2.bitwise_not(mask)
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            img_fg_raw = cv2.bitwise_and(
                watermark_rgb, watermark_rgb, mask=mask)

            # Terapkan opacity pada foreground
            img_fg_transparent = cv2.addWeighted(
                img_fg_raw, opacity, np.zeros_like(img_fg_raw), 0, 0)

            dst = cv2.add(img_bg, img_fg_transparent)
            img_main_final[y1:y1+watermark_rgb.shape[0],
                           x1:x1+watermark_rgb.shape[1]] = dst

        return img_main_final, None
    except Exception as e:
        print(f"Error di proses_watermark: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Terjadi kesalahan internal saat menambahkan watermark: {str(e)}"
# --- END FUNGSI LAMA ---


# Kita bisa tambahkan save_debug_frames lagi jika perlu
def apply_sketch_to_frame(img_cv):
    if img_cv is None:
        print(
            "[UTILS_CV] apply_sketch_to_frame: img_cv adalah None, mengembalikan placeholder.")
        return np.zeros((100, 100, 3), dtype=np.uint8)

    try:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # 1. Gaussian Blur (untuk menghaluskan gambar dan mengurangi noise)
        #    - Ukuran Kernel (ksize): (width, height), harus ganjil.
        #      Kernel lebih besar -> blur lebih kuat, bisa menghilangkan detail halus tapi juga noise.
        #      Kernel lebih kecil -> blur lebih ringan, menjaga detail tapi mungkin noise lebih banyak.
        #    - SigmaX (standar deviasi pada arah X): Jika 0, dihitung dari ksize.
        #      Nilai sigma lebih besar -> blur lebih kuat.

        # Pilihan A: Blur Ringan (jika gambar asli sudah cukup baik)
        # gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Pilihan B: Blur Sedang (titik awal yang baik)
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Pilihan C: Blur Lebih Kuat (jika banyak noise atau ingin garis lebih tebal/menyatu)
        # gray_blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # (Opsional: Simpan frame yang di-blur untuk debugging jika perlu)
        # cv2.imwrite("debug_blurred.jpg", gray_blurred)

        # 2. Canny Edge Detection
        #    - threshold1: Ambang batas bawah. Piksel dengan gradien di bawah ini pasti bukan tepi.
        #    - threshold2: Ambang batas atas. Piksel dengan gradien di atas ini pasti tepi.
        #    - Piksel dengan gradien antara threshold1 dan threshold2 akan dianggap tepi
        #      jika terhubung ke piksel yang sudah pasti tepi (hysteresis thresholding).
        #
        #    Untuk mendapatkan LEBIH BANYAK TEPI:
        #    - TURUNKAN kedua threshold (terutama threshold1).
        #    - Jaga agar threshold2 lebih tinggi dari threshold1 (rasio umum 1:2 hingga 1:3).
        #
        #    Untuk mendapatkan TEPI LEBIH HALUS/KURANG PUTUS-PUTUS:
        #    - Blur yang tepat sebelum Canny sangat membantu.
        #    - Menurunkan threshold1 bisa membantu menyambungkan segmen garis.
        #    - Parameter L2gradient=True (opsional, sedikit lebih akurat tapi lebih lambat):
        #      edges = cv2.Canny(gray_blurred, threshold1, threshold2, L2gradient=True)

        # Titik awal yang seimbang
        threshold1 = 50
        threshold2 = 150

        # Untuk lebih banyak tepi (mungkin lebih banyak noise juga):
        # threshold1 = 20
        # threshold2 = 70

        # Untuk tepi yang lebih sedikit tapi mungkin lebih bersih:
        # threshold1 = 70
        # threshold2 = 200

        edges = cv2.Canny(gray_blurred, threshold1, threshold2)

        # (Opsional: Simpan frame edges untuk debugging)
        # cv2.imwrite("debug_edges.jpg", edges)

        # 3. Invert warna
        sketch_mono = cv2.bitwise_not(edges)

        # 4. Konversi ke BGR
        sketch_bgr = cv2.cvtColor(sketch_mono, cv2.COLOR_GRAY2BGR)

        return sketch_bgr

    except Exception as e:
        print(f"[UTILS_CV] Error saat menerapkan sketsa: {str(e)}")
        import traceback
        traceback.print_exc()
        return img_cv
