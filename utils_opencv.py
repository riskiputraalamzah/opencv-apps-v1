# aplikasi_opencv_flask/utils_opencv.py

import cv2
import numpy as np


def proses_foto_profil(img_bytes, target_size=(500, 500), grayscale=False, blur=False, blur_kernel_size=5):
    """
    Memproses gambar untuk foto profil: crop, resize, opsional grayscale & blur.
    Mengembalikan tuple: (gambar_hasil_cv, error_message_string_atau_None)
    """
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_original is None:
            return None, "Gagal membaca format gambar. Pastikan file adalah JPG, PNG, dll."

        # 1. Cropping menjadi persegi dari tengah
        tinggi, lebar, _ = img_original.shape
        sisi_persegi = min(tinggi, lebar)
        y_mulai = (tinggi - sisi_persegi) // 2
        x_mulai = (lebar - sisi_persegi) // 2
        img_cropped = img_original[y_mulai: y_mulai +
                                   sisi_persegi, x_mulai: x_mulai + sisi_persegi]

        # 2. Resizing
        img_resized = cv2.resize(
            img_cropped, target_size, interpolation=cv2.INTER_AREA)
        gambar_diproses = img_resized.copy()

        # 3. Konversi Warna (Opsional)
        if grayscale:
            gambar_diproses = cv2.cvtColor(gambar_diproses, cv2.COLOR_BGR2GRAY)

        # 4. Image Smoothing (Opsional)
        if blur:
            kernel_val = int(blur_kernel_size)
            # Pastikan kernel ganjil
            if kernel_val % 2 == 0:
                kernel_val += 1
            kernel = (kernel_val, kernel_val)

            # Cek apakah gambar sudah grayscale atau masih BGR sebelum blur
            if grayscale and len(gambar_diproses.shape) == 2:  # Jika sudah grayscale
                gambar_diproses = cv2.GaussianBlur(gambar_diproses, kernel, 0)
            elif not grayscale and len(gambar_diproses.shape) == 3:  # Jika BGR
                gambar_diproses = cv2.GaussianBlur(gambar_diproses, kernel, 0)
            # Jika gambar grayscale tapi blur tidak aktif, atau sebaliknya, tidak masalah,
            # kondisi di atas hanya memastikan blur diterapkan pada format yang benar.

        return gambar_diproses, None  # Tidak ada error
    except Exception as e:
        # Logging error di server bisa membantu
        print(f"Error di proses_foto_profil: {e}")
        return None, f"Terjadi kesalahan internal saat memproses gambar: {str(e)}"


def proses_watermark(main_img_bytes, watermark_img_bytes, position="bottom-right", scale_percent=20, opacity=0.7):
    """
    Menambahkan watermark ke gambar utama.
    Mengembalikan tuple: (gambar_hasil_cv, error_message_string_atau_None)
    """
    try:
        nparr_main = np.frombuffer(main_img_bytes, np.uint8)
        img_main = cv2.imdecode(nparr_main, cv2.IMREAD_COLOR)

        nparr_wm = np.frombuffer(watermark_img_bytes, np.uint8)
        # Baca watermark dengan alpha channel jika ada (IMREAD_UNCHANGED)
        img_watermark_original = cv2.imdecode(nparr_wm, cv2.IMREAD_UNCHANGED)

        if img_main is None:
            return None, "Gagal membaca gambar utama."
        if img_watermark_original is None:
            return None, "Gagal membaca gambar watermark."

        # Resize watermark
        wm_h_orig, wm_w_orig = img_watermark_original.shape[:2]
        main_h, main_w = img_main.shape[:2]

        # Skala watermark berdasarkan persentase lebar gambar utama
        new_wm_w = int(main_w * (scale_percent / 100.0))
        # Jaga aspek rasio watermark
        new_wm_h = int(wm_h_orig * (new_wm_w / wm_w_orig)
                       ) if wm_w_orig > 0 else 0

        if new_wm_w <= 0 or new_wm_h <= 0:
            return None, "Ukuran watermark setelah skala tidak valid."

        img_watermark_resized = cv2.resize(
            img_watermark_original, (new_wm_w, new_wm_h), interpolation=cv2.INTER_AREA)

        # Ambil channel RGB dan alpha (jika ada) dari watermark yang sudah diresize
        if img_watermark_resized.shape[2] == 4:  # Ada alpha channel (BGRA)
            b, g, r, alpha = cv2.split(img_watermark_resized)
            watermark_rgb = cv2.merge((b, g, r))
            # Gunakan alpha channel asli sebagai mask, normalisasikan ke 0-1 jika perlu untuk blending
            # OpenCV biasanya menangani mask 8-bit (0-255) dengan baik
            mask = alpha
        else:  # Tidak ada alpha channel (BGR), buat mask sederhana
            watermark_rgb = img_watermark_resized
            # Buat mask dari area non-hitam di watermark, atau asumsikan semua bagian adalah watermark
            gray_wm = cv2.cvtColor(watermark_rgb, cv2.COLOR_BGR2GRAY)
            # Anggap piksel > 5 adalah bagian dari watermark
            _, mask = cv2.threshold(gray_wm, 5, 255, cv2.THRESH_BINARY)

        # Tentukan posisi Region of Interest (ROI) di gambar utama
        h_wm, w_wm = watermark_rgb.shape[:2]
        margin = 10  # Jarak dari tepi

        # Pastikan watermark tidak lebih besar dari gambar utama setelah margin
        if h_wm + margin > main_h or w_wm + margin > main_w:
            # Jika terlalu besar, mungkin resize watermark lebih lanjut atau berikan error
            # Untuk sekarang, kita crop watermark jika melebihi batas ROI
            pass  # Atau bisa throw error: return None, "Watermark terlalu besar untuk gambar utama."

        if position == "bottom-right":
            y1, y2 = main_h - h_wm - margin, main_h - margin
            x1, x2 = main_w - w_wm - margin, main_w - margin
        elif position == "top-left":
            y1, y2 = margin, margin + h_wm
            x1, x2 = margin, margin + w_wm
        elif position == "top-right":
            y1, y2 = margin, margin + h_wm
            x1, x2 = main_w - w_wm - margin, main_w - margin
        elif position == "bottom-left":
            y1, y2 = main_h - h_wm - margin, main_h - margin
            x1, x2 = margin, margin + w_wm
        elif position == "center":
            y1 = (main_h - h_wm) // 2
            y2 = y1 + h_wm
            x1 = (main_w - w_wm) // 2
            x2 = x1 + w_wm
        else:  # Default ke bottom-right jika posisi tidak dikenal
            y1, y2 = main_h - h_wm - margin, main_h - margin
            x1, x2 = main_w - w_wm - margin, main_w - margin

        # Pastikan koordinat ROI valid (tidak keluar dari batas gambar utama)
        y1 = max(0, y1)
        y2 = min(main_h, y2)
        x1 = max(0, x1)
        x2 = min(main_w, x2)

        # Sesuaikan ukuran watermark_rgb dan mask jika ROI lebih kecil dari watermark
        # Ini penting jika watermarknya lebih besar dari area yang tersedia setelah penentuan posisi
        current_h_wm = y2 - y1
        current_w_wm = x2 - x1

        if current_h_wm < h_wm or current_w_wm < w_wm:
            # Jika area ROI lebih kecil, kita perlu crop watermark_rgb dan mask agar pas
            watermark_rgb = watermark_rgb[0:current_h_wm, 0:current_w_wm]
            mask = mask[0:current_h_wm, 0:current_w_wm]
            # Perbarui dimensi watermark yang akan digunakan
            h_wm, w_wm = watermark_rgb.shape[:2]
            if h_wm == 0 or w_wm == 0:  # Jika setelah crop jadi 0, tidak bisa lanjut
                return None, "Ukuran watermark menjadi tidak valid setelah penyesuaian ROI."

        # Ambil ROI dari gambar utama
        roi = img_main[y1:y2, x1:x2]

        # Jika ROI dan watermark_rgb tidak punya ukuran yang sama, ada masalah
        if roi.shape[:2] != watermark_rgb.shape[:2]:
            # Ini bisa terjadi jika perhitungan ROI atau resize watermark tidak tepat
            # Atau jika watermark (setelah resize) lebih besar dari gambar utama
            # Untuk kasus ini, kita bisa skip atau berikan error
            return None, f"Dimensi ROI ({roi.shape[:2]}) dan watermark ({watermark_rgb.shape[:2]}) tidak cocok."

        # Blending menggunakan mask
        # Buat inverse dari mask untuk area background ROI
        mask_inv = cv2.bitwise_not(mask)

        # Hitamkan area watermark di ROI (bagian yang akan diganti)
        img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        # Ambil hanya bagian watermark yang sesuai dengan mask
        img_fg = cv2.bitwise_and(watermark_rgb, watermark_rgb, mask=mask)

        # Gabungkan background ROI yang sudah dihitamkan dengan foreground watermark
        # Tambahkan opasitas di sini jika perlu
        if opacity < 1.0:
            # Perlu blending alpha manual jika img_fg adalah BGR dan kita ingin opasitas
            # Cara termudah adalah addWeighted, tapi ini mengasumsikan kedua gambar BGR
            # Jika img_fg sudah punya alpha, cara blendingnya berbeda
            # Untuk kesederhanaan, jika ada alpha dari PNG, kita abaikan opacity dari form
            # Jika tidak ada alpha asli dan opacity < 1.0, kita gunakan addWeighted
            # Tidak ada alpha channel asli
            if img_watermark_resized.shape[2] != 4:
                blended_fg = cv2.addWeighted(
                    img_fg, opacity, np.zeros_like(img_fg), 1-opacity, 0)
                dst = cv2.add(img_bg, blended_fg)
            else:  # Ada alpha channel asli, gunakan itu
                # Opacity dari form diabaikan jika ada alpha dari file
                dst = cv2.add(img_bg, img_fg)
        else:
            dst = cv2.add(img_bg, img_fg)

        # Tempatkan hasil blending kembali ke gambar utama
        img_main[y1:y2, x1:x2] = dst

        return img_main, None
    except Exception as e:
        print(f"Error di proses_watermark: {e}")
        return None, f"Terjadi kesalahan internal saat menambahkan watermark: {str(e)}"


def generate_frames_sketch_logic():
    """
    Generator function untuk menghasilkan frame sketch dari webcam.
    Akan di-yield oleh endpoint Flask.
    """
    camera = cv2.VideoCapture(0)  # Coba ganti ke 1 jika 0 tidak bekerja
    if not camera.isOpened():
        print("Error: Tidak bisa membuka kamera di utils_opencv.")
        # Membuat frame error statis untuk dikirim jika kamera gagal
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "Kamera Error", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_img)
        frame_bytes = buffer.tobytes()
        while True:  # Loop tak terbatas mengirim frame error
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            # Tunggu 1 detik sebelum mengirim lagi (agar tidak membanjiri client)
            cv2.waitKey(1000)

    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                # Jika gagal membaca frame, bisa kirim frame error atau skip
                print("Gagal membaca frame dari kamera di utils_opencv.")
                # Untuk konsistensi, kita bisa yield frame error juga, atau cukup continue
                error_img_read = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_img_read, "Frame Read Error", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                ret_err, buffer_err = cv2.imencode('.jpg', error_img_read)
                if ret_err:
                    frame_bytes_err = buffer_err.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes_err + b'\r\n')
                cv2.waitKey(100)  # Tunggu sebentar
                continue

            # 1. Konversi ke Grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 2. Smoothing
            gray_blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            # 3. Edge Detection (Canny)
            edges = cv2.Canny(gray_blurred, 50, 150)
            # 4. Invert warna (agar garis hitam, latar putih)
            sketch = cv2.bitwise_not(edges)

            # Encode frame ke JPEG
            ret_encode, buffer_encode = cv2.imencode('.jpg', sketch)
            if not ret_encode:
                continue  # Skip frame jika encoding gagal

            frame_bytes_encode = buffer_encode.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes_encode + b'\r\n')
    finally:
        # Pastikan kamera di-release ketika generator selesai atau ada error
        print("Melepaskan kamera di utils_opencv.")
        camera.release()
