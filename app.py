from flask import Flask, render_template, request, redirect, url_for, send_file, Response, flash
import io
from datetime import datetime
import os

# Hapus import cv2 dan numpy dari sini jika semua logika OpenCV sudah di utils
# import cv2
# import numpy as np
# from PIL import Image # PIL mungkin masih berguna di app.py untuk info awal

# Import fungsi helper dari utils_opencv.py
from utils_opencv import proses_foto_profil, proses_watermark, generate_frames_sketch_logic

app = Flask(__name__)
# Ganti dengan kunci acak yang kuat
app.secret_key = "kunci_rahasia_super_aman_sekali_lagi"

# --- Rute Aplikasi (Logika OpenCV dipanggil dari utils_opencv) ---


@app.context_processor
def inject_now():
    return {'now': datetime.now()}


@app.route('/')
def index():
    return render_template('index.html', title="Beranda",)


@app.route('/foto-profil', methods=['GET', 'POST'])
def foto_profil():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("Tidak ada file yang diupload.", "danger")
            return redirect(url_for('foto_profil'))

        file = request.files['file']
        if file.filename == '':
            flash("Tidak ada file yang dipilih.", "danger")
            return redirect(url_for('foto_profil'))

        # and allowed_file(file.filename) -> tambahkan validasi ekstensi jika perlu
        if file:
            try:
                img_bytes = file.read()

                target_width = int(request.form.get('target_width', 500))
                target_height = int(request.form.get('target_height', 500))
                grayscale = 'grayscale' in request.form
                blur = 'blur' in request.form
                blur_kernel = int(request.form.get('blur_kernel', 5))

                # Panggil fungsi dari utils_opencv.py
                gambar_hasil_cv, error_msg = proses_foto_profil(
                    img_bytes,
                    target_size=(target_width, target_height),
                    grayscale=grayscale,
                    blur=blur,
                    blur_kernel_size=blur_kernel
                )

                if error_msg:
                    flash(f"Error pemrosesan: {error_msg}", "danger")
                    return redirect(url_for('foto_profil'))

                if gambar_hasil_cv is not None:
                    # Re-import cv2 di sini hanya untuk imencode jika belum diimpor global
                    import cv2
                    is_success, buffer = cv2.imencode(".png", gambar_hasil_cv)
                    if is_success:
                        img_io = io.BytesIO(buffer)
                        return send_file(img_io, mimetype='image/png', as_attachment=True, download_name='fotoprofil_jadi.png')
                    else:
                        flash("Gagal mengencode gambar hasil.", "danger")
                else:
                    # error_msg seharusnya sudah menangani ini, tapi sebagai fallback
                    flash("Gagal memproses gambar.", "danger")

            except Exception as e:
                flash(f"Terjadi kesalahan tidak terduga: {str(e)}", "danger")

            # Redirect setelah POST untuk menghindari resubmit
            return redirect(url_for('foto_profil'))

    return render_template('foto_profil.html', title="Foto Profil")


@app.route('/watermark', methods=['GET', 'POST'])
def watermark():
    if request.method == 'POST':
        if 'main_image' not in request.files or 'watermark_image' not in request.files:
            flash("Harap upload gambar utama dan gambar watermark.", "danger")
            return redirect(url_for('watermark'))

        main_file = request.files['main_image']
        watermark_file = request.files['watermark_image']

        if main_file.filename == '' or watermark_file.filename == '':
            flash("Salah satu atau kedua file belum dipilih.", "danger")
            return redirect(url_for('watermark'))

        if main_file and watermark_file:
            try:
                main_img_bytes = main_file.read()
                watermark_img_bytes = watermark_file.read()

                position = request.form.get('position', 'bottom-right')
                scale = int(request.form.get('scale', 20))
                opacity_form = float(request.form.get('opacity', 0.7))

                # Panggil fungsi dari utils_opencv.py
                gambar_hasil_cv, error_msg = proses_watermark(
                    main_img_bytes,
                    watermark_img_bytes,
                    position=position,
                    scale_percent=scale,
                    opacity=opacity_form
                )

                if error_msg:
                    flash(f"Error pemrosesan watermark: {error_msg}", "danger")
                    return redirect(url_for('watermark'))

                if gambar_hasil_cv is not None:
                    import cv2  # Hanya untuk imencode
                    is_success, buffer = cv2.imencode(".png", gambar_hasil_cv)
                    if is_success:
                        img_io = io.BytesIO(buffer)
                        return send_file(img_io, mimetype='image/png', as_attachment=True, download_name='gambar_watermarked.png')
                    else:
                        flash("Gagal mengencode gambar hasil watermark.", "danger")
                else:
                    flash("Gagal memproses gambar watermark.", "danger")

            except Exception as e:
                flash(
                    f"Terjadi kesalahan tidak terduga saat watermark: {str(e)}", "danger")

            return redirect(url_for('watermark'))

    return render_template('watermark.html', title="Watermark")


@app.route('/sketch-webcam')
def sketch_webcam():
    return render_template('sketch_webcam.html', title="Sketch dari Webcam")


@app.route('/video_feed_sketch')
def video_feed_sketch():
    # Panggil generator dari utils_opencv.py
    return Response(generate_frames_sketch_logic(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'true').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0',
            port=int(os.getenv('PORT', 5000)))
