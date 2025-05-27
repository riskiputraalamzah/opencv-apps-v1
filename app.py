from flask import Flask, render_template, request, redirect, url_for, send_file, Response, flash, jsonify
import io
from datetime import datetime
import os
import base64
import numpy as np
import cv2

from utils_opencv import (
    proses_foto_profil,
    proses_watermark,
    apply_sketch_to_frame  # Pastikan ini diimpor
)

app = Flask(__name__)
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY", "kunci_rahasia_default_yang_sangat_aman")


@app.context_processor
def inject_now():
    return {'now': datetime.now()}


@app.route('/')
def index():
    return render_template('index.html', title="Beranda")

# --- FOTO PROFIL ---


@app.route('/foto-profil', methods=['GET', 'POST'])
def foto_profil():
    # ... (kode Anda, pastikan sudah benar) ...
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("Tidak ada file yang diupload.", "danger")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash("Tidak ada file yang dipilih.", "danger")
            return redirect(request.url)
        if file:
            try:
                img_bytes = file.read()
                # ... (sisa parameter dari form) ...
                target_width = int(request.form.get('target_width', 500))
                target_height = int(request.form.get('target_height', 500))
                grayscale = 'grayscale' in request.form
                blur = 'blur' in request.form
                blur_kernel = int(request.form.get('blur_kernel', 5))

                gambar_hasil_cv, error_msg = proses_foto_profil(
                    img_bytes,
                    target_size=(target_width, target_height),
                    grayscale=grayscale,
                    blur=blur,
                    blur_kernel_size=blur_kernel
                )
                # ... (penanganan error dan send_file) ...
                if error_msg:
                    flash(f"Error pemrosesan: {error_msg}", "danger")
                elif gambar_hasil_cv is not None:
                    is_success, buffer = cv2.imencode(".png", gambar_hasil_cv)
                    if is_success:
                        img_io = io.BytesIO(buffer)
                        return send_file(img_io, mimetype='image/png', as_attachment=True, download_name='fotoprofil_jadi.png')
                    else:
                        flash("Gagal mengencode gambar hasil.", "danger")
                else:
                    flash("Gagal memproses gambar.", "danger")
            except Exception as e:
                app.logger.error(f"Error foto_profil: {str(e)}")
                flash(f"Terjadi kesalahan: {str(e)}", "danger")
        return redirect(request.url)
    return render_template('foto_profil.html', title="Foto Profil")

# --- WATERMARK ---


@app.route('/watermark', methods=['GET', 'POST'])
def watermark():
    # ... (kode Anda, pastikan sudah benar) ...
    if request.method == 'POST':
        if 'main_image' not in request.files or 'watermark_image' not in request.files:
            flash("Harap upload gambar utama dan gambar watermark.", "danger")
            return redirect(request.url)
        main_file = request.files['main_image']
        watermark_file = request.files['watermark_image']
        if main_file.filename == '' or watermark_file.filename == '':
            flash("Salah satu atau kedua file belum dipilih.", "danger")
            return redirect(request.url)
        if main_file and watermark_file:
            try:
                main_img_bytes = main_file.read()
                watermark_img_bytes = watermark_file.read()
                # ... (sisa parameter dari form) ...
                position = request.form.get('position', 'bottom-right')
                scale = int(request.form.get('scale', 20))
                opacity_form = float(request.form.get('opacity', 0.7))

                gambar_hasil_cv, error_msg = proses_watermark(
                    main_img_bytes,
                    watermark_img_bytes,
                    position=position,
                    scale_percent=scale,
                    opacity=opacity_form
                )
                # ... (penanganan error dan send_file) ...
                if error_msg:
                    flash(f"Error watermark: {error_msg}", "danger")
                elif gambar_hasil_cv is not None:
                    is_success, buffer = cv2.imencode(".png", gambar_hasil_cv)
                    if is_success:
                        img_io = io.BytesIO(buffer)
                        return send_file(img_io, mimetype='image/png', as_attachment=True, download_name='gambar_watermarked.png')
                    else:
                        flash("Gagal mengencode gambar watermark.", "danger")
                else:
                    flash("Gagal memproses watermark.", "danger")
            except Exception as e:
                app.logger.error(f"Error watermark: {str(e)}")
                flash(f"Terjadi kesalahan: {str(e)}", "danger")
        return redirect(request.url)
    return render_template('watermark.html', title="Watermark")

# --- SKETCH WEBCAM ---


@app.route('/sketch-webcam')
def sketch_webcam_page():
    return render_template('sketch_webcam.html', title="Sketch dari Webcam")


@app.route('/api/sketch_frame', methods=['POST'])
def api_process_sketch_frame():
    # app.logger.info("API_SKETCH_FRAME: Request diterima.") # Bisa di-nonaktifkan jika sudah stabil
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Tidak ada data JSON."}), 400

        image_data_url = data.get('image_data_url')
        if not image_data_url:
            return jsonify({"error": "'image_data_url' tidak ditemukan."}), 400

        try:
            header, encoded_data = image_data_url.split(",", 1)
        except ValueError:
            return jsonify({"error": "Format data URL tidak valid."}), 400

        img_bytes = base64.b64decode(encoded_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame_original is None:
            return jsonify({"error": "Gagal mendekode gambar."}), 400

        # Panggil fungsi sketsa
        frame_processed = apply_sketch_to_frame(frame_original)

        if frame_processed is None:
            return jsonify({"error": "Pemrosesan sketsa gagal."}), 500

        ret, buffer_encode = cv2.imencode('.jpg', frame_processed)
        if not ret or buffer_encode is None or len(buffer_encode) == 0:
            return jsonify({"error": "Gagal mengenkode gambar hasil."}), 500

        output_base64_bytes = base64.b64encode(buffer_encode)
        output_data_url = f"data:image/jpeg;base64,{output_base64_bytes.decode('utf-8')}"

        return jsonify({"sketch_image_data_url": output_data_url})

    except Exception as e:
        app.logger.error(
            f"API_SKETCH_FRAME Exception: {str(e)}", exc_info=True)
        return jsonify({"error": f"Kesalahan server internal."}), 500


if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
