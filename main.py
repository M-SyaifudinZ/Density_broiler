# main.py
import schedule
import time
import threading
import os

from config import Config
from calibrator import Calibrator
from supabase_handler import SupabaseHandler
from telegram_notifier import TelegramNotifier
from chicken_analyzer import ChickenDensityAnalyzer
from mjpeg_streamer import MJPEGStreamer
from flask import render_template, jsonify 
from flask import Flask 
from web_routes import register_web_routes

if __name__ == "__main__":
    config = Config()

    # Pengecekan dan proses kalibrasi otomatis
    homography_exists = os.path.exists(config.HOMOGRAPHY_MATRIX_PATH)
    points_exists = os.path.exists(config.SELECTED_AREA_POINTS_PATH)
    if not homography_exists or not points_exists:
        print("="*50 + "\n⚠️  File kalibrasi tidak ditemukan!\n   Aplikasi akan masuk ke mode kalibrasi interaktif.\n" + "="*50)
        calibrator = Calibrator(config)
        if not calibrator.run_calibration():
            print("\n❌ Kalibrasi tidak selesai. Aplikasi tidak dapat dilanjutkan. Keluar."); exit()
        print("\n✅ Kalibrasi berhasil disimpan. Melanjutkan start-up aplikasi...")
    else:
        print("✅ File kalibrasi ditemukan. Melewatkan langkah kalibrasi.")

   # Inisialisasi komponen backend
    supabase = SupabaseHandler(config.SUPABASE_URL, config.SUPABASE_KEY)
    notifier = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
    analyzer = ChickenDensityAnalyzer(config, supabase, notifier)

    if analyzer.is_ready():
        # --- PENGATURAN APLIKASI WEB ---
        # 1. Buat instance aplikasi Flask utama di sini
        app = Flask(__name__, template_folder='templates')

        # 2. Daftarkan rute-rute dari file lain
        register_web_routes(app, supabase) # Daftarkan rute dashboard dan API
        
        streamer = MJPEGStreamer(analyzer)
        streamer.register_stream_route(app) # Daftarkan rute video stream

        # 3. Jalankan thread pemrosesan video untuk MJPEG
        threading.Thread(target=analyzer.process_rtsp_stream_for_mjpeg, daemon=True).start()
        
        # 4. Jalankan server Flask di thread terpisah
        print(f"Memulai server dashboard di http://{config.MJPEG_HOST}:{config.MJPEG_PORT}/")
        threading.Thread(
            target=lambda: app.run(host=config.MJPEG_HOST, port=config.MJPEG_PORT, debug=False, use_reloader=False),
            daemon=True
        ).start()

        # --- PENJADWALAN PROSES ANALISIS (tetap sama) ---
        print("\nMenjalankan siklus pemetaan pertama kali...")
        analyzer.run_mapping_cycle()
        schedule.every(10).minutes.do(analyzer.run_mapping_cycle)
        print(f"Pemetaan kepadatan dijadwalkan berjalan setiap 10 menit. Tekan Ctrl+C untuk berhenti.")

        try:
            while True: schedule.run_pending(); time.sleep(1)
        except KeyboardInterrupt: print("\nMenghentikan skrip...")
        finally: print("Skrip dihentikan.")
    else:
        print("\n❌ Inisialisasi Gagal. Ada komponen penting (model/kalibrasi) yang tidak berhasil dimuat.")
