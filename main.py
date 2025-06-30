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

# main.py
# ... (impor dan kode lainnya tetap sama) ...
if __name__ == "__main__":
    config = Config()
    # ... (kode kalibrasi tetap sama) ...

    # Inisialisasi komponen backend
    supabase = SupabaseHandler(config.SUPABASE_URL, config.SUPABASE_KEY)
    notifier = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
    analyzer = ChickenDensityAnalyzer(config, supabase, notifier)

    if analyzer.is_ready():
        # --- TAMBAHKAN BLOK INI UNTUK MEMULAI SEMUA THREAD ---
        
        # 1. Mulai thread pembaca frame (Si Cepat)
        threading.Thread(target=analyzer._read_video_source_thread, daemon=True).start()

        # 2. Mulai thread analis untuk MJPEG Stream (Si Pintar)
        threading.Thread(target=analyzer.process_rtsp_stream_for_mjpeg, daemon=True).start()

        # --------------------------------------------------------

        # Pengaturan Aplikasi Web (Flask)
        app = Flask(__name__, template_folder='templates')
        register_web_routes(app, supabase)
        streamer = MJPEGStreamer(analyzer)
        streamer.register_stream_route(app)
        
        print(f"Memulai server dashboard di http://{config.MJPEG_HOST}:{config.MJPEG_PORT}/")
        threading.Thread(
            target=lambda: app.run(host=config.MJPEG_HOST, port=config.MJPEG_PORT, debug=False, use_reloader=False),
            daemon=True
        ).start()

        # ... (Sisa file, penjadwalan, dll, tetap sama) ...
        print("\nMenjalankan siklus pemetaan pertama kali...")
        # Beri jeda agar thread pembaca sempat mengambil frame pertama
        time.sleep(5) 
        analyzer.run_mapping_cycle()
        schedule.every(2).minutes.do(analyzer.run_mapping_cycle)
        print(f"Pemetaan kepadatan dijadwalkan berjalan setiap 10 menit. Tekan Ctrl+C untuk berhenti.")

        try:
            while True: schedule.run_pending(); time.sleep(1)
        except KeyboardInterrupt: print("\nMenghentikan skrip...")
        finally: print("Skrip dihentikan.")
    else:
        print("\n❌ Inisialisasi Gagal.")