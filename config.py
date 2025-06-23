import os
from dotenv import load_dotenv

# Baris ini akan mencari file .env dan memuat isinya sebagai environment variables
load_dotenv()

class Config:
    """Menyimpan semua konfigurasi aplikasi, membaca kredensial dari .env"""
    def __init__(self):
        # --- Konfigurasi yang Tidak Rahasia ---
        self.YOLO_MODEL_PATH = r"D:\Downloads\v8n.pt"
        self.VIDEO_SOURCE = r"D:\skripsi\ayam_panjang.mp4"#"rtsp://localhost:8554/mystream"
        self.TARGET_CLASS_NAME = 'broiler'
        self.MJPEG_HOST = '0.0.0.0'
        self.MJPEG_PORT = 8080
        self.HOMOGRAPHY_MATRIX_PATH = r'D:\Downloads\homography_matrix.npy'
        self.SELECTED_AREA_POINTS_PATH = r'D:\Downloads\selected_area_points.npy'
        self.REAL_WORLD_WIDTH_M = 3.0
        self.REAL_WORLD_HEIGHT_M = 3.0
        self.GRID_SIZE_X = 1.0
        self.GRID_SIZE_Y = 1.0
        self.MAX_AYAM_PER_METER_PERSEGI = 12
        self.TEMP_PLOT_DIR = r"D:\Downloads\temp_files"
        self.BUCKET_SNAPSHOT_NAME = "ssayam"
        self.BUCKET_PLOT_NAME = "plotayam"

        # --- Konfigurasi Rahasia dibaca dari Environment Variables (.env) ---
        self.SUPABASE_URL = os.getenv("SUPABASE_URL")
        self.SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        self.TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") 
        self.TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")