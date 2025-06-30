# chicken_analyzer.py (VERSI SUDAH DIPERBAIKI)

import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math, os
from datetime import datetime, timezone
from config import Config
from supabase_handler import SupabaseHandler
from telegram_notifier import TelegramNotifier
import threading
import time

class ChickenDensityAnalyzer:
    """Otak dari aplikasi: memuat model, memproses gambar, dan analisis."""
    def __init__(self, config: Config, supabase_handler: SupabaseHandler, notifier: TelegramNotifier):
        self.config = config
        self.supabase_handler = supabase_handler
        self.notifier = notifier
        
        print("Memuat model YOLOv8n untuk Live Stream...")
        self.yolo_model_n = self._load_yolo_model(config.YOLO_MODEL_PATH_NANO)
        print("Memuat model YOLOv8s untuk Analisis Kepadatan...")
        self.yolo_model_s = self._load_yolo_model(config.YOLO_MODEL_PATH_SMALL)
        
        self.homography_matrix = self._load_numpy_file(config.HOMOGRAPHY_MATRIX_PATH)
        self.roi_polygon_coords = self._load_numpy_file(config.SELECTED_AREA_POINTS_PATH, dtype=np.float32)
        
        self.latest_frame_from_source = None
        self.latest_annotated_frame = None
        self.source_lock = threading.Lock()
        self.annotation_lock = threading.Lock()

        self.selected_area_points = None
        self.load_calibration_data()
        
        os.makedirs(config.TEMP_PLOT_DIR, exist_ok=True)

    def _load_yolo_model(self, model_path):
        try:
            if os.path.exists(model_path):
                print(f"Model YOLO berhasil dimuat dari: {model_path}")
                return YOLO(model_path)
        except Exception as e:
            print(f"Error fatal saat memuat model YOLO dari {model_path}: {e}")
        return None

    def _read_video_source_thread(self):
        print("[Frame Reader] Thread pembaca video dimulai.")
        cap = None
        while True:
            try:
                if cap is None or not cap.isOpened():
                    print(f"[Frame Reader] Membuka koneksi ke {self.config.VIDEO_SOURCE}...")
                    if cap: cap.release()
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                    cap = cv2.VideoCapture(self.config.VIDEO_SOURCE, cv2.CAP_FFMPEG)
                    time.sleep(2)
                    if not cap.isOpened():
                        print("[Frame Reader] Gagal membuka koneksi. Mencoba lagi dalam 5 detik...")
                        time.sleep(5)
                        continue

                ret, frame = cap.read()
                if ret:
                    with self.source_lock:
                        self.latest_frame_from_source = frame.copy()
                else:
                    print("[Frame Reader] Frame tidak terbaca, koneksi mungkin terputus. Mencoba lagi...")
                    cap.release()
                    cap = None
                    time.sleep(2)
                time.sleep(0.01)
            except Exception as e:
                print(f"[Frame Reader] Error: {e}")
                time.sleep(5)

    def process_rtsp_stream_for_mjpeg(self):
        print(f"[Analyzer] Thread analisis dimulai dengan interval {self.config.STREAM_PROCESSING_INTERVAL_S} detik.")
        while True:
            loop_start_time = time.time()
            
            current_frame = None
            with self.source_lock:
                if self.latest_frame_from_source is not None:
                    current_frame = self.latest_frame_from_source.copy()

            if current_frame is None:
                time.sleep(0.5)
                continue

            annotated_frame = self._analyze_and_annotate_frame(current_frame)
            with self.annotation_lock:
                self.latest_annotated_frame = annotated_frame

            processing_time = time.time() - loop_start_time
            sleep_duration = self.config.STREAM_PROCESSING_INTERVAL_S - processing_time
            if sleep_duration > 0:
                time.sleep(sleep_duration)

    def _analyze_and_annotate_frame(self, frame):
        target_width = 640
        frame_for_yolo = frame
        if frame.shape[1] > target_width:
            aspect_ratio = frame.shape[1] / frame.shape[0]
            target_height = int(target_width / aspect_ratio)
            frame_for_yolo = cv2.resize(frame, (target_width, target_height))
        
        # Gunakan model NANO untuk live view
        results = self.yolo_model_n(frame_for_yolo, conf=0.4, verbose=False, half=True, device='cpu')

        if results and results[0].boxes:
            for box in results[0].boxes:
                # === PERBAIKAN #1 DI SINI ===
                if self.yolo_model_n.names[int(box.cls[0].item())].lower() == self.config.TARGET_CLASS_NAME.lower():
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    if frame.shape[1] > target_width:
                        scale_x = frame.shape[1] / frame_for_yolo.shape[1]
                        scale_y = frame.shape[0] / frame_for_yolo.shape[0]
                        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if self.selected_area_points is not None:
            cv2.polylines(frame, [self.selected_area_points], isClosed=True, color=(255, 0, 0), thickness=2)

        return frame

    def load_calibration_data(self):
        try:
            if self.config.SELECTED_AREA_POINTS_PATH and os.path.exists(self.config.SELECTED_AREA_POINTS_PATH):
                self.selected_area_points = np.load(self.config.SELECTED_AREA_POINTS_PATH)
                if self.selected_area_points.ndim == 2:
                    self.selected_area_points = self.selected_area_points.reshape((-1, 1, 2))
                print(f"Titik area kalibrasi berhasil dimuat dari {self.config.SELECTED_AREA_POINTS_PATH}")
            else:
                self.selected_area_points = None
        except Exception as e:
            print(f"Error memuat titik area kalibrasi: {e}")
            self.selected_area_points = None

    def _load_numpy_file(self, path, dtype=None):
        try:
            if os.path.exists(path):
                return np.load(path).astype(dtype) if dtype else np.load(path)
        except Exception as e:
            print(f"Error fatal saat memuat file kalibrasi {path}: {e}")
        return None

    def is_ready(self):
        return all([self.yolo_model_s is not None, self.yolo_model_n is not None, self.homography_matrix is not None, self.roi_polygon_coords is not None])

    # === PERBAIKAN #2: FUNGSI INI DIRAPIKAN TOTAL ===
    def run_mapping_cycle(self):
        # Tambahkan print ini untuk debugging, untuk memastikan fungsi yang benar dipanggil
        print("\n>>> DEBUG: Memulai run_mapping_cycle. Akan menggunakan model 's'.")

        if not self.yolo_model_s:
            print("Model YOLOv8s tidak siap, siklus dibatalkan.")
            return
        
        # 1. Ambil frame terbaru dari thread pembaca.
        frame = None
        with self.source_lock:
            if self.latest_frame_from_source is not None:
                frame = self.latest_frame_from_source.copy()

        if frame is None:
            print("Gagal mengambil snapshot dari stream. Siklus dihentikan.")
            return
        
        # 2. Lakukan deteksi AKURAT langsung pada frame yang sudah didapat.
        print(">>> DEBUG: Memanggil self.yolo_model_s untuk deteksi...")
        detections = []
        try:
            results = self.yolo_model_s(frame, conf=0.2, verbose=False) 
            if results and results[0].boxes:
                for box in results[0].boxes:
                    if self.yolo_model_s.names[int(box.cls[0].item())].lower() == self.config.TARGET_CLASS_NAME.lower():
                        detections.append({'xyxy': box.xyxy[0].tolist(), 'conf': float(box.conf[0].item())})
        except Exception as e:
            print(f"Error saat deteksi YOLOv8s: {e}")
        
        print(f"Deteksi dengan YOLOv8s selesai, ditemukan {len(detections)} ayam.")
        
        # 3. Sisa fungsi berjalan seperti biasa untuk membuat plot dan upload.
        current_ts = datetime.now()
        ts_str = current_ts.strftime("%Y%m%d_%H%M%S")
        
        annotated_image, world_coords, in_roi_count = self._process_detections(frame.copy(), detections)
        plot_path, grid_data, high_density_alerts = self._create_density_plot(world_coords, current_ts)

        # ... (sisa fungsi untuk upload ke supabase tidak perlu diubah)
        annotated_img_path = os.path.join(self.config.TEMP_PLOT_DIR, f"annotated_snapshot_{ts_str}.jpg")
        cv2.imwrite(annotated_img_path, annotated_image)
        snapshot_url = self.supabase_handler.upload_file(self.config.BUCKET_SNAPSHOT_NAME, annotated_img_path, os.path.basename(annotated_img_path))
        plot_url = self.supabase_handler.upload_file(self.config.BUCKET_PLOT_NAME, plot_path, os.path.basename(plot_path))
        heatmap_placeholder_url = "https://via.placeholder.com/300x200/343a40/ffffff?text=Heatmap+Belum+Tersedia"
        if snapshot_url and plot_url:
            db_data = {
                "mapping_timestamp": current_ts.replace(tzinfo=timezone.utc).isoformat(),
                "source_screenshot_url": snapshot_url,
                "density_plot_url": plot_url,
                "heatmap_plot_url": heatmap_placeholder_url,
                "chickens_in_roi_count": in_roi_count,
                "grid_density_data": grid_data
            }
            self.supabase_handler.insert_mapping_data(db_data)
        if high_density_alerts:
            # ... (notifikasi telegram tidak berubah)
            pass
        for f_path in [annotated_img_path, plot_path]:
            if f_path and os.path.exists(f_path):
                try: os.remove(f_path)
                except OSError as e: print(f"Error menghapus file sementara '{f_path}': {e}")

        print("--- Siklus Pemetaan (v8s) Selesai ---")


    def _process_detections(self, image_to_draw, detections):
        # ... (fungsi ini tidak berubah) ...
        x_world, y_world, in_roi_count = [], [], 0
        pts_for_polylines = np.array(self.roi_polygon_coords, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image_to_draw, [pts_for_polylines], isClosed=True, color=(255, 0, 0), thickness=3)
        for det in detections:
            x1, y1, x2, y2 = map(int, det['xyxy'])
            cv2.rectangle(image_to_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            center_x, center_y, bottom_y = (x1 + x2) / 2, (y1 + y2) / 2, float(y2)
            point_to_test = (center_x, bottom_y)
            if cv2.pointPolygonTest(self.roi_polygon_coords, point_to_test, False) >= 0:
                in_roi_count += 1
                pt_pixel = np.float32([[center_x, center_y]]).reshape(1, 1, 2)
                pt_world = cv2.perspectiveTransform(pt_pixel, self.homography_matrix)
                if pt_world is not None:
                    xw, yw = pt_world[0][0][0], pt_world[0][0][1]
                    x_world.append(np.clip(xw, 0, self.config.REAL_WORLD_WIDTH_M))
                    y_world.append(np.clip(yw, 0, self.config.REAL_WORLD_HEIGHT_M))
                    cv2.circle(image_to_draw, (int(center_x), int(bottom_y)), 5, (0, 255, 0), -1)
            else:
                cv2.circle(image_to_draw, (int(center_x), int(bottom_y)), 5, (0, 0, 255), -1)
        return image_to_draw, (x_world, y_world), in_roi_count

    def _create_density_plot(self, world_coords, timestamp):
        # ... (fungsi ini tidak berubah) ...
        x_coords, y_coords = world_coords
        cfg = self.config
        num_cols, num_rows = math.ceil(cfg.REAL_WORLD_WIDTH_M / cfg.GRID_SIZE_X), math.ceil(cfg.REAL_WORLD_HEIGHT_M / cfg.GRID_SIZE_Y)
        counts_per_grid = {}
        for xw, yw in zip(x_coords, y_coords):
            gx, gy = math.floor(xw / cfg.GRID_SIZE_X), math.floor(yw / cfg.GRID_SIZE_Y)
            counts_per_grid[(gx, gy)] = counts_per_grid.get((gx, gy), 0) + 1
        grid_data_for_db = {f"{k[0]}_{k[1]}": v for k, v in counts_per_grid.items()}
        high_density_alerts = []
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.set_facecolor('whitesmoke')
        ax.add_patch(plt.Rectangle((0, 0), cfg.REAL_WORLD_WIDTH_M, cfg.REAL_WORLD_HEIGHT_M, facecolor='skyblue', alpha=0.2, edgecolor='none'))
        for r in range(num_rows):
            for c in range(num_cols):
                count = counts_per_grid.get((c, r), 0)
                grid_area = cfg.GRID_SIZE_X * cfg.GRID_SIZE_Y
                density = count / grid_area if grid_area > 0 else 0
                if count > 0:
                    plt.text((c + 0.5) * cfg.GRID_SIZE_X, (r + 0.5) * cfg.GRID_SIZE_Y, f"{count}\n({density:.1f}/m²)", ha='center', va='center', fontsize=10, color='blue', weight='bold')
                if density > cfg.MAX_AYAM_PER_METER_PERSEGI:
                    ax.add_patch(plt.Rectangle((c * cfg.GRID_SIZE_X, r * cfg.GRID_SIZE_Y), cfg.GRID_SIZE_X, cfg.GRID_SIZE_Y, facecolor='red', alpha=0.4, edgecolor='none'))
                    high_density_alerts.append({'grid_x': c, 'grid_y': r, 'count': count, 'density': density})
        if x_coords:
            plt.scatter(x_coords, y_coords, c='red', s=50, edgecolors='black', label='Posisi Ayam (Dalam ROI)')
        plt.xlim(-0.1, cfg.REAL_WORLD_WIDTH_M + 0.1)
        plt.ylim(-0.1, cfg.REAL_WORLD_HEIGHT_M + 0.1)
        plt.title(f"Kepadatan Ayam (ROI) @ {timestamp.strftime('%Y-%m-%d %H:%M:%S')} (Max {cfg.MAX_AYAM_PER_METER_PERSEGI}/m²)", fontsize=14)
        plt.xlabel("Lebar Area (X - meter)", fontsize=12)
        plt.ylabel("Kedalaman Area (Y - meter)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        ax.set_aspect('equal', 'box')
        plot_path = os.path.join(cfg.TEMP_PLOT_DIR, f"density_plot_{timestamp.strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        plt.close()
        return plot_path, grid_data_for_db, high_density_alerts