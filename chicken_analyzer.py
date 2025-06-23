import cv2
from ultralytics import YOLO
import numpy as np
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
        
        self.yolo_model = self._load_yolo_model()
        self.homography_matrix = self._load_numpy_file(config.HOMOGRAPHY_MATRIX_PATH)
        self.roi_polygon_coords = self._load_numpy_file(config.SELECTED_AREA_POINTS_PATH, dtype=np.float32)
        
        self.latest_annotated_frame = None
        self.frame_lock = threading.Lock() # Untuk thread-safety

        os.makedirs(config.TEMP_PLOT_DIR, exist_ok=True)
    
    def _load_yolo_model(self):
        try:
            if os.path.exists(self.config.YOLO_MODEL_PATH):
                print(f"Model YOLO berhasil dimuat dari: {self.config.YOLO_MODEL_PATH}")
                return YOLO(self.config.YOLO_MODEL_PATH)
        except Exception as e:
            print(f"Error fatal saat memuat model YOLO: {e}")
        return None

    def _load_numpy_file(self, path, dtype=None):
        try:
            if os.path.exists(path):
                print(f"File kalibrasi berhasil dimuat: {path}")
                return np.load(path).astype(dtype) if dtype else np.load(path)
        except Exception as e:
            print(f"Error fatal saat memuat file kalibrasi {path}: {e}")
        return None

    def is_ready(self):
        """Cek jika semua komponen penting telah berhasil dimuat."""
        return all([
            self.yolo_model is not None,
            self.homography_matrix is not None,
            self.roi_polygon_coords is not None
        ])

    def _capture_and_detect(self, cap_source):
        """Mengambil satu frame dan melakukan deteksi YOLO."""
        ret, frame = cap_source.read()
        if not ret: return None, []

        detections = []
        try:
            results = self.yolo_model(frame, verbose=False)
            if results and results[0].boxes:
                for box in results[0].boxes:
                    if self.yolo_model.names[int(box.cls[0].item())].lower() == self.config.TARGET_CLASS_NAME.lower():
                        detections.append({
                            'xyxy': box.xyxy[0].tolist(),
                            'conf': float(box.conf[0].item())
                        })
        except Exception as e:
            print(f"Error saat deteksi YOLO: {e}")
        
        return frame, detections

    def run_mapping_cycle(self):
        """Menjalankan satu siklus penuh: capture, detect, analyze, plot, upload, notify."""
        print("\n--- Memulai Siklus Pemetaan Kepadatan ---")
        if not self.is_ready():
            print("Analyzer tidak siap, siklus dibatalkan.")
            return
            
        cap = cv2.VideoCapture(self.config.VIDEO_SOURCE)
        if not cap.isOpened():
            print(f"Error: Tidak dapat membuka sumber video {self.config.VIDEO_SOURCE}")
            return
        
        frame, detections = self._capture_and_detect(cap)
        cap.release()

        if frame is None:
            print("Gagal mengambil snapshot. Siklus dihentikan.")
            return
        
        current_ts = datetime.now()
        ts_str = current_ts.strftime("%Y%m%d_%H%M%S")
        
        # Proses pemetaan dan pembuatan plot
        annotated_image, world_coords, in_roi_count = self._process_detections(frame.copy(), detections)
        plot_path, grid_data, high_density_alerts = self._create_density_plot(world_coords, current_ts)

        # Simpan gambar teranotasi sementara
        annotated_img_path = os.path.join(self.config.TEMP_PLOT_DIR, f"annotated_snapshot_{ts_str}.jpg")
        cv2.imwrite(annotated_img_path, annotated_image)
        
        # Unggah ke Supabase
        snapshot_url = self.supabase_handler.upload_file(self.config.BUCKET_SNAPSHOT_NAME, annotated_img_path, os.path.basename(annotated_img_path))
        plot_url = self.supabase_handler.upload_file(self.config.BUCKET_PLOT_NAME, plot_path, os.path.basename(plot_path))
        
        # Simpan data ke database Supabase
        if snapshot_url and plot_url:
            db_data = {
                "mapping_timestamp": current_ts.replace(tzinfo=timezone.utc).isoformat(),
                "source_screenshot_url": snapshot_url,
                "density_plot_url": plot_url,
                "chickens_in_roi_count": in_roi_count,
                "grid_density_data": grid_data
            }
            self.supabase_handler.insert_mapping_data(db_data)

        # [FITUR BARU] Kirim notifikasi jika ada kepadatan tinggi
        if high_density_alerts:
            alert_message = f"<b>⚠️ Peringatan Kepadatan Tinggi!</b>\n"
            alert_message += f"Terdeteksi pada {current_ts.strftime('%d-%m-%Y %H:%M:%S')}\n\n"
            for alert in high_density_alerts:
                alert_message += f"• <b>Grid ({alert['grid_x']}, {alert['grid_y']})</b>: {alert['count']} ayam ({alert['density']:.1f}/m²)\n"
            alert_message += f"\nSegera cek <a href='{plot_url}'>plot kepadatan</a> untuk detail."
            self.notifier.send_notification(alert_message)
            
        # Hapus file sementara
        for f_path in [annotated_img_path, plot_path]:
            if f_path and os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except OSError as e:
                    print(f"Error menghapus file sementara '{f_path}': {e}")
        
        print("--- Siklus Pemetaan Selesai ---")

    def _process_detections(self, image_to_draw, detections):
        """Memproses deteksi, melakukan transformasi perspektif, dan menganotasi gambar."""
        x_world, y_world = [], []
        in_roi_count = 0
        
        pts_for_polylines = np.array(self.roi_polygon_coords, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image_to_draw, [pts_for_polylines], isClosed=True, color=(255, 0, 0), thickness=3)

        for det in detections:
            x1, y1, x2, y2 = map(int, det['xyxy'])
            cv2.rectangle(image_to_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Ambil titik tengah dan bawah dari bounding box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2 # <-- HITUNG CENTER_Y YANG BENAR
            bottom_y = float(y2)
            point_to_test = (center_x, bottom_y)
            
            if cv2.pointPolygonTest(self.roi_polygon_coords, point_to_test, False) >= 0:
                in_roi_count += 1
                
                # === INI BAGIAN YANG DIPERBAIKI ===
                # Gunakan (center_x, center_y) untuk transformasi, bukan (center_x, center_x)
                pt_pixel = np.float32([[center_x, center_y]]).reshape(1, 1, 2)
                # ==================================

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
        """Membuat plot kepadatan menggunakan Matplotlib dengan gaya visual yang bagus."""
        x_coords, y_coords = world_coords
        cfg = self.config
        
        num_cols = math.ceil(cfg.REAL_WORLD_WIDTH_M / cfg.GRID_SIZE_X)
        num_rows = math.ceil(cfg.REAL_WORLD_HEIGHT_M / cfg.GRID_SIZE_Y)
        
        counts_per_grid = {}
        for xw, yw in zip(x_coords, y_coords):
            gx = math.floor(xw / cfg.GRID_SIZE_X)
            gy = math.floor(yw / cfg.GRID_SIZE_Y)
            counts_per_grid[(gx, gy)] = counts_per_grid.get((gx, gy), 0) + 1
        
        grid_data_for_db = {f"{k[0]}_{k[1]}": v for k, v in counts_per_grid.items()}
        high_density_alerts = []

        # === GAYA VISUAL BARU DIMULAI DARI SINI ===
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.set_facecolor('whitesmoke') # Ganti background agar tidak transparan

        # Gambar area utama dengan warna biru muda
        ax.add_patch(plt.Rectangle((0, 0), cfg.REAL_WORLD_WIDTH_M, cfg.REAL_WORLD_HEIGHT_M,
                                    facecolor='skyblue', alpha=0.2, edgecolor='none'))

        # Gambar teks kepadatan dan highlight area merah
        for r in range(num_rows):
            for c in range(num_cols):
                count = counts_per_grid.get((c, r), 0)
                grid_area = cfg.GRID_SIZE_X * cfg.GRID_SIZE_Y
                density = count / grid_area if grid_area > 0 else 0
                
                # Tampilkan teks jika ada ayam di grid tersebut
                if count > 0:
                    plt.text((c + 0.5) * cfg.GRID_SIZE_X, (r + 0.5) * cfg.GRID_SIZE_Y,
                            f"{count}\n({density:.1f}/m²)", ha='center', va='center', fontsize=10, color='blue', weight='bold')
                
                # Highlight area merah jika padat
                if density > cfg.MAX_AYAM_PER_METER_PERSEGI:
                    ax.add_patch(plt.Rectangle((c * cfg.GRID_SIZE_X, r * cfg.GRID_SIZE_Y), cfg.GRID_SIZE_X, cfg.GRID_SIZE_Y,
                                            facecolor='red', alpha=0.4, edgecolor='none'))
                    high_density_alerts.append({'grid_x': c, 'grid_y': r, 'count': count, 'density': density})

        # Gambar titik ayam (merah dengan outline hitam)
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
        plt.savefig(plot_path, bbox_inches='tight', dpi=150) # Tingkatkan DPI untuk kualitas
        plt.close()

        return plot_path, grid_data_for_db, high_density_alerts

    
    def process_rtsp_stream_for_mjpeg(self):
        """Looping untuk membaca stream, deteksi, dan update frame untuk MJPEG."""
        if not self.yolo_model:
            print("[MJPEG Stream] Model YOLO tidak siap.")
            return

        print("[MJPEG Stream] Memulai thread pemrosesan video untuk MJPEG...")
        cap = None # Inisialisasi cap di luar loop

        while True:
            # Pastikan cap terinisialisasi dan terbuka
            if cap is None or not cap.isOpened():
                print(f"[MJPEG Stream] Mencoba membuka video: {self.config.VIDEO_SOURCE}")
                if cap is not None:
                    cap.release() # Pastikan resource lama dilepaskan
                
                cap = cv2.VideoCapture(self.config.VIDEO_SOURCE)
                
                # Coba set buffer size ke 1 untuk mengurangi latency (opsional, bisa membantu)
                # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                time.sleep(0.5) # Beri sedikit waktu untuk file terbuka
                
                if not cap.isOpened():
                    print("[MJPEG Stream] Gagal membuka video. Mencoba lagi dalam 5 detik...")
                    time.sleep(5)
                    continue # Langsung coba lagi di iterasi berikutnya

            ret, frame = cap.read()

            if not ret:
                # Jika tidak ada frame yang berhasil dibaca (akhir video atau error)
                print("[MJPEG Stream] Akhir video atau gagal membaca frame. Mencoba mengulang video...")
                cap.release() # Lepaskan resource
                cap = None # Set cap ke None agar di iterasi berikutnya akan mencoba membuka kembali
                time.sleep(0.1) # Jeda singkat sebelum mengulang
                continue # Langsung coba lagi di iterasi berikutnya

            # --- OPTIMASI KUNCI UNTUK KECEPATAN: RESIZE FRAME SEBELUM YOLO ---
            # Resolusi yang lebih rendah mempercepat inferensi YOLO secara drastis
            # Contoh: Resize ke lebar 640px sambil menjaga aspect ratio
            target_width = 640 
            if frame.shape[1] > target_width: # Hanya resize jika frame lebih besar dari target
                aspect_ratio = float(frame.shape[1]) / float(frame.shape[0])
                target_height = int(target_width / aspect_ratio)
                frame_for_yolo = cv2.resize(frame, (target_width, target_height))
            else:
                frame_for_yolo = frame.copy() # Jika sudah kecil, gunakan frame asli
            # --- AKHIR OPTIMASI ---

            # Lakukan deteksi YOLO
            # Gunakan frame_for_yolo yang sudah di-resize
            results = self.yolo_model(frame_for_yolo, verbose=False, half=True, device='cpu') 
            
            # Selalu mulai dengan salinan frame asli untuk memastikan video bergerak
            annotated_frame = frame.copy() 

            if results and results[0].boxes:
                for box in results[0].boxes:
                    detected_class_id = int(box.cls[0].item())
                    detected_class_name = self.yolo_model.names[detected_class_id].lower()

                    if detected_class_name == self.config.TARGET_CLASS_NAME.lower():
                        # Koordinat dari YOLO (mungkin dari frame_for_yolo) perlu diskalakan kembali
                        # ke ukuran frame asli untuk menggambar di annotated_frame
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        
                        # Skalakan koordinat jika frame di-resize sebelumnya
                        if frame.shape[1] > target_width:
                            scale_x = frame.shape[1] / frame_for_yolo.shape[1]
                            scale_y = frame.shape[0] / frame_for_yolo.shape[0]
                            x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Update frame yang akan di-stream secara thread-safe
            with self.frame_lock:
                self.latest_annotated_frame = annotated_frame.copy()
            
            # HAPUS time.sleep() di sini atau set ke nilai sangat kecil (misal 0.001)
            # Biarkan loop berjalan secepat mungkin jika performa YOLO sudah baik
            # time.sleep(0.001)
        cap.release()