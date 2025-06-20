# calibrator.py
import cv2
import numpy as np
from config import Config

class Calibrator:
    """Kelas untuk menangani proses kalibrasi interaktif."""
    def __init__(self, config: Config):
        self.config = config
        self.clicked_points = []
        self.scale_factor = 1.0
        self.original_res_image = None

    def _mouse_callback(self, event, x, y, flags, param):
        display_img = param['display_image']
        clean_copy = param['clean_copy']
        current_drawing = clean_copy.copy()
        if event == cv2.EVENT_LBUTTONDOWN and len(self.clicked_points) < 4:
            self.clicked_points.append([x, y])
            labels = ["Pojok Kiri Atas", "Pojok Kanan Atas", "Pojok Kanan Bawah", "Pojok Kiri Bawah"]
            print(f"Titik ke-{len(self.clicked_points)} ({labels[len(self.clicked_points)-1]}): ({x}, {y})")
        for i, pt in enumerate(self.clicked_points):
            cv2.circle(current_drawing, tuple(pt), 7, (0, 0, 255), -1)
            cv2.putText(current_drawing, str(i + 1), (pt[0] + 10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if i > 0: cv2.line(current_drawing, tuple(self.clicked_points[i - 1]), tuple(pt), (0, 255, 0), 2)
        if len(self.clicked_points) == 4:
            cv2.line(current_drawing, tuple(self.clicked_points[3]), tuple(self.clicked_points[0]), (0, 255, 0), 2)
            overlay = current_drawing.copy()
            pts = np.array(self.clicked_points, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], (0, 200, 0))
            current_drawing = cv2.addWeighted(overlay, 0.3, current_drawing, 0.7, 0)
        display_img[:] = current_drawing[:]

    def run_calibration(self):
        print(f"Mencoba mengambil frame referensi dari: {self.config.VIDEO_SOURCE}...")
        cap = cv2.VideoCapture(self.config.VIDEO_SOURCE)
        if not cap.isOpened():
            print(f"FATAL: Gagal membuka stream video '{self.config.VIDEO_SOURCE}' untuk kalibrasi.")
            return False
        ret, frame = cap.read(); cap.release()
        if not ret or frame is None:
            print("FATAL: Gagal mengambil frame dari stream video. Pastikan stream aktif.")
            return False
        self.original_res_image = frame
        h_orig, w_orig = self.original_res_image.shape[:2]
        display_image = self.original_res_image.copy()
        if w_orig > 1280:
            self.scale_factor = 1280 / w_orig
            display_image = cv2.resize(display_image, (int(w_orig * self.scale_factor), int(h_orig * self.scale_factor)))
        window_name = "KALIBRASI: Pilih 4 Titik Area Lantai"
        cv2.namedWindow(window_name)
        callback_params = {'display_image': display_image, 'clean_copy': display_image.copy()}
        cv2.setMouseCallback(window_name, self._mouse_callback, callback_params)
        print("\n--- INSTRUKSI KALIBRASI ---\nPada window yang muncul, klik 4 titik sudut di lantai kandang.\nURUTAN PENTING: Kiri Atas -> Kanan Atas -> Kanan Bawah -> Kiri Bawah.\n\nSetelah 4 titik dipilih:\n  - Tekan 'c' untuk KONFIRMASI dan simpan kalibrasi.\n  - Tekan 'r' untuk RESET pilihan titik.\n  - Tekan 'q' untuk KELUAR tanpa menyimpan.")
        while True:
            cv2.imshow(window_name, display_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('r'): self.clicked_points = []; print("Titik direset."); display_image[:] = callback_params['clean_copy'][:]
            if key == ord('c') and len(self.clicked_points) == 4:
                src_pts_original_res = np.float32(self.clicked_points) / self.scale_factor
                dst_pts_realworld = np.float32([[0, self.config.REAL_WORLD_HEIGHT_M], [self.config.REAL_WORLD_WIDTH_M, self.config.REAL_WORLD_HEIGHT_M], [self.config.REAL_WORLD_WIDTH_M, 0], [0, 0]])
                homography_matrix, _ = cv2.findHomography(src_pts_original_res, dst_pts_realworld)
                if homography_matrix is not None:
                    np.save(self.config.HOMOGRAPHY_MATRIX_PATH, homography_matrix); print(f"✅ Matriks Homografi disimpan di: {self.config.HOMOGRAPHY_MATRIX_PATH}")
                    np.save(self.config.SELECTED_AREA_POINTS_PATH, src_pts_original_res.astype(np.int32)); print(f"✅ Titik Area (ROI) disimpan di: {self.config.SELECTED_AREA_POINTS_PATH}")
                    print("\nKalibrasi berhasil! Menutup window..."); cv2.destroyAllWindows(); return True
                else: print("❌ Gagal menghitung matriks homografi. Coba lagi.")
        cv2.destroyAllWindows()
        return False