# mjpeg_streamer.py
import cv2, time, threading
from flask import Response # Hanya butuh Response

class MJPEGStreamer:
    """Hanya menangani logika untuk generate frame MJPEG."""
    def __init__(self, analyzer):
        # Tidak ada lagi app = Flask(__name__) di sini
        self.analyzer = analyzer

    def _generate_frames(self):
        """Generator yang menghasilkan frame JPEG untuk stream."""
        while True:
            with self.analyzer.frame_lock: frame = self.analyzer.latest_annotated_frame
            if frame is None: time.sleep(0.1); continue
            (flag, encoded_image) = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not flag: continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                  bytearray(encoded_image) + b'\r\n')
            time.sleep(1/20)

    # Fungsi untuk mendaftarkan route video stream ke aplikasi Flask utama
    def register_stream_route(self, app):
         @app.route('/video_feed')
         def video_feed():
            return Response(self._generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')