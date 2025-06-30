# mjpeg_streamer.py
import cv2, time
from flask import Response

class MJPEGStreamer:
    def __init__(self, analyzer):
        self.analyzer = analyzer

    def _generate_frames(self):
        while True:
            # --- PERUBAHAN: Gunakan lock yang benar ---
            with self.analyzer.annotation_lock:
                frame = self.analyzer.latest_annotated_frame
            
            if frame is None:
                time.sleep(0.1)
                continue
                
            (flag, encoded_image) = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not flag:
                continue
                
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                  bytearray(encoded_image) + b'\r\n')
            # Beri jeda agar tidak membanjiri browser
            time.sleep(1/30) # Target stream ~30 FPS

    def register_stream_route(self, app):
        @app.route('/video_feed')
        def video_feed():
            return Response(self._generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')