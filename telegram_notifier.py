import requests

class TelegramNotifier:
    """Mengirim notifikasi ke channel/user Telegram."""
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        if token and chat_id: self.base_url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        else: self.base_url = None
    def send_notification(self, message):
        if not self.base_url: print("WARNING: Token/Chat ID Telegram belum diatur. Notifikasi dilewati."); return
        payload = {'chat_id': self.chat_id, 'text': message, 'parse_mode': 'HTML'}
        try:
            response = requests.post(self.base_url, data=payload, timeout=5)
            if response.status_code == 200: print("Notifikasi Telegram berhasil dikirim.")
            else: print(f"Gagal mengirim notifikasi Telegram: {response.status_code} - {response.text}")
        except Exception as e: print(f"Error saat mengirim notifikasi Telegram: {e}")