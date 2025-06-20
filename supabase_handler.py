import os
from supabase import create_client, Client

class SupabaseHandler:
    """Mengelola semua interaksi dengan Supabase (Auth, DB, Storage)."""
    def __init__(self, url, key):
        self.client: Client = None
        try:
            if url and key: self.client = create_client(url, key); print("Koneksi ke Supabase berhasil.")
            else: print("WARNING: URL atau Key Supabase tidak ada di .env.")
        except Exception as e: print(f"Gagal terkoneksi ke Supabase: {e}")
    def upload_file(self, bucket_name: str, local_file_path: str, storage_file_name: str):
        if not self.client or not os.path.exists(local_file_path): return None
        try:
            with open(local_file_path, 'rb') as f:
                self.client.storage.from_(bucket_name).upload(path=storage_file_name, file=f, file_options={"upsert": "true", "cacheControl": "3600"})
            public_url = self.client.storage.from_(bucket_name).get_public_url(storage_file_name)
            print(f"  Berhasil unggah '{storage_file_name}' ke bucket '{bucket_name}'.")
            return public_url
        except Exception as e: print(f"  Error saat mengunggah file ke Supabase: {e}"); return None
    def insert_mapping_data(self, data_to_insert: dict):
        if not self.client: return False
        try:
            response = self.client.table("density_mappings").insert(data_to_insert).execute()
            if response.data: print(f"  Data pemetaan berhasil disimpan ke DB (ID: {response.data[0].get('id')})."); return True
            elif hasattr(response, 'error') and response.error: print(f"  Gagal menyimpan data ke DB. Error: {response.error}")
            return False
        except Exception as e: print(f"  Error saat menyimpan data ke DB: {e}"); return False