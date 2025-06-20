# web_routes.py

from flask import render_template, jsonify

def register_web_routes(app, supabase_handler):

    @app.route('/')
    def index():
        return render_template('index.html', title='Dashboard Kandang Broiler')

    @app.route('/api/dashboard_data')
    def get_dashboard_data():
        try:
            # === PERBAIKAN DI SINI ===
            # 1. Hapus .single() dari query
            response = supabase_handler.client.table("density_mappings") \
                .select("density_plot_url") \
                .order("id", desc=True) \
                .limit(1) \
                .execute()

            # 2. Periksa apakah `response.data` memiliki isi
            if response.data:
                # Jika ada data, ambil item pertama dari list
                data_to_send = response.data[0] 
            else:
                # Jika tidak ada data (tabel kosong), buat data placeholder
                data_to_send = {
                    "density_plot_url": "https://via.placeholder.com/400x400/343a40/ffffff?text=Belum+Ada+Data+Analisis"
                }
            
            return jsonify({ "mapping": data_to_send })

        except Exception as e:
            print(f"Error saat mengambil data untuk API: {e}")
            return jsonify({"error": str(e)}), 500