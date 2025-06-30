# web_routes.py

from flask import render_template, jsonify

def register_web_routes(app, supabase_handler):

    @app.route('/')
    def index():
        return render_template('index.html', title='Dashboard Kandang Broiler')

    # --- API BARU YANG MENGGABUNGKAN SEMUA DATA ---
    @app.route('/api/dashboard_data')
    def get_dashboard_data():
        try:
            # 1. Ambil data plot terakhir dari tabel 'density_mappings'
            # --- Pastikan memilih 'heatmap_plot_url' juga ---
            mapping_response = supabase_handler.client.table("density_mappings") \
                .select("density_plot_url, heatmap_plot_url") \
                .order("id", desc=True) \
                .limit(1) \
                .execute()

            if mapping_response.data:
                latest_mapping_data = mapping_response.data[0]
            else:
                # Sediakan data placeholder jika tabel mapping kosong
                latest_mapping_data = {
                    "density_plot_url": "https://via.placeholder.com/300x200/343a40/ffffff?text=No+Density+Data",
                    "heatmap_plot_url": "https://via.placeholder.com/300x200/343a40/ffffff?text=No+Heatmap+Data"
                }

            # 2. Ambil data suhu terakhir dari semua sensor
            # --- Panggil fungsi baru yang sudah kita buat ---
            temperature_data = supabase_handler.get_latest_temperature_data()

            # Buat struktur data suhu untuk dikirim
            if temperature_data is None:
                # Jika terjadi error saat mengambil data
                print("API Warning: get_latest_temperature_data() mengembalikan None (error).")
                temperature_data_to_send = {"individual_sensors": []}
            else:
                temperature_data_to_send = {"individual_sensors": temperature_data}

            # 3. Gabungkan semua data menjadi satu response JSON
            final_data = {
                "mapping": latest_mapping_data,
                "temperature": temperature_data_to_send
            }
            
            return jsonify(final_data)

        except Exception as e:
            print(f"Error fatal di API /api/dashboard_data: {e}")
            # Kirim response error yang jelas ke frontend agar tidak crash
            return jsonify({
                "error": str(e),
                "mapping": {
                    "density_plot_url": "https://via.placeholder.com/300x200/343a40/ffffff?text=API+Error",
                    "heatmap_plot_url": "https://via.placeholder.com/300x200/343a40/ffffff?text=API+Error"
                },
                "temperature": {"individual_sensors": []}
            }), 500