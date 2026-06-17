import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="GeoDiversitas - Dashboard Diversifikasi Pangan",
    page_icon="🌾",
    layout="wide"
)

# Title
st.title("🌾 GeoDiversitas: Dashboard Diversifikasi Bahan Pangan")
st.markdown("### Pemetaan Potensi Diversifikasi Berbasis Machine Learning dan Geospasial")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    ml_data = pd.read_csv('dashboard_data/ml_data.csv')
    ketahanan = pd.read_csv('dashboard_data/ketahanan_pangan.csv')
    komoditas = pd.read_csv('dashboard_data/komoditas_potensial.csv')
    importance = pd.read_csv('dashboard_data/feature_importance.csv')
    return ml_data, ketahanan, komoditas, importance

@st.cache_resource
def load_models():
    with open('dashboard_data/rf_model.pkl', 'rb') as f:
        rf = pickle.load(f)
    return rf

# Cek apakah file data ada
try:
    ml_data, ketahanan, komoditas, importance = load_data()
    rf_model = load_models()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False
    st.error("""
    ⚠️ **Data tidak ditemukan!**
    
    Jalankan notebook terlebih dahulu untuk generate data di folder `dashboard_data/`.
    
    Pastikan folder `dashboard_data/` berisi:
    - ml_data.csv
    - ketahanan_pangan.csv
    - komoditas_potensial.csv
    - feature_importance.csv
    - rf_model.pkl
    """)

if data_loaded:
    # =====================================================
    # TAB 1: OVERVIEW
    # =====================================================
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", 
        "🗺️ Pemetaan Potensi", 
        "🤖 Prediksi & Rekomendasi",
        "📈 Simulasi Kebijakan"
    ])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Nilai Ekspor", f"Rp {komoditas['Nilai_Ekspor_Juta_USD'].sum():,.0f} Juta")
        with col2:
            st.metric("Jumlah Komoditas", len(komoditas))
        with col3:
            st.metric("Provinsi Jawa", 6)
        with col4:
            # Ambil akurasi model terbaik
            st.metric("Model Terbaik", "Random Forest")
        
        st.subheader("🏆 Top 10 Komoditas Ekspor Potensial")
        fig = px.bar(
            komoditas.head(10),
            x='Nilai_Ekspor_Juta_USD',
            y='Komoditas',
            orientation='h',
            color='Nilai_Ekspor_Juta_USD',
            color_continuous_scale='Viridis',
            title='Nilai Ekspor Komoditas (Juta US$)'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Feature Importance")
            fig2 = px.bar(
                importance.head(10),
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='Plasma',
                title='Top 10 Faktor Penentu Diversifikasi'
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            st.subheader("🏷️ Status Ketahanan Pangan")
            status_count = ketahanan['Tingkat_Ketahanan'].value_counts().reset_index()
            status_count.columns = ['Tingkat', 'Jumlah']
            fig3 = px.pie(
                status_count,
                values='Jumlah',
                names='Tingkat',
                color='Tingkat',
                color_discrete_map={
                    'Sangat Rendah': '#e74c3c',
                    'Rendah': '#f39c12',
                    'Sedang': '#f1c40f',
                    'Tinggi': '#2ecc71'
                },
                title='Distribusi Tingkat Ketahanan Pangan'
            )
            st.plotly_chart(fig3, use_container_width=True)

    # =====================================================
    # TAB 2: PEMETAAN POTENSI
    # =====================================================
    
    with tab2:
        st.subheader("🗺️ Peta Potensi Diversifikasi per Provinsi")
        
        # Pilih provinsi
        provinsi = st.selectbox("Pilih Provinsi", ml_data['Provinsi'].unique())
        
        # Data provinsi
        prov_data = ml_data[ml_data['Provinsi'] == provinsi].iloc[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Indeks Ketahanan Pangan", f"{prov_data['Indeks_Ketahanan_Pangan']:.3f}")
            st.metric("Jumlah Komoditas", f"{prov_data['Jumlah_Komoditas']}")
            st.metric("Potensi Ekspor", "✅ Tinggi" if prov_data['Potensi_Ekspor'] == 1 else "❌ Rendah")
        
        with col2:
            # Rekomendasi komoditas
            st.subheader("📋 Rekomendasi Komoditas")
            
            # Ambil data produksi provinsi
            prov_produksi = {}
            for col in ml_data.columns:
                if 'Produksi' in col:
                    nama = col.replace('Produksi ', '')
                    prov_produksi[nama] = prov_data[col]
            
            # Filter yang memiliki produksi > 0
            rekomendasi = {k: v for k, v in prov_produksi.items() if v > 0}
            if rekomendasi:
                rekomendasi_df = pd.DataFrame({
                    'Komoditas': list(rekomendasi.keys()),
                    'Produksi (Kuintal)': list(rekomendasi.values())
                }).sort_values('Produksi (Kuintal)', ascending=False)
                
                st.dataframe(rekomendasi_df, use_container_width=True, hide_index=True)
            else:
                st.warning("Belum ada data produksi untuk provinsi ini")
        
        # Peta sederhana (menggunakan plotly)
        st.subheader("🗺️ Peta Indeks Ketahanan Pangan")
        
        # Koordinat provinsi Jawa
        koordinat = {
            'BANTEN': {'lat': -6.2, 'lon': 106.0},
            'DKI JAKARTA': {'lat': -6.2, 'lon': 106.8},
            'JAWA BARAT': {'lat': -6.9, 'lon': 107.6},
            'JAWA TENGAH': {'lat': -7.5, 'lon': 109.0},
            'DI YOGYAKARTA': {'lat': -7.8, 'lon': 110.4},
            'JAWA TIMUR': {'lat': -7.8, 'lon': 112.0}
        }
        
        map_data = ketahanan.copy()
        map_data['lat'] = map_data['Provinsi'].map(lambda x: koordinat.get(x, {}).get('lat', 0))
        map_data['lon'] = map_data['Provinsi'].map(lambda x: koordinat.get(x, {}).get('lon', 0))
        
        fig_map = px.scatter_mapbox(
            map_data,
            lat='lat',
            lon='lon',
            color='Indeks_Ketahanan_Pangan',
            size=[10]*len(map_data),
            hover_name='Provinsi',
            hover_data={'Tingkat_Ketahanan': True, 'Indeks_Ketahanan_Pangan': ':.3f'},
            color_continuous_scale='RdYlGn',
            title='Peta Indeks Ketahanan Pangan Provinsi Jawa',
            zoom=7,
            height=500
        )
        fig_map.update_layout(mapbox_style="open-street-map")
        fig_map.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)

    # =====================================================
    # TAB 3: PREDIKSI & REKOMENDASI
    # =====================================================
    
    with tab3:
        st.subheader("🤖 Prediksi Potensi Diversifikasi")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Masukkan Data Provinsi")
            
            # Input sederhana
            provinsi_input = st.selectbox("Provinsi", ml_data['Provinsi'].unique(), key='prediksi_provinsi')
            
            # Tampilkan data provinsi
            prov_data = ml_data[ml_data['Provinsi'] == provinsi_input].iloc[0]
            
            st.write(f"**Indeks Ketahanan:** {prov_data['Indeks_Ketahanan_Pangan']:.3f}")
            st.write(f"**Jumlah Komoditas:** {prov_data['Jumlah_Komoditas']}")
            
            if st.button("🔮 Prediksi Potensi Ekspor"):
                # Prepare input
                features = ml_data.drop(columns=['Provinsi', 'Potensi_Ekspor']).columns
                X_input = pd.DataFrame([prov_data[features].values], columns=features)
                X_input = X_input.fillna(0)
                
                pred = rf_model.predict(X_input)[0]
                proba = rf_model.predict_proba(X_input)[0]
                
                if pred == 1:
                    st.success(f"✅ Provinsi **{provinsi_input}** berpotensi **TINGGI** untuk diversifikasi ekspor")
                    st.info(f"Probabilitas: {proba[1]*100:.1f}%")
                else:
                    st.warning(f"⚠️ Provinsi **{provinsi_input}** berpotensi **RENDAH** untuk diversifikasi ekspor")
                    st.info(f"Probabilitas: {proba[0]*100:.1f}%")
        
        with col2:
            st.markdown("### 📋 Rekomendasi Diversifikasi")
            
            # Komoditas potensial untuk provinsi ini
            prov_komoditas = []
            for col in ml_data.columns:
                if 'Produksi' in col and prov_data[col] > 0:
                    komoditas_nama = col.replace('Produksi ', '')
                    prov_komoditas.append({
                        'Komoditas': komoditas_nama,
                        'Produksi': prov_data[col]
                    })
            
            if prov_komoditas:
                df_rekom = pd.DataFrame(prov_komoditas).sort_values('Produksi', ascending=False)
                st.dataframe(df_rekom, use_container_width=True, hide_index=True)
                
                # Rekomendasi ekspor
                st.markdown("#### 🚀 Rekomendasi Ekspor")
                top_komoditas = df_rekom.iloc[0]['Komoditas']
                st.info(f"**Prioritas Ekspor:** {top_komoditas}")
                
                # Tambahan rekomendasi
                st.markdown("#### 💡 Saran Kebijakan")
                if prov_data['Potensi_Ekspor'] == 1:
                    st.success("""
                    - ✅ Fokus pada pengembangan komoditas unggulan
                    - ✅ Tingkatkan kualitas dan volume produksi
                    - ✅ Kembangkan kemasan dan pemasaran ekspor
                    """)
                else:
                    st.warning("""
                    - ⚠️ Perlu peningkatan indeks ketahanan pangan
                    - ⚠️ Diversifikasi komoditas sayuran
                    - ⚠️ Optimasi luas panen dan produktivitas
                    """)
            else:
                st.warning("Belum ada data produksi untuk provinsi ini")

    # =====================================================
    # TAB 4: SIMULASI KEBIJAKAN
    # =====================================================
    
    with tab4:
        st.subheader("📈 Simulasi Kebijakan Diversifikasi")
        
        st.markdown("""
        ### Skenario What-If Analysis
        Simulasikan dampak kebijakan diversifikasi terhadap:
        1. **Peningkatan nilai ekspor provinsi Jawa (target: 15-20%)**
        2. **Pengurangan ketergantungan impor**
        3. **Peningkatan indeks ketahanan pangan nasional**
        """)
        
        # Input skenario
        col1, col2 = st.columns(2)
        
        with col1:
            target_ekspor = st.slider(
                "Target Peningkatan Ekspor (%)", 
                min_value=5, 
                max_value=30, 
                value=15,
                step=5
            )
            
            komoditas_target = st.selectbox(
                "Komoditas Prioritas",
                komoditas['Komoditas'].unique()
            )
        
        with col2:
            investasi = st.selectbox(
                "Skema Investasi",
                ["Agroindustri", "Infrastruktur", "Teknologi", "Kombinasi"]
            )
        
        if st.button("🚀 Jalankan Simulasi"):
            # Simulasi sederhana
            st.subheader("📊 Hasil Simulasi")
            
            # Hitung dampak
            nilai_ekspor_saat_ini = komoditas['Nilai_Ekspor_Juta_USD'].sum()
            nilai_ekspor_baru = nilai_ekspor_saat_ini * (1 + target_ekspor/100)
            
            # Dampak terhadap ketahanan pangan
            rata_rata_ketahanan = ketahanan['Indeks_Ketahanan_Pangan'].mean()
            peningkatan_ketahanan = rata_rata_ketahanan * (1 + target_ekspor/200)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Nilai Ekspor", 
                    f"Rp {nilai_ekspor_baru:,.0f} Juta",
                    delta=f"+{target_ekspor}%"
                )
            
            with col2:
                st.metric(
                    "Indeks Ketahanan Pangan",
                    f"{peningkatan_ketahanan:.3f}",
                    delta=f"{(peningkatan_ketahanan - rata_rata_ketahanan):.3f}"
                )
            
            with col3:
                st.metric(
                    "Komoditas Prioritas",
                    komoditas_target,
                    "Rekomendasi"
                )
            
            # Grafik dampak
            st.subheader("📈 Proyeksi Dampak Kebijakan")
            
            # Data simulasi
            skenario = ['Saat Ini', 'Simulasi']
            nilai_ekspor = [nilai_ekspor_saat_ini, nilai_ekspor_baru]
            
            fig_sim = px.bar(
                x=skenario,
                y=nilai_ekspor,
                color=skenario,
                title=f'Dampak Peningkatan Ekspor {target_ekspor}%',
                labels={'x': 'Skenario', 'y': 'Nilai Ekspor (Juta US$)'}
            )
            st.plotly_chart(fig_sim, use_container_width=True)
            
            # Rekomendasi kebijakan
            st.subheader("📋 Rekomendasi Kebijakan")
            
            if target_ekspor >= 20:
                st.success("""
                **Strategi Agresif:**
                - 🚀 Percepatan sertifikasi ekspor
                - 🚀 Kerjasama B2B dengan buyer internasional
                - 🚀 Insentif pajak untuk eksportir
                """)
            elif target_ekspor >= 15:
                st.info("""
                **Strategi Moderat:**
                - 📈 Peningkatan kualitas produk
                - 📈 Pengembangan kemasan dan branding
                - 📈 Pelatihan petani dan pelaku usaha
                """)
            else:
                st.warning("""
                **Strategi Konservatif:**
                - 📊 Stabilisasi produksi
                - 📊 Penguatan pasar domestik
                - 📊 Persiapan infrastruktur ekspor
                """)
    
    # Footer
    st.markdown("---")
    st.caption("🌾 GeoDiversitas - Dashboard Diversifikasi Bahan Pangan | Satria Data 2026")