import pickle
import time
import numpy as np
import pandas as pd
import streamlit as st

# Konfigurasi Halaman
st.set_page_config(
    page_title="Apex Vehicle Intelligence | Financial Dashboard",
    page_icon="🚘",
    layout="wide"
)

# --- CUSTOM CSS: BACKGROUND, INPUT STYLING & HIGH CONTRAST LABELS ---
st.markdown("""
    <style>
    /* Background Image dengan Overlay Profesional */
    .stApp {
        background-image: linear-gradient(rgba(3, 7, 18, 0.88), rgba(15, 23, 42, 0.92)), 
                          url("https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?q=80&w=1920&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit Default Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Styling Kartu Parameter */
    .parameter-card {
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        margin-bottom: 20px;
    }

    /* Memperjelas Nama Kolom (Label) Menjadi Putih Terang dan Tebal */
    .parameter-card label, div[data-testid="stNumberInput"] label, div[data-testid="stSelectbox"] label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }

    /* Mengubah Warna Latar Belakang Kotak Input Agar Tidak Abu-Abu Kusam */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #0b0f19 !important;
        color: #ffffff !important;
        border: 1px solid rgba(6, 182, 212, 0.4) !important;
        border-radius: 10px !important;
    }

    /* Metric Card Professional Styling dengan Visibilitas Tinggi */
    .stMetric {
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 20px;
        border-radius: 14px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .stMetric:hover {
        border-color: rgba(6, 182, 212, 0.6);
        transform: translateY(-2px);
    }
    .stMetric label {
        color: #94a3b8 !important;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700;
        font-size: 1.6rem !important;
    }

    /* Gradient Button dalam Form */
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 14px;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stFormSubmitButton > button:hover {
        opacity: 0.95;
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(6, 182, 212, 0.6);
    }

    /* Typography */
    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-weight: 700;
        letter-spacing: -0.025em;
    }
    </style>
""", unsafe_allow_html=True)

# Memuat model Machine Learning dengan cache
@st.cache_resource
def load_model():
    return pickle.load(open('estimasi_mobil.sav', 'rb'))

model = load_model()

# --- HEADER UTAMA ---
st.markdown("""
    <div style="padding: 5px 0px 15px 0px;">
        <div style="display: inline-flex; align-items: center; background: rgba(6, 182, 212, 0.15); border: 1px solid rgba(6, 182, 212, 0.4); padding: 4px 12px; border-radius: 20px; margin-bottom: 12px;">
            <span style="height: 8px; width: 8px; background-color: #22d3ee; border-radius: 50%; display: inline-block; margin-right: 8px; box-shadow: 0 0 8px #22d3ee;"></span>
            <span style="color: #22d3ee; font-size: 11px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;">AI VALUATION ENGINE ACTIVE</span>
        </div>
        <h1 style="margin: 0px; font-size: 2.3rem;">Apex Vehicle Intelligence</h1>
        <p style="color: #94a3b8; font-size: 1rem; margin-top: 6px;">Sistem valuasi harga kendaraan profesional berbasis regresi prediktif real-time.</p>
    </div>
""", unsafe_allow_html=True)

# --- KAMUS KURS MATA UANG ---
exchange_rates = {
    "IDR (Rp)": {"rate": 19000.0, "symbol": "Rp", "format": "{:,.0f}"},
    "EUR (€)": {"rate": 1.0, "symbol": "€", "format": "{:,.2f}"},
    "USD ($)": {"rate": 1.08, "symbol": "$", "format": "{:,.2f}"},
    "GBP (£)": {"rate": 0.85, "symbol": "£", "format": "{:,.2f}"},
    "SGD (S$)": {"rate": 1.45, "symbol": "S$", "format": "{:,.2f}"},
    "JPY (¥)": {"rate": 165.0, "symbol": "¥", "format": "{:,.0f}"},
    "MYR (RM)": {"rate": 5.10, "symbol": "RM", "format": "{:,.2f}"}
}

# --- INISIALISASI SESSION STATE UNTUK HASIL PREDIKSI ---
if 'harga_eur' not in st.session_state:
    st.session_state.harga_eur = 0.0
if 'has_calculated' not in st.session_state:
    st.session_state.has_calculated = False

# --- TATA LETAK UTAMA: 2 KOLOM (KIRI = KARTU PARAMETER, KANAN = DASHBOARD HASIL) ---
input_col, main_col = st.columns([1, 2.2], gap="large")

with input_col:
    st.markdown("""
        <div class="parameter-card">
            <h3 style="margin-top: 0px; font-size: 1.25rem;">🎛️ Parameter Kendaraan</h3>
            <p style='font-size: 12px; color: #94a3b8; margin-top:-6px; margin-bottom: 15px;'>Masukkan spesifikasi teknis mobil</p>
    """, unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        year = st.number_input('Tahun Pembuatan', min_value=1990, max_value=2026, value=2002, step=1)
        mileage = st.number_input('Jarak Tempuh (Km)', min_value=0, value=50000, step=1000)
        tax = st.number_input('Pajak Tahunan (Tax)', min_value=0, value=145, step=5)
        mpg = st.number_input('Efisiensi Bahan Bakar (MPG)', min_value=0.0, value=30.2, step=0.1)
        engineSize = st.number_input('Kapasitas Mesin (Liter)', min_value=0.0, max_value=10.0, value=2.0, step=0.1)

        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        target_currency = st.selectbox(
            "💱 Mata Uang Utama Display", 
            ["IDR (Rp)", "EUR (€)", "USD ($)", "GBP (£)", "SGD (S$)", "JPY (¥)", "MYR (RM)"]
        )
        
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button('Hitung Estimasi Harga')
        
    st.markdown("</div>", unsafe_allow_html=True)

# Eksekusi kalkulasi prediksi saat tombol di dalam form diklik
if submit_btn:
    with st.spinner('Memproses kalkulasi matriks pasar...'):
        time.sleep(0.3)
        try:
            input_data = pd.DataFrame([[year, mileage, tax, mpg, engineSize]], 
                                      columns=['year', 'mileage', 'tax', 'mpg', 'engineSize'])
            
            predict = model.predict(input_data)
            st.session_state.harga_eur = float(predict[0])
            st.session_state.has_calculated = True
        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")

harga_eur = st.session_state.harga_eur
has_calculated = st.session_state.has_calculated

selected_info = exchange_rates[target_currency]
harga_utama = harga_eur * selected_info["rate"]

with main_col:
    # Baris 1: Kartu Metrik Utama
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            label=f"Valuasi Utama ({target_currency})", 
            value=f"{selected_info['symbol']} {selected_info['format'].format(harga_utama)}"
        )
    with m2:
        st.metric(
            label="Base Value (EUR)", 
            value=f"€ {harga_eur:,.2f}"
        )
    with m3:
        status_label = "Selesai Dikalkulasi" if has_calculated else "Menunggu Parameter"
        st.metric(
            label="Status Engine", 
            value=status_label
        )

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

    # Petunjuk / Info Status
    if not has_calculated:
        st.markdown("""
            <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(255, 255, 255, 0.1); border-left: 4px solid #06b6d4; padding: 16px; border-radius: 12px; margin-bottom: 25px;">
                <h4 style="margin: 0px 0px 4px 0px; color: #22d3ee !important; font-size: 14px;">Petunjuk Operasional</h4>
                <p style="margin: 0px; color: #94a3b8; font-size: 13px;">Atur spesifikasi kendaraan di panel kiri, lalu klik tombol <b>Hitung Estimasi Harga</b> untuk memicu kalkulasi valuasi AI.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(255, 255, 255, 0.1); border-left: 4px solid #3b82f6; padding: 16px; border-radius: 12px; margin-bottom: 25px;">
                <h4 style="margin: 0px 0px 4px 0px; color: #60a5fa !important; font-size: 14px;">Analisis Valuasi Berhasil</h4>
                <p style="margin: 0px; color: #94a3b8; font-size: 13px;">Model berhasil mengevaluasi nilai kendaraan berdasarkan tren harga pasar dan atribut teknis yang dimasukkan.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

    # Layout Bagian Bawah: Kiri (Matriks Multi-Mata Uang), Kanan (Grafik Perbandingan)
    sub_left, sub_right = st.columns([1.2, 1], gap="medium")

    with sub_left:
        st.markdown("### 🌍 Matriks Multi-Mata Uang")
        st.markdown("<p style='color: #94a3b8; font-size: 12px; margin-bottom: 15px;'>Konversi harga real-time terkelompok berdasarkan wilayah.</p>", unsafe_allow_html=True)
        
        # Kelompok 1: Lokal & Utama
        st.markdown("<p style='color: #38bdf8; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;'>📍 Lokal & Utama</p>", unsafe_allow_html=True)
        sub1, sub2 = st.columns(2)
        with sub1:
            val_idr = harga_eur * exchange_rates["IDR (Rp)"]["rate"]
            st.metric(label="IDR (Rp)", value=f"Rp {val_idr:,.0f}")
        with sub2:
            val_usd = harga_eur * exchange_rates["USD ($)"]["rate"]
            st.metric(label="USD ($)", value=f"$ {val_usd:,.2f}")

        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        
        # Kelompok 2: Internasional
        st.markdown("<p style='color: #38bdf8; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;'>🇪🇺 🇬🇧 Internasional</p>", unsafe_allow_html=True)
        sub3, sub4 = st.columns(2)
        with sub3:
            val_eur = harga_eur * exchange_rates["EUR (€)"]["rate"]
            st.metric(label="EUR (€)", value=f"€ {val_eur:,.2f}")
        with sub4:
            val_gbp = harga_eur * exchange_rates["GBP (£)"]["rate"]
            st.metric(label="GBP (£)", value=f"£ {val_gbp:,.2f}")

        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

        # Kelompok 3: Regional Asia
        st.markdown("<p style='color: #38bdf8; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;'>🌏 Regional Asia</p>", unsafe_allow_html=True)
        sub5, sub6, sub7 = st.columns(3)
        with sub5:
            val_sgd = harga_eur * exchange_rates["SGD (S$)"]["rate"]
            st.metric(label="SGD (S$)", value=f"S$ {val_sgd:,.2f}")
        with sub6:
            val_jpy = harga_eur * exchange_rates["JPY (¥)"]["rate"]
            st.metric(label="JPY (¥)", value=f"¥ {val_jpy:,.0f}")
        with sub7:
            val_myr = harga_eur * exchange_rates["MYR (RM)"]["rate"]
            st.metric(label="MYR (RM)", value=f"RM {val_myr:,.2f}")

    with sub_right:
        st.markdown("### 📊 Grafik Perbandingan Nilai")
        st.markdown("<p style='color: #94a3b8; font-size: 12px; margin-bottom: 15px;'>Visualisasi perbandingan estimasi harga dalam mata uang utama.</p>", unsafe_allow_html=True)
        
        chart_data = pd.DataFrame({
            'Mata Uang': ["IDR (Juta)", "USD ($)", "EUR (€)", "GBP (£)", "SGD ($)"],
            'Estimasi': [
                (harga_eur * exchange_rates["IDR (Rp)"]["rate"]) / 1000000, 
                harga_eur * exchange_rates["USD ($)"]["rate"],
                harga_eur * exchange_rates["EUR (€)"]["rate"],
                harga_eur * exchange_rates["GBP (£)"]["rate"],
                harga_eur * exchange_rates["SGD (S$)"]["rate"]
            ]
        }).set_index('Mata Uang')
        
        st.bar_chart(chart_data, color="#06b6d4")
        st.markdown("<p style='font-size: 11px; color: #64748b; text-align: center; margin-top: 5px;'>*Catatan: Nilai IDR pada grafik disederhanakan dalam satuan Juta Rupiah.</p>", unsafe_allow_html=True)

# Footer / Informasi Sistem Ringkas di Bagian Bawah
st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
with st.expander("🛡️ Informasi Arsitektur & Spesifikasi Sistem"):
    st.markdown("""
- **Prediktif Engine:** Regresi Linier Terlatih yang dimuat dari artefak `estimasi_mobil.sav`.
- **Desain Antarmuka:** Kartu parameter sisi kiri dengan label putih kontras tinggi dan kotak input gelap modern.
- **Modul Konversi:** Kalkulator kurs mata uang global yang tersusun rapi berdasarkan kategori wilayah (Lokal, Internasional, dan Regional Asia).
    """)