import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import streamlit as st

# ==============================================================================
# 1. DEFINISI MODEL FIS
# ==============================================================================

# Membuat variabel input fuzzy
kualitas_tidur = ctrl.Antecedent(np.arange(0, 11, 1), 'kualitas_tidur')
beban_akademik = ctrl.Antecedent(np.arange(0, 11, 1), 'beban_akademik')
dukungan_sosial = ctrl.Antecedent(np.arange(0, 11, 1), 'dukungan_sosial')
kondisi_keuangan = ctrl.Antecedent(np.arange(0, 11, 1), 'kondisi_keuangan')
motivasi_belajar = ctrl.Antecedent(np.arange(0, 11, 1), 'motivasi_belajar')
hubungan_dosen = ctrl.Antecedent(np.arange(0, 11, 1), 'hubungan_dosen')
aktivitas_digital = ctrl.Antecedent(np.arange(0, 11, 1), 'aktivitas_digital')

# Output
tingkat_stres = ctrl.Consequent(np.arange(0, 101, 1), 'tingkat_stres')

# Fungsi Keanggotaan INPUT
for var in [kualitas_tidur, dukungan_sosial, motivasi_belajar, hubungan_dosen]:
    var['rendah'] = fuzz.trimf(var.universe, [0, 0, 4])
    var['sedang'] = fuzz.trimf(var.universe, [2, 5, 8])
    var['tinggi'] = fuzz.trimf(var.universe, [6, 10, 10])

for var in [beban_akademik, aktivitas_digital]:
    var['rendah'] = fuzz.trimf(var.universe, [0, 0, 4])
    var['sedang'] = fuzz.trimf(var.universe, [2, 5, 8])
    var['tinggi'] = fuzz.trimf(var.universe, [6, 10, 10])

kondisi_keuangan['buruk'] = fuzz.trimf(kondisi_keuangan.universe, [0, 0, 4])
kondisi_keuangan['cukup'] = fuzz.trimf(kondisi_keuangan.universe, [2, 5, 8])
kondisi_keuangan['baik'] = fuzz.trimf(kondisi_keuangan.universe, [6, 10, 10])

# Fungsi Keanggotaan OUTPUT
tingkat_stres['normal'] = fuzz.trimf(tingkat_stres.universe, [0, 0, 40])
tingkat_stres['sedang'] = fuzz.trimf(tingkat_stres.universe, [30, 50, 70])
tingkat_stres['tinggi'] = fuzz.trimf(tingkat_stres.universe, [60, 100, 100])

# ==============================================================================
# 2. DEFINISI ATURAN & SISTEM KONTROL
# ==============================================================================

rule1 = ctrl.Rule(kualitas_tidur['rendah'] & beban_akademik['tinggi'], tingkat_stres['tinggi'])
rule2 = ctrl.Rule(dukungan_sosial['tinggi'] & motivasi_belajar['tinggi'], tingkat_stres['normal'])
rule3 = ctrl.Rule(aktivitas_digital['tinggi'] & kualitas_tidur['rendah'], tingkat_stres['tinggi'])
rule4 = ctrl.Rule(kondisi_keuangan['buruk'] & beban_akademik['tinggi'], tingkat_stres['tinggi'])
rule5 = ctrl.Rule(hubungan_dosen['tinggi'] & motivasi_belajar['tinggi'], tingkat_stres['normal'])
rule6 = ctrl.Rule(kualitas_tidur['tinggi'] & dukungan_sosial['tinggi'], tingkat_stres['normal'])
rule7 = ctrl.Rule(motivasi_belajar['rendah'] & aktivitas_digital['tinggi'], tingkat_stres['tinggi'])
rule8 = ctrl.Rule(dukungan_sosial['rendah'] & beban_akademik['tinggi'], tingkat_stres['tinggi'])
rule9 = ctrl.Rule(kondisi_keuangan['baik'] & motivasi_belajar['tinggi'], tingkat_stres['normal'])
rule10 = ctrl.Rule(kualitas_tidur['sedang'] & beban_akademik['sedang'], tingkat_stres['sedang'])
rule11 = ctrl.Rule(dukungan_sosial['sedang'] & kondisi_keuangan['cukup'], tingkat_stres['sedang'])
rule12 = ctrl.Rule(aktivitas_digital['sedang'] & kualitas_tidur['sedang'], tingkat_stres['sedang'])

stres_ctrl = ctrl.ControlSystem([
    rule1, rule2, rule3, rule4, rule5, rule6,
    rule7, rule8, rule9, rule10, rule11, rule12
])

stres_simulasi = ctrl.ControlSystemSimulation(stres_ctrl)


# ==============================================================================
# 3. FUNGSI VISUALISASI INPUT (Dipanggil Sekali)
# ==============================================================================

def plot_all_antecedents():
    """Membuat plot semua fungsi keanggotaan input."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    funcs = [
        (kualitas_tidur, "Kualitas Tidur"), (beban_akademik, "Beban Akademik"),
        (dukungan_sosial, "Dukungan Sosial"), (kondisi_keuangan, "Kondisi Keuangan"),
        (motivasi_belajar, "Motivasi Belajar"), (hubungan_dosen, "Hubungan Dosen"),
        (aktivitas_digital, "Aktivitas Digital"),
    ]

    for i, (func, name) in enumerate(funcs):
        ax = axes[i]
        for label in func.terms:
            ax.plot(func.universe, func[label].mf, label=label)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Nilai Input (0-10)", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True)
    
    for i in range(len(funcs), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    return fig


# ==============================================================================
# 4. INTERFACE STREAMLIT UTAMA (MENGGUNAKAN KOLOM)
# ==============================================================================

st.set_page_config(
    page_title="Sistem FIS Deteksi Stres Mahasiswa",
    layout="wide"
)

st.title("🧠 Sistem Cerdas FIS Deteksi Tingkat Stres Mahasiswa")
st.markdown("---")

# Membuat dua kolom utama
col_input, col_output = st.columns([1, 2]) # Input 1 bagian, Output 2 bagian

# --- KOLOM INPUT (KIRI) ---
with col_input:
    st.header("Masukkan Variabel Input (0-10)")
    
    # Input Sliders
    tidur = st.slider("1. Kualitas Tidur (0=Buruk, 10=Baik)", 0, 10, 1, 1)
    beban = st.slider("2. Beban Akademik (0=Ringan, 10=Berat)", 0, 10, 1, 1)
    sosial = st.slider("3. Dukungan Sosial (0=Rendah, 10=Tinggi)", 0, 10, 5, 1)
    uang = st.slider("4. Kondisi Keuangan (0=Buruk, 10=Baik)", 0, 10, 1, 1)
    motivasi = st.slider("5. Motivasi Belajar (0=Rendah, 10=Tinggi)", 0, 10, 4, 1)
    dosen = st.slider("6. Hubungan Dosen (0=Buruk, 10=Baik)", 0, 10, 8, 1)
    digital = st.slider("7. Aktivitas Digital (0=Rendah, 10=Tinggi)", 0, 10, 10, 1)

    # Tampilkan Plot Fungsi Keanggotaan di bawah input
    with st.expander("Lihat Plot Fungsi Keanggotaan Input", expanded=False):
        st.pyplot(plot_all_antecedents())
        plt.close(plot_all_antecedents())


# --- KOLOM OUTPUT (KANAN) ---
with col_output:
    st.header("📊 Hasil Analisis Fuzzy")
    
    # 1. Memasukkan nilai ke simulasi
    stres_simulasi.input['kualitas_tidur'] = tidur
    stres_simulasi.input['beban_akademik'] = beban
    stres_simulasi.input['dukungan_sosial'] = sosial
    stres_simulasi.input['kondisi_keuangan'] = uang
    stres_simulasi.input['motivasi_belajar'] = motivasi
    stres_simulasi.input['hubungan_dosen'] = dosen
    stres_simulasi.input['aktivitas_digital'] = digital

    # 2. Jalankan FIS
    try:
        stres_simulasi.compute()
        crisp_result = stres_simulasi.output['tingkat_stres']
    except ValueError:
        st.error("⚠️ Perhitungan FIS Gagal! Tidak ada aturan fuzzy yang terpicu. Coba ubah kombinasi input.")
        st.stop()
        
    # 3. Tentukan Kategori
    if crisp_result < 30:
        kategori = "NORMAL (Risiko Rendah)"
        st.success(f"Tingkat Stres: {kategori}")
    elif crisp_result < 70:
        kategori = "SEDANG (Perlu Perhatian)"
        st.warning(f"Tingkat Stres: {kategori}")
    else:
        kategori = "TINGGI (Risiko Berat)"
        st.error(f"Tingkat Stres: {kategori}")

    # 4. Tampilkan Detail Perhitungan
    st.subheader(f"Nilai Crisp: {crisp_result:.2f}")
    
    # 5. Tampilkan Plot Output Fuzzy
    st.subheader("Visualisasi Output Fuzzy")
    
    # Periksa dan tampilkan plot
    fig_output, ax_output = plt.subplots(figsize=(7, 5))
    tingkat_stres.view(sim=stres_simulasi, ax=ax_output)
    ax_output.set_title(f"Output Fuzzy: {kategori} (Crisp: {crisp_result:.2f})", fontsize=12)
    st.pyplot(fig_output) # Streamlit menampilkan plot Matplotlib dengan benar
    plt.close(fig_output) # Tutup figure untuk membebaskan memori
    
    # Tampilkan Ringkasan Input di Kolom Output (opsional)
    st.markdown("---")
    st.write(f"**Ringkasan Input:** Tidur: {tidur}, Beban: {beban}, Sosial: {sosial}, Uang: {uang}, Motivasi: {motivasi}, Dosen: {dosen}, Digital: {digital}")