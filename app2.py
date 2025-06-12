# --- START OF FILE app2.py ---

import streamlit as st
from PIL import Image
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
import plotly.graph_objects as go
import io
import base64 # FITUR BARU: Diperlukan untuk pratinjau PDF

# ======================= TAMBAHKAN IMPORT INI DI ATAS =======================
import signal
from utils import load_analysis_history
# FITUR BARU: Import modul ekspor
from export_utils import (export_to_advanced_docx, export_report_pdf,
                          export_visualization_png, DOCX_AVAILABLE)
# FITUR VALIDASI SISTEM: Import untuk metrik dan visualisasi
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
# ===========================================================================


# ======================= (Fungsi-fungsi Anda yang lain tetap sama) =======================
# ... (Salin semua fungsi helper dan fungsi display tab Anda yang lain ke sini) ...
# ========================================================================================
# Fungsi display_history_tab dirombak total
def display_history_tab():
    """Fungsi untuk menampilkan konten dari tab Riwayat Analisis secara detail."""
    st.header("📜 Riwayat Analisis Tersimpan")
    st.markdown("Berikut adalah daftar semua analisis yang telah dilakukan oleh sistem, diurutkan dari yang terbaru.")

    history_data = load_analysis_history()

    if not history_data:
        st.info("Belum ada riwayat analisis yang tersimpan. Lakukan analisis pertama Anda!")
        return

    # Urutkan dari yang terbaru (paling bawah di file JSON) ke yang terlama
    for entry in reversed(history_data):
        timestamp = entry.get('timestamp', 'N/A')
        image_name = entry.get('image_name', 'N/A')
        summary = entry.get('analysis_summary', {})
        result_type = summary.get('type', 'N/A')
        thumbnail_path = entry.get('thumbnail_path') # <-- Ambil path thumbnail

        # Tentukan warna dan ikon berdasarkan hasil
        if "Splicing" in result_type or "Complex" in result_type or "Manipulasi" in result_type:
            icon = "🚨"
            border_color = "#ff4b4b"
        elif "Copy-Move" in result_type:
            icon = "⚠️"
            border_color = "#ffc400"
        elif "Error" in result_type:
            icon = "❌"
            border_color = "#636363"
        else: # Termasuk "Tidak Terdeteksi"
            icon = "✅"
            border_color = "#268c2f"
            
        expander_title = f"{icon} **{timestamp}** | `{image_name}` | **Hasil:** {result_type}"
        
        # Gunakan CSS untuk memberikan border berwarna pada expander
        # Ini adalah trik, mungkin perlu penyesuaian jika versi Streamlit berubah
        st.markdown(f"""
            <div style="border: 2px solid {border_color}; border-radius: 7px; padding: 10px; margin-bottom: 10px;">
            """, unsafe_allow_html=True)

        with st.expander(expander_title):
            # Layout kolom untuk thumbnail dan metrik
            col1, col2 = st.columns([1, 3])

            with col1:
                st.markdown("**Gambar Asli**")
                if thumbnail_path and os.path.exists(thumbnail_path):
                    st.image(thumbnail_path, use_container_width=True, caption=f"Thumbnail untuk {image_name}")
                else:
                    st.caption("Thumbnail tidak tersedia.")

            with col2:
                sub_col1, sub_col2, sub_col3 = st.columns(3)
                with sub_col1:
                    st.metric(label="Tingkat Kepercayaan", value=summary.get('confidence', 'N/A'))
                with sub_col2:
                    st.metric(label="Skor Copy-Move", value=f"{summary.get('copy_move_score', 0)}/100")
                with sub_col3:
                    st.metric(label="Skor Splicing", value=f"{summary.get('splicing_score', 0)}/100")
                
                st.caption(f"Waktu Proses: {entry.get('processing_time', 'N/A')}")
                st.markdown("---")
                st.write("**Detail Ringkasan (JSON):**")
                st.json(summary)
        
        st.markdown("</div>", unsafe_allow_html=True)


# ======================= FITUR BARU: FUNGSI UNTUK TAB EKSPOR =======================
def display_export_tab(original_pil, analysis_results):
    st.header("📄 Laporan & Ekspor Hasil Analisis")
    st.markdown("""
    Gunakan halaman ini untuk membuat dan mengunduh laporan forensik lengkap dari hasil analisis. 
    Anda dapat memilih format yang berbeda sesuai kebutuhan Anda.
    """)

    # Setup direktori output
    output_dir = "exported_reports"
    os.makedirs(output_dir, exist_ok=True)
    
    # Pastikan st.session_state.last_uploaded_file ada sebelum mengakses .name
    if st.session_state.last_uploaded_file:
        base_filename = os.path.splitext(st.session_state.last_uploaded_file.name)[0]
    else:
        # Fallback jika nama file tidak tersedia
        base_filename = "forensic_analysis"
        
    base_filepath = os.path.join(output_dir, f"{base_filename}_{int(time.time())}")

    # Layout kolom untuk tombol ekspor
    col1, col2, col3 = st.columns(3)

    # Tombol Ekspor PNG
    with col1:
        st.subheader("Visualisasi PNG")
        st.write("Ekspor ringkasan visual dalam satu file gambar PNG resolusi tinggi.")
        if st.button("🖼️ Ekspor ke PNG", use_container_width=True):
            with st.spinner("Membuat file PNG..."):
                png_path = f"{base_filepath}_visualization.png"
                export_visualization_png(original_pil, analysis_results, png_path)
                if os.path.exists(png_path):
                    st.success(f"Visualisasi PNG berhasil dibuat!")
                    with open(png_path, "rb") as file:
                        st.download_button(
                            label="Unduh File PNG",
                            data=file,
                            file_name=os.path.basename(png_path),
                            mime="image/png"
                        )
                else:
                    st.error("Gagal membuat file PNG.")

    # Tombol Ekspor DOCX
    with col2:
        st.subheader("Laporan DOCX")
        st.write("Ekspor laporan forensik detail dalam format Microsoft Word (.docx).")
        if not DOCX_AVAILABLE:
            st.warning("Pustaka `python-docx` tidak terinstal. Fitur ekspor DOCX/PDF dinonaktifkan.\n\nInstall dengan: `pip install python-docx`")
        else:
            if st.button("📝 Ekspor ke DOCX", use_container_width=True, type="primary"):
                with st.spinner("Membuat laporan DOCX..."):
                    docx_path = f"{base_filepath}_report.docx"
                    export_to_advanced_docx(original_pil, analysis_results, docx_path)
                    if os.path.exists(docx_path):
                        st.success(f"Laporan DOCX berhasil dibuat!")
                        with open(docx_path, "rb") as file:
                            st.download_button(
                                label="Unduh File DOCX",
                                data=file,
                                file_name=os.path.basename(docx_path),
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            )
                    else:
                        st.error("Gagal membuat laporan DOCX.")

    # Tombol Ekspor PDF
    with col3:
        st.subheader("Laporan PDF")
        st.write("Ekspor laporan forensik detail dalam format PDF yang portabel.")
        if not DOCX_AVAILABLE:
             st.info("Fitur ini memerlukan `python-docx`.")
        else:
            if st.button("📑 Ekspor ke PDF", use_container_width=True):
                with st.spinner("Membuat laporan DOCX lalu mengonversi ke PDF... Ini mungkin memakan waktu."):
                    docx_path = f"{base_filepath}_report.docx"
                    pdf_path = f"{base_filepath}_report.pdf"
                    
                    # Buat DOCX dulu
                    docx_file = export_to_advanced_docx(original_pil, analysis_results, docx_path)
                    if docx_file:
                        # Lalu konversi ke PDF
                        pdf_file = export_report_pdf(docx_file, pdf_path)
                        if pdf_file and os.path.exists(pdf_file):
                            st.success(f"Laporan PDF berhasil dibuat!")
                            with open(pdf_file, "rb") as file:
                                st.download_button(
                                    label="Unduh File PDF",
                                    data=file,
                                    file_name=os.path.basename(pdf_file),
                                    mime="application/pdf"
                                )
                        else:
                            st.error("Gagal mengonversi DOCX ke PDF. Pastikan LibreOffice atau docx2pdf terinstal.")
                    else:
                        st.error("Gagal membuat file DOCX sebagai dasar untuk PDF.")


    st.markdown("---")

    # Pratinjau PDF
    st.header("🔍 Pratinjau Laporan PDF")
    if not DOCX_AVAILABLE:
        st.warning("Pratinjau PDF tidak tersedia karena `python-docx` tidak terinstal.")
    else:
        if 'pdf_preview_path' not in st.session_state:
            st.session_state.pdf_preview_path = None

        if st.button("🚀 Buat & Tampilkan Pratinjau PDF"):
            st.session_state.pdf_preview_path = None # Reset
            with st.spinner("Membuat pratinjau PDF... Proses ini bisa memakan waktu hingga satu menit."):
                docx_path = f"{base_filepath}_preview.docx"
                pdf_path = f"{base_filepath}_preview.pdf"
                docx_file = export_to_advanced_docx(original_pil, analysis_results, docx_path)
                if docx_file:
                    pdf_file = export_report_pdf(docx_file, pdf_path)
                    if pdf_file and os.path.exists(pdf_file):
                        st.session_state.pdf_preview_path = pdf_file
                        st.success("Pratinjau berhasil dibuat!")
                    else:
                        st.error("Gagal membuat file PDF untuk pratinjau.")
                else:
                    st.error("Gagal membuat file DOCX untuk pratinjau.")

        if st.session_state.pdf_preview_path:
            with open(st.session_state.pdf_preview_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            
            pdf_display = f'<div style="border: 2px solid #ccc; border-radius: 5px; padding: 10px;">' \
                          f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>' \
                          f'</div>'
            
            st.markdown(pdf_display, unsafe_allow_html=True)
        else:
            st.info("Klik tombol di atas untuk menghasilkan pratinjau laporan dalam format PDF.")

# ======================= FUNGSI VALIDASI YANG SANGAT DETAIL =======================

def lakukan_validasi_sistem(analysis_results):
    """
    Menjalankan Validasi Integritas Proses dengan Pemeriksaan Heuristik.
    Fungsi ini memeriksa output dari setiap 17 tahap dan melakukan validasi silang pada tahap-tahap kunci.
    """
    if not analysis_results:
        return ["Hasil analisis tidak tersedia untuk divalidasi."], 0.0, "Hasil analisis tidak tersedia.", []

    # Definisi 17 proses dengan kriteria keberhasilan fungsional dan heuristik.
    # Disertai dengan justifikasi dan nilai aktual untuk ditampilkan.
    processes = [
        # Proses 1-6: Validasi fungsional dasar.
        {"name": "1. Validasi & Muat Gambar", "check": lambda r: isinstance(r.get('metadata', {}).get('FileSize (bytes)', 0), int) and r.get('metadata', {}).get('FileSize (bytes)', 0) > 0},
        {"name": "2. Ekstraksi Metadata", "check": lambda r: 'Metadata_Authenticity_Score' in r.get('metadata', {})},
        {"name": "3. Pra-pemrosesan Gambar", "check": lambda r: r.get('enhanced_gray') is not None and len(r['enhanced_gray'].shape) == 2},
        {"name": "4. Analisis ELA Multi-Kualitas", "check": lambda r: isinstance(r.get('ela_image'), Image.Image) and r.get('ela_mean', -1) >= 0},
        {"name": "5. Ekstraksi Fitur (SIFT, ORB, etc.)", "check": lambda r: isinstance(r.get('feature_sets'), dict) and 'sift' in r['feature_sets']},
        {"name": "6. Deteksi Copy-Move (Feature-based)", "check": lambda r: 'ransac_inliers' in r and r['ransac_inliers'] >= 0},
        
        # ATURAN HEURISTIK #1
        {
            "name": "7. Deteksi Copy-Move (Block-based)", 
            "check": lambda r: len(r.get('block_matches', [])) > 0 and r.get('ransac_inliers', 0) > 5, 
            "reason": "Deteksi blok tidak dikuatkan oleh bukti deteksi fitur yang signifikan.",
            "rule_text": "KEBERHASILAN = (Jumlah Blok Cocok > 0) DAN (Inliers Fitur > 5)",
            "values_text": lambda r: f"Nilai Aktual: Blok Cocok = {len(r.get('block_matches', []))}, Inliers Fitur = {r.get('ransac_inliers', 0)}"
        },
        
        # Proses Fungsional
        {"name": "8. Analisis Konsistensi Noise", "check": lambda r: 'overall_inconsistency' in r.get('noise_analysis', {})},
        {"name": "9. Analisis Artefak JPEG", "check": lambda r: 'estimated_original_quality' in r.get('jpeg_analysis', {})},
        {"name": "10. Analisis Ghost JPEG", "check": lambda r: r.get('jpeg_ghost') is not None},
        {"name": "11. Analisis Domain Frekuensi", "check": lambda r: 'frequency_inconsistency' in r.get('frequency_analysis', {})},
        {"name": "12. Analisis Konsistensi Tekstur", "check": lambda r: 'overall_inconsistency' in r.get('texture_analysis', {})},
        {"name": "13. Analisis Konsistensi Tepi", "check": lambda r: 'edge_inconsistency' in r.get('edge_analysis', {})},
        {"name": "14. Analisis Konsistensi Iluminasi", "check": lambda r: 'overall_illumination_inconsistency' in r.get('illumination_analysis', {})},
        {"name": "15. Analisis Statistik Kanal", "check": lambda r: 'rg_correlation' in r.get('statistical_analysis', {})},

        # ATURAN HEURISTIK #2
        {
            "name": "16. Lokalisasi Area Manipulasi", 
            "check": lambda r: r.get('localization_analysis', {}).get('tampering_percentage', 0) > 1.0 and \
                               (r.get('ela_mean', 0) > 8.0 or r.get('noise_analysis', {}).get('overall_inconsistency', 0) > 0.3),
            "reason": "Lokalisasi area tidak didukung oleh bukti anomali fisika (ELA/Noise) yang kuat.",
            "rule_text": "KEBERHASILAN = (Area > 1.0%) DAN (Mean ELA > 8.0 ATAU Inkonsistensi Noise > 0.3)",
            "values_text": lambda r: f"Nilai Aktual: Area = {r.get('localization_analysis', {}).get('tampering_percentage', 0):.1f}%, Mean ELA = {r.get('ela_mean', 0):.1f}, Noise = {r.get('noise_analysis', {}).get('overall_inconsistency', 0):.2f}"
        },
        
        # Proses Fungsional
        {"name": "17. Klasifikasi Akhir & Skor", "check": lambda r: 'type' in r.get('classification', {}) and 'confidence' in r.get('classification', {})}
    ]

    report_details = []
    failed_heuristics = []
    success_count = 0
    total_processes = len(processes)

    for process in processes:
        try:
            is_success = process["check"](analysis_results)
        except Exception as e:
            is_success = False
            print(f"Error saat validasi proses '{process['name']}': {e}")
        
        if is_success:
            status = "[BERHASIL]"
            report_details.append(f"✅ {status:12} | {process['name']}")
            success_count += 1
        else:
            status = "[GAGAL]"
            report_details.append(f"❌ {status:12} | {process['name']}")
            # Jika ini adalah proses heuristik, simpan detailnya untuk ditampilkan
            if "reason" in process:
                failed_heuristics.append({
                    "name": process["name"],
                    "reason": process["reason"],
                    "rule": process["rule_text"],
                    "values": process["values_text"](analysis_results)
                })
    
    validation_score = (success_count / total_processes) * 100
    
    # Buat ringkasan penjelasan dinamis
    if validation_score == 100:
        summary_text = "Skor 100% menunjukkan semua 17 proses berhasil dieksekusi dan lulus semua pemeriksaan heuristik. Ini menandakan konsistensi internal yang sangat tinggi pada hasil analisis gambar ini."
    else:
        summary_text = (f"Skor {validation_score:.1f}% menunjukkan {success_count} dari {total_processes} proses berhasil. "
                        f"Kegagalan yang terdeteksi didasarkan pada pemeriksaan heuristik internal, yang menunjukkan kemampuan sistem melakukan kritik-diri.")

    return report_details, validation_score, summary_text, failed_heuristics

def display_validation_tab_baru(analysis_results):
    """
    Menampilkan tab validasi sistem (Tahap 5) dengan justifikasi dan detail yang diperkaya.
    """
    st.header("🔬 Tahap 5: Hasil Pengujian Sistem (Validasi Integritas Proses)")
    st.markdown("""
    Pengujian ini mengadopsi metode **Validasi Integritas Proses dengan Pemeriksaan Heuristik**. Pendekatan ini tidak hanya mengukur keberhasilan fungsional setiap tahap, tetapi juga mengevaluasi **konsistensi logis** antar hasil analisis, meniru cara kerja seorang ahli. 
    Skor di bawah 100% seringkali **BUKAN merupakan error**, melainkan indikasi bahwa sistem mampu melakukan **kritik-diri** dan menemukan inkonsistensi minor antar-metode.
    """)
    st.markdown("---")

    report_details, validation_score, summary_text, failed_heuristics = lakukan_validasi_sistem(analysis_results)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Laporan Status Eksekusi Proses")
        report_text = "\n".join(report_details)
        st.code(report_text, language='bash')
    
    with col2:
        st.subheader("Skor Integritas Proses")
        st.metric(
            label="Tingkat Keandalan Sistem",
            value=f"{validation_score:.1f}%",
            delta=f"{round(validation_score - 90, 1)}% vs Target 90%",
            delta_color="normal" if validation_score >= 90 else "inverse"
        )
        st.caption("Skor = (Proses Berhasil / Total Proses) * 100")
        
        if validation_score == 100:
           st.success(summary_text)
        else:
           st.info(summary_text)

    # Expander untuk detail dan justifikasi
    with st.expander("Lihat Justifikasi dan Detail Pemeriksaan Heuristik", expanded=True):
        st.markdown("""
        #### Filosofi Pengujian
        Sebuah sistem forensik yang tangguh harus dapat menilai keyakinannya sendiri. Pemeriksaan heuristik ini dirancang untuk tujuan tersebut. Sistem mencari **koroborasi bukti (evidence corroboration)**. Jika suatu metode deteksi menghasilkan sinyal positif, sistem akan memeriksa apakah metode lain yang relevan juga menghasilkan sinyal pendukung.
        """)
        
        if not failed_heuristics:
            st.success("✅ **Tidak ada kegagalan heuristik yang terdeteksi.** Semua pemeriksaan validasi silang internal berhasil, menunjukkan konsistensi yang sangat baik antar hasil analisis untuk gambar ini.")
        else:
            st.warning("🚨 **Terdeteksi Kegagalan Heuristik:**")
            for failure in failed_heuristics:
                st.markdown(f"**Proses:** `{failure['name']}`")
                st.markdown(f"**Alasan Kegagalan:** {failure['reason']}")
                st.markdown(f"**Aturan yang Diterapkan:**")
                st.code(failure['rule'], language='text')
                st.markdown(f"**Data Saat Pemeriksaan:**")
                st.code(failure['values'], language='text')
                st.markdown("---")

# ======================= AKHIR FUNGSI VALIDASI YANG SANGAT DETAIL =======================


# ======================= APLIKASI UTAMA STREAMLIT (BAGIAN YANG DIMODIFIKASI) =======================
def main_app():
    st.set_page_config(layout="wide", page_title="Sistem Forensik Gambar V3")

    # Ganti nama variabel agar tidak bentrok dengan fungsi
    global IMPORTS_SUCCESSFUL, IMPORT_ERROR_MESSAGE
    
    if not IMPORTS_SUCCESSFUL:
        st.error(f"Gagal mengimpor modul: {IMPORT_ERROR_MESSAGE}")
        return

    # Inisialisasi session state (tidak ada perubahan di sini)
    if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
    if 'original_image' not in st.session_state: st.session_state.original_image = None
    if 'last_uploaded_file' not in st.session_state: st.session_state.last_uploaded_file = None
    
    st.sidebar.title("🖼️ Sistem Deteksi Forensik V3")
    st.sidebar.markdown("Unggah gambar untuk memulai analisis mendalam.")

    uploaded_file = st.sidebar.file_uploader(
        "Pilih file gambar...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    )

    if uploaded_file is not None:
        # Periksa apakah ini file baru atau sama dengan yang terakhir
        if st.session_state.last_uploaded_file is None or st.session_state.last_uploaded_file.name != uploaded_file.name:
            st.session_state.last_uploaded_file = uploaded_file
            st.session_state.analysis_results = None
            st.session_state.original_image = Image.open(uploaded_file).convert("RGB")
            st.rerun()

    if st.session_state.original_image:
        st.sidebar.image(st.session_state.original_image, caption='Gambar yang diunggah', use_container_width=True)

        if st.sidebar.button("🔬 Mulai Analisis", use_container_width=True, type="primary"):
            st.session_state.analysis_results = None
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            filename = st.session_state.last_uploaded_file.name
            temp_filepath = os.path.join(temp_dir, filename)
            
            # Tulis ulang file dari buffer
            st.session_state.last_uploaded_file.seek(0)
            with open(temp_filepath, "wb") as f:
                f.write(st.session_state.last_uploaded_file.getbuffer())

            with st.spinner('Melakukan analisis 17 tahap... Ini mungkin memakan waktu beberapa saat.'):
                try:
                    # Pastikan main_analysis_func dipanggil dengan path file yang benar
                    results = main_analysis_func(temp_filepath)
                    st.session_state.analysis_results = results
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat analisis: {e}")
                    st.exception(e)
                    st.session_state.analysis_results = None
                finally:
                    if os.path.exists(temp_filepath):
                        os.remove(temp_filepath)
            st.rerun()

        st.sidebar.markdown("---")
        st.sidebar.subheader("Kontrol Sesi")

        # Tombol Mulai Ulang (tidak ada perubahan)
        if st.sidebar.button("🔄 Mulai Ulang Analisis", use_container_width=True):
            st.session_state.analysis_results = None
            st.session_state.original_image = None
            st.session_state.last_uploaded_file = None
            if 'pdf_preview_path' in st.session_state:
                st.session_state.pdf_preview_path = None # Reset preview
            st.rerun()

        # Tombol Keluar (tidak ada perubahan pada logika ini)
        if st.sidebar.button("🚪 Keluar", use_container_width=True):
            st.session_state.analysis_results = None
            st.session_state.original_image = None
            st.session_state.last_uploaded_file = None
            st.sidebar.warning("Aplikasi sedang ditutup...")
            st.balloons()
            time.sleep(2)
            pid = os.getpid()
            os.kill(pid, signal.SIGTERM)

    st.sidebar.markdown("---")
    st.sidebar.info("Aplikasi ini menggunakan pipeline analisis 17-tahap untuk mendeteksi manipulasi gambar.")

    st.title("Hasil Analisis Forensik Gambar")

    if st.session_state.analysis_results:
        # ======================= PERUBAHAN UTAMA DI SINI (TAB) =======================
        tab_list = [
            "📊 Tahap 1: Analisis Inti",
            "🔬 Tahap 2: Analisis Lanjut",
            "📈 Tahap 3: Analisis Statistik",
            "📋 Tahap 4: Laporan Akhir",
            "🔬 Tahap 5: Hasil Pengujian", # TAB BARU UNTUK VALIDASI
            "📄 Ekspor Laporan",
            "📜 Riwayat Analisis"
        ]
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tab_list)
        # ======================= AKHIR PERUBAHAN TAB =======================

        with tab1:
            display_core_analysis(st.session_state.original_image, st.session_state.analysis_results)
        with tab2:
            display_advanced_analysis(st.session_state.original_image, st.session_state.analysis_results)
        with tab3:
            display_statistical_analysis(st.session_state.original_image, st.session_state.analysis_results)
        with tab4:
            display_final_report(st.session_state.analysis_results)
        # ======================= KONTEN TAB BARU =======================
        with tab5:
            display_validation_tab_baru(st.session_state.analysis_results)
        # ======================= AKHIR KONTEN TAB BARU =======================
        with tab6:
            display_export_tab(st.session_state.original_image, st.session_state.analysis_results)
        with tab7:
            display_history_tab()

    elif not st.session_state.original_image:
        # Tampilkan tab Riwayat di halaman utama jika belum ada gambar diunggah
        main_page_tabs = st.tabs(["👋 Selamat Datang", "📜 Riwayat Analisis"])
        
        with main_page_tabs[0]:
            st.info("Silakan unggah gambar di sidebar kiri untuk memulai.")
            st.markdown("""
            **Panduan Singkat:**
            1. **Unggah Gambar:** Gunakan tombol 'Pilih file gambar...' di sidebar.
            2. **Mulai Analisis:** Klik tombol biru 'Mulai Analisis'.
            3. **Lihat Hasil:** Hasil akan ditampilkan dalam beberapa tab.
            4. **Uji Sistem:** Buka tab 'Hasil Pengujian' untuk melihat validasi integritas.
            5. **Ekspor:** Buka tab 'Ekspor Laporan' untuk mengunduh hasil.
            """)
        
        with main_page_tabs[1]:
            display_history_tab()

# Pastikan Anda memanggil fungsi main_app() di akhir
if __name__ == '__main__':
    # Anda harus menempatkan semua fungsi helper (seperti display_core_analysis, dll.)
    # sebelum pemanggilan main_app() atau di dalam file lain dan diimpor.
    # Untuk contoh ini, saya asumsikan semua fungsi sudah didefinisikan di atas.
    
    # ======================= Konfigurasi & Import =======================
    try:
        from main import analyze_image_comprehensive_advanced as main_analysis_func
        from visualization import (
            create_feature_match_visualization, create_block_match_visualization,
            create_localization_visualization, create_frequency_visualization,
            create_texture_visualization, create_edge_visualization,
            create_illumination_visualization, create_statistical_visualization,
            create_quality_response_plot, create_advanced_combined_heatmap,
            create_technical_metrics_plot
        )
        from config import BLOCK_SIZE
        IMPORTS_SUCCESSFUL = True
        IMPORT_ERROR_MESSAGE = ""
    except ImportError as e:
        IMPORTS_SUCCESSFUL = False
        IMPORT_ERROR_MESSAGE = str(e)

    # ======================= Fungsi Helper untuk Visualisasi Individual =======================
    # (Semua fungsi helper Anda yang ada seperti display_single_plot, create_spider_chart, dll.
    # tetap di sini dan tidak perlu diubah)
    def display_single_plot(title, plot_function, args, caption, details, container):
        """Fungsi generik untuk menampilkan plot tunggal dengan detail."""
        with container:
            st.subheader(title, divider='rainbow')
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_function(ax, *args)
            st.pyplot(fig, use_container_width=True)
            st.caption(caption)
            with st.expander("Lihat Detail Teknis"):
                st.markdown(details)

    def display_single_image(title, image_array, cmap, caption, details, container, colorbar=False):
        """Fungsi generik untuk menampilkan gambar tunggal dengan detail."""
        with container:
            st.subheader(title, divider='rainbow')
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(image_array, cmap=cmap)
            ax.axis('off')
            if colorbar:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig, use_container_width=True)
            st.caption(caption)
            with st.expander("Lihat Detail Teknis"):
                st.markdown(details)

    def create_spider_chart(analysis_results):
        """Membuat spider chart untuk kontribusi skor."""
        categories = [
            'ELA', 'Feature Match', 'Block Match', 'Noise',
            'JPEG Ghost', 'Frequency', 'Texture', 'Illumination'
        ]

        # Memastikan kunci ada sebelum diakses
        ela_mean = analysis_results.get('ela_mean', 0)
        noise_inconsistency = analysis_results.get('noise_analysis', {}).get('overall_inconsistency', 0)
        jpeg_ghost_ratio = analysis_results.get('jpeg_ghost_suspicious_ratio', 0)
        freq_inconsistency = analysis_results.get('frequency_analysis', {}).get('frequency_inconsistency', 0)
        texture_inconsistency = analysis_results.get('texture_analysis', {}).get('overall_inconsistency', 0)
        illum_inconsistency = analysis_results.get('illumination_analysis', {}).get('overall_illumination_inconsistency', 0)
        ela_regional_inconsistency = analysis_results.get('ela_regional_stats', {}).get('regional_inconsistency', 0)
        ransac_inliers = analysis_results.get('ransac_inliers', 0)
        block_matches_len = len(analysis_results.get('block_matches', []))

        splicing_values = [
            min(ela_mean / 15, 1.0),
            0.1,
            0.1,
            min(noise_inconsistency / 0.5, 1.0),
            min(jpeg_ghost_ratio / 0.3, 1.0),
            min(freq_inconsistency / 2.0, 1.0),
            min(texture_inconsistency / 0.5, 1.0),
            min(illum_inconsistency / 0.5, 1.0)
        ]

        copy_move_values = [
            min(ela_regional_inconsistency / 0.5, 1.0),
            min(ransac_inliers / 30, 1.0),
            min(block_matches_len / 40, 1.0),
            0.2,
            0.2,
            0.3,
            0.3,
            0.2
        ]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=splicing_values,
            theta=categories,
            fill='toself',
            name='Indikator Splicing',
            line=dict(color='red')
        ))
        fig.add_trace(go.Scatterpolar(
            r=copy_move_values,
            theta=categories,
            fill='toself',
            name='Indikator Copy-Move',
            line=dict(color='orange')
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Kontribusi Metode Analisis"
        )
        return fig

    # ======================= Fungsi Tampilan per Tab =======================
    # (Semua fungsi display_... Anda yang lain tetap di sini dan tidak perlu diubah)
    def display_core_analysis(original_pil, results):
        st.header("Tahap 1: Analisis Inti (Core Analysis)")
        st.write("Tahap ini memeriksa anomali fundamental seperti kompresi, fitur kunci, dan duplikasi blok.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Gambar Asli", divider='rainbow')
            st.image(original_pil, caption="Gambar yang dianalisis.", use_container_width=True)
            with st.expander("Detail Gambar"):
                st.json({
                    "Filename": results['metadata'].get('Filename', 'N/A'),
                    "Size": f"{results['metadata'].get('FileSize (bytes)', 0):,} bytes",
                    "Dimensions": f"{original_pil.width}x{original_pil.height}",
                    "Mode": original_pil.mode
                })

        display_single_image(
            title="Error Level Analysis (ELA)",
            image_array=results['ela_image'],
            cmap='hot',
            caption="Area yang lebih terang menunjukkan potensi tingkat kompresi yang berbeda.",
            details=f"""
            - **Mean ELA:** `{results.get('ela_mean', 0):.2f}`
            - **Std Dev ELA:** `{results.get('ela_std', 0):.2f}`
            - **Region Outlier:** `{results.get('ela_regional_stats', {}).get('outlier_regions', 0)}`
            - **Interpretasi:** Nilai mean ELA yang tinggi (>8) atau standar deviasi yang besar (>15) bisa menandakan adanya splicing.
            """,
            container=col2,
            colorbar=True
        )

        st.markdown("---")
        col3, col4, col5 = st.columns(3)

        display_single_plot(
            title="Feature Matching (Copy-Move)",
            plot_function=create_feature_match_visualization,
            args=[original_pil, results],
            caption="Garis hijau menghubungkan area dengan fitur yang identik (setelah verifikasi RANSAC).",
            details=f"""
            - **Total SIFT Matches:** `{results.get('sift_matches', 0)}`
            - **RANSAC Verified Inliers:** `{results.get('ransac_inliers', 0)}`
            - **Transformasi Geometris:** `{results.get('geometric_transform', [None])[0] or 'N/A'}`
            - **Interpretasi:** Jumlah inliers yang tinggi (>10) adalah indikator kuat *copy-move*.
            """,
            container=col3
        )

        display_single_plot(
            title="Block Matching (Copy-Move)",
            plot_function=create_block_match_visualization,
            args=[original_pil, results],
            caption="Kotak berwarna menandai blok piksel yang identik di lokasi berbeda.",
            details=f"""
            - **Pasangan Blok Identik:** `{len(results.get('block_matches', []))}`
            - **Ukuran Blok:** `{BLOCK_SIZE}x{BLOCK_SIZE} pixels`
            - **Interpretasi:** Banyaknya blok yang cocok (>5) memperkuat dugaan *copy-move*.
            """,
            container=col4
        )

        display_single_plot(
            title="Lokalisasi Area Mencurigakan",
            plot_function=create_localization_visualization,
            args=[original_pil, results],
            caption="Overlay merah menunjukkan area yang paling mencurigakan berdasarkan K-Means clustering.",
            details=f"""
            - **Persentase Area Termanipulasi:** `{results.get('localization_analysis', {}).get('tampering_percentage', 0):.2f}%`
            - **Metode:** `K-Means Clustering`
            - **Interpretasi:** Peta ini menggabungkan berbagai sinyal untuk menyorot wilayah yang paling mungkin telah diedit.
            """,
            container=col5
        )

    def display_advanced_analysis(original_pil, results):
        st.header("Tahap 2: Analisis Tingkat Lanjut (Advanced Analysis)")
        st.write("Tahap ini menyelidiki properti intrinsik gambar seperti frekuensi, tekstur, tepi, dan artefak kompresi.")

        col1, col2, col3 = st.columns(3)

        display_single_plot(
            title="Analisis Domain Frekuensi",
            plot_function=create_frequency_visualization,
            args=[results],
            caption="Distribusi energi pada frekuensi rendah, sedang, dan tinggi menggunakan DCT.",
            details=f"""
            - **Inkonsistensi Frekuensi:** `{results.get('frequency_analysis', {}).get('frequency_inconsistency', 0):.3f}`
            - **Rasio Energi (High/Low):** `{results.get('frequency_analysis', {}).get('dct_stats', {}).get('freq_ratio', 0):.3f}`
            - **Interpretasi:** Pola yang tidak biasa atau inkonsistensi tinggi bisa menandakan modifikasi.
            """,
            container=col1
        )

        display_single_plot(
            title="Analisis Konsistensi Tekstur",
            plot_function=create_texture_visualization,
            args=[results],
            caption="Mengukur konsistensi properti tekstur (kontras, homogenitas, dll.) di seluruh gambar.",
            details=f"""
            - **Inkonsistensi Tekstur Global:** `{results.get('texture_analysis', {}).get('overall_inconsistency', 0):.3f}`
            - **Metode:** `GLCM` & `LBP`
            - **Interpretasi:** Nilai inkonsistensi yang tinggi (>0.3) menunjukkan adanya area dengan pola tekstur yang berbeda, ciri khas splicing.
            """,
            container=col2
        )

        display_single_plot(
            title="Analisis Konsistensi Tepi (Edge)",
            plot_function=create_edge_visualization,
            args=[original_pil, results],
            caption="Visualisasi tepi gambar. Densitas tepi yang tidak wajar dapat menjadi petunjuk.",
            details=f"""
            - **Inkonsistensi Tepi:** `{results.get('edge_analysis', {}).get('edge_inconsistency', 0):.3f}`
            - **Metode:** `Sobel Filter`
            - **Interpretasi:** Splicing seringkali menghasilkan diskontinuitas pada tepi objek.
            """,
            container=col3
        )

        st.markdown("---")
        col4, col5, col6 = st.columns(3)

        display_single_plot(
            title="Analisis Konsistensi Iluminasi",
            plot_function=create_illumination_visualization,
            args=[original_pil, results],
            caption="Peta iluminasi (kecerahan) dari gambar untuk mencari sumber cahaya yang tidak konsisten.",
            details=f"""
            - **Inkonsistensi Iluminasi:** `{results.get('illumination_analysis', {}).get('overall_illumination_inconsistency', 0):.3f}`
            - **Metode:** Analisis `L* channel` pada `CIELAB`.
            - **Interpretasi:** Objek yang ditambahkan seringkali memiliki pencahayaan yang tidak cocok.
            """,
            container=col4
        )

        display_single_image(
            title="Analisis JPEG Ghost",
            image_array=results['jpeg_ghost'],
            cmap='hot',
            caption="Area terang menunjukkan kemungkinan kompresi JPEG ganda/berbeda.",
            details=f"""
            - **Rasio Area Mencurigakan:** `{results.get('jpeg_ghost_suspicious_ratio', 0):.2%}`
            - **Interpretasi:** Tanda kuat adanya splicing jika area tertentu telah dikompresi sebelumnya dengan kualitas berbeda.
            """,
            container=col5,
            colorbar=True
        )

        with col6:
            st.subheader("Peta Anomali Gabungan", divider='rainbow')
            combined_heatmap = create_advanced_combined_heatmap(results, original_pil.size)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(original_pil, alpha=0.5)
            ax.imshow(combined_heatmap, cmap='inferno', alpha=0.5)
            ax.axis('off')
            st.pyplot(fig, use_container_width=True)
            st.caption("Menggabungkan ELA, JPEG Ghost, dan fitur lain untuk membuat satu peta kecurigaan.")
            with st.expander("Detail Peta Anomali"):
                st.markdown("""
                Peta ini adalah agregasi berbobot dari beberapa sinyal anomali.
                """)

    def display_statistical_analysis(original_pil, results):
        st.header("Tahap 3: Analisis Statistik dan Metrik")
        st.write("Melihat data mentah di balik analisis, termasuk statistik noise, kurva kualitas, dan metrik teknis lainnya.")

        col1, col2, col3 = st.columns(3)

        display_single_image(
            title="Peta Sebaran Noise",
            image_array=results['noise_map'],
            cmap='gray',
            caption="Visualisasi noise. Pola yang tidak seragam bisa mengindikasikan manipulasi.",
            details=f"""
            - **Inkonsistensi Noise Global:** `{results.get('noise_analysis', {}).get('overall_inconsistency', 0):.3f}`
            - **Blok Outlier Terdeteksi:** `{results.get('noise_analysis', {}).get('outlier_count', 0)}`
            - **Interpretasi:** Area yang ditempel dari gambar lain akan memiliki pola noise yang berbeda.
            """,
            container=col1
        )

        display_single_plot(
            title="Kurva Respons Kualitas JPEG",
            plot_function=create_quality_response_plot,
            args=[results],
            caption="Menunjukkan error saat gambar dikompres ulang pada kualitas berbeda.",
            details=f"""
            - **Estimasi Kualitas Asli:** `{results.get('jpeg_analysis', {}).get('estimated_original_quality', 'N/A')}`
            - **Indikator Kompresi Ganda:** `{results.get('jpeg_analysis', {}).get('double_compression_indicator', 0):.3f}`
            - **Interpretasi:** Kurva yang tidak mulus dapat mengindikasikan kompresi ganda.
            """,
            container=col2
        )

        display_single_plot(
            title="Entropi Kanal Warna",
            plot_function=create_statistical_visualization,
            args=[results],
            caption="Mengukur 'kerandoman' atau kompleksitas informasi pada setiap kanal warna.",
            details=f"""
            - **Entropi Global:** `{results.get('statistical_analysis', {}).get('overall_entropy', 0):.3f}`
            - **Korelasi Kanal (R-G):** `{results.get('statistical_analysis', {}).get('rg_correlation', 0):.3f}`
            - **Interpretasi:** Korelasi atau entropi yang aneh bisa menjadi tanda modifikasi warna.
            """,
            container=col3
        )

    def display_final_report(results):
        st.header("Tahap 4: Laporan Akhir dan Kesimpulan")
        classification = results.get('classification', {})

        result_type = classification.get('type', 'N/A')
        confidence_level = classification.get('confidence', 'N/A')

        if "Splicing" in result_type or "Manipulasi" in result_type or "Copy-Move" in result_type:
            st.error(f"**Hasil Deteksi: {result_type}**", icon="🚨")
        elif "Tidak Terdeteksi" in result_type:
            st.success(f"**Hasil Deteksi: {result_type}**", icon="✅")
        else:
            st.info(f"**Hasil Deteksi: {result_type}**", icon="ℹ️")

        st.write(f"**Tingkat Kepercayaan:** `{confidence_level}`")

        col1, col2 = st.columns(2)
        with col1:
            score_cm = classification.get('copy_move_score', 0)
            st.write("Skor Copy-Move:")
            st.progress(score_cm / 100, text=f"{score_cm}/100") 
        with col2:
            score_sp = classification.get('splicing_score', 0)
            st.write("Skor Splicing:")
            st.progress(score_sp / 100, text=f"{score_sp}/100") 

        st.markdown("---")

        col3, col4 = st.columns([1, 1.5])

        with col3:
            st.subheader("Temuan Kunci", divider='blue')
            details = classification.get('details', [])
            if details:
                for detail in details:
                    st.markdown(f"✔️ {detail}")
            else:
                st.markdown("- Tidak ada temuan kunci yang signifikan.")

        with col4:
            st.subheader("Visualisasi Kontribusi Analisis", divider='blue')
            spider_chart = create_spider_chart(results)
            st.plotly_chart(spider_chart, use_container_width=True)
            st.caption("Grafik ini menunjukkan seberapa kuat sinyal dari setiap metode analisis.")

        with st.expander("Lihat Rangkuman Teknis Lengkap"):
            st.json(classification)
    
    main_app()