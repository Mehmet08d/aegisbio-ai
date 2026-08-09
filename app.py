import streamlit as st
from Bio import Entrez, SeqIO
from Bio.Align import PairwiseAligner
from Bio.Align import substitution_matrices
import plotly.express as px
import pandas as pd
import requests
import py3Dmol
from st3dmol import showmol
import openai

# --- SAYFA KONFİGÜRASYONU ---
st.set_page_config(
    page_title="Mgen Analysis | Enterprise Bio-Suite & De-Extinction Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ÜNİVERSİTE DÜZEYİ GLASSMORPHISM UI TASARIMI ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stCodeBlock, code {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .main { 
        background: radial-gradient(circle at 50% 0%, #0f172a 0%, #020617 100%);
        color: #f8fafc; 
    }
    
    section[data-testid="stSidebar"] { 
        background-color: rgba(15, 23, 42, 0.88) !important; 
        backdrop-filter: blur(20px); 
        border-right: 1px solid rgba(56, 189, 248, 0.15); 
    }

    div[data-testid="stMetric"] { 
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%) !important; 
        backdrop-filter: blur(12px); 
        border: 1px solid rgba(148, 163, 184, 0.15); 
        border-radius: 16px !important; 
        padding: 20px !important; 
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
    }
    
    div[data-testid="stMetric"]:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transition: all 0.3s ease;
    }

    .stButton>button { 
        width: 100%; 
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%); 
        color: #ffffff !important; 
        font-weight: 700 !important; 
        letter-spacing: 0.5px;
        border-radius: 12px !important; 
        height: 50px !important; 
        border: 1px solid rgba(255,255,255,0.1) !important; 
        box-shadow: 0 4px 20px rgba(2, 132, 199, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(2, 132, 199, 0.5);
    }

    .brand-header {
        text-align: center;
        padding: 15px 0 25px 0;
    }
    
    .brand-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }

    .academic-badge {
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #38bdf8;
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# BLOSUM62 MATRİSİ VE AMİNO ASİT BİYOFİZİK PARAMETRELERİ
blosum62 = substitution_matrices.load("BLOSUM62")

AMINO_ACID_PROPERTIES = {
    'A': {'charge': 0, 'hydrophobicity': 1.8, 'size': 88.6, 'helix_propensity': 1.42},
    'R': {'charge': 1, 'hydrophobicity': -4.5, 'size': 173.4, 'helix_propensity': 0.98},
    'N': {'charge': 0, 'hydrophobicity': -3.5, 'size': 114.1, 'helix_propensity': 0.67},
    'D': {'charge': -1, 'hydrophobicity': -3.5, 'size': 111.1, 'helix_propensity': 1.01},
    'C': {'charge': 0, 'hydrophobicity': 2.5, 'size': 108.5, 'helix_propensity': 0.70},
    'Q': {'charge': 0, 'hydrophobicity': -3.5, 'size': 143.8, 'helix_propensity': 1.11},
    'E': {'charge': -1, 'hydrophobicity': -3.5, 'size': 138.4, 'helix_propensity': 1.51},
    'G': {'charge': 0, 'hydrophobicity': -0.4, 'size': 60.1, 'helix_propensity': 0.57},
    'H': {'charge': 1, 'hydrophobicity': -3.2, 'size': 153.2, 'helix_propensity': 1.00},
    'I': {'charge': 0, 'hydrophobicity': 4.5, 'size': 166.7, 'helix_propensity': 1.08},
    'L': {'charge': 0, 'hydrophobicity': 3.8, 'size': 166.7, 'helix_propensity': 1.21},
    'K': {'charge': 1, 'hydrophobicity': -3.9, 'size': 168.6, 'helix_propensity': 1.16},
    'M': {'charge': 0, 'hydrophobicity': 1.9, 'size': 162.9, 'helix_propensity': 1.45},
    'F': {'charge': 0, 'hydrophobicity': 2.8, 'size': 189.9, 'helix_propensity': 1.13},
    'P': {'charge': 0, 'hydrophobicity': -1.6, 'size': 112.7, 'helix_propensity': 0.57},
    'S': {'charge': 0, 'hydrophobicity': -0.8, 'size': 89.0, 'helix_propensity': 0.77},
    'T': {'charge': 0, 'hydrophobicity': -0.7, 'size': 116.1, 'helix_propensity': 0.83},
    'W': {'charge': 0, 'hydrophobicity': -0.9, 'size': 227.8, 'helix_propensity': 1.08},
    'Y': {'charge': 0, 'hydrophobicity': -1.3, 'size': 193.6, 'helix_propensity': 0.69},
    'V': {'charge': 0, 'hydrophobicity': 4.2, 'size': 140.0, 'helix_propensity': 1.06}
}

# --- GENİŞLETİLMİŞ TÜR VERİ TABANI ---
HAZIR_TURLER = {
    "🦘 Numbat (Myrmecobius fasciatus) [Taşıyıcı Konak]": "NC_011949.1",
    "🐅 Tasman Kaplanı / Thylacine (Thylacinus cynocephalus) [Tükenmiş]": "NC_028319.1",
    "🦘 Kızıl Kanguru (Macropus rufus)": "NC_023522.1",
    "🐨 Koala (Phascolarctos cinereus)": "NC_021975.1",
    "🐘 Asya Fili (Elephas maximus) [Taşıyıcı Konak]": "NC_005129.2",
    "🐘 Afrika Savan Fili (Loxodonta africana)": "NC_000934.1",
    "🦣 Yünlü Mamut (Mammuthus primigenius) [Tükenmiş]": "NC_007596.2",
    "🦣 Kolombiya Mamutu (Mammuthus columbi) [Tükenmiş]": "NC_015529.1",
    "🦣 Düz Dişli Fil (Palaeoloxodon antiquus) [Tükenmiş]": "NC_034825.1",
    "🕊️ Şeritli Güvercin (Patagioenas fasciata) [Taşıyıcı Konak]": "NC_028503.1",
    "🕊️ Yolcu Güvercini (Ectopistes migratorius) [Tükenmiş]": "NC_026723.1",
    "🕊️ Kaya Güvercini (Columba livia)": "NC_013978.1",
    "🦤 Dodo Kuşu (Raphus cucullatus) [Tükenmiş]": "NC_008263.1",
    "🦅 İmparator Penguen (Aptenodytes forsteri)": "NC_023091.1",
    "🦅 Gökdoğan Şahini (Falco peregrinus)": "NC_029348.1",
    "👤 Modern İnsan (Homo sapiens)": "NC_012920.1",
    "🦴 Neandertal İnsanı (Homo neanderthalensis) [Tükenmiş]": "NC_011137.1",
    "🦴 Denisova İnsanı (Denisova hominin) [Tükenmiş]": "NC_013993.1",
    "🐒 Şempanze (Pan troglodytes)": "NC_001643.1",
    "🦍 Batı Gorili (Gorilla gorilla)": "NC_011120.1",
    "⚔️ Kılıç Dişli Kaplan (Smilodon fatalis) [Tükenmiş]": "NC_030752.1",
    "🦁 Aslan (Panthera leo)": "NC_028321.1",
    "🐅 Kaplan (Panthera tigris)": "NC_010642.1",
    "🐆 Kar Leoparı (Panthera uncia)": "NC_010638.1",
    "🐺 Gri Kurt (Canis lupus)": "NC_009686.1",
    "🦭 Dugong (Dugong dugon) [Taşıyıcı Konak]": "NC_003314.1",
    "🐋 Steller Deniz İneği (Hydrodamalis gigas) [Tükenmiş]": "NC_045268.1",
    "🦈 Grönland Köpekbalığı (Somniosus microcephalus)": "NC_050842.1",
    "🦔 Çıplak Kör Fare (Heterocephalus glaber)": "NC_015112.1",
    "🐻 Kutup Ayısı (Ursus maritimus)": "NC_009692.1",
    "🐻 Bozayı (Ursus arctos)": "NC_003427.1",
    "✏️ Custom (Özel NCBI Accession ID)": "CUSTOM"
}

GEN_KOD_HARITASI = {
    "🧬 TÜM 13 GENİ BİRLİKTE TARA (Toplu Analiz)": "ALL_13",
    "ND1 (Complex I)": "ND1", "ND2 (Complex I)": "ND2", "ND3 (Complex I)": "ND3",
    "ND4 (Complex I)": "ND4", "ND4L (Complex I)": "ND4L", "ND5 (Complex I)": "ND5", "ND6 (Complex I)": "ND6",
    "CYTB (Complex III)": "CYTB", "COX1 (Complex IV)": "COX1", "COX2 (Complex IV)": "COX2",
    "COX3 (Complex IV)": "COX3", "ATP6 (Complex V)": "ATP6", "ATP8 (Complex V)": "ATP8"
}

# --- BAŞLIK ---
st.markdown("""
    <div class="brand-header">
        <span class="academic-badge">Institutional Genomic Suite v4.0 Enterprise</span>
        <h1 class="brand-title">Mgen Analysis</h1>
        <p style='color: #94a3b8; font-size: 1.1rem; max-width: 800px; margin: 0 auto;'>
            Mitonuclear Incompatibility Engine, De-Extinction Genomic Analytics, ESMFold 3D Prediction & Universal Bio-LLM
        </p>
    </div>
""", unsafe_allow_html=True)

# ANA SEKMELER
tab_analiz, tab_3d_view, tab_esmfold, tab_ai_bot = st.tabs([
    "🚀 Biyoinformatik Analiz Engine", 
    "🧬 py3Dmol PDB Görselleştirici", 
    "🔮 ESMFold Anlık 3D Yapı Tahmini",
    "🤖 AI Biyo-Danışman (7/24 Aktif)"
])

# ==========================================
# AKADEMİK HESAPLAMA VE ANALİZ FONKSİYONLARI
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def ncbi_genom_getir(accession_id, email):
    Entrez.email = email
    try:
        handle = Entrez.efetch(db="nucleotide", id=accession_id, rettype="gb", retmode="text")
        record = SeqIO.read(handle, "genbank")
        handle.close()
        return record
    except Exception:
        return None

def recorddan_protein_ayikla(record, gen_adi):
    if not record:
        return None
    gen_sozlugu = {
        "ND1": ["ND1", "NAD1"], "ND2": ["ND2", "NAD2"], "ND3": ["ND3", "NAD3"],
        "ND4": ["ND4", "NAD4"], "ND4L": ["ND4L", "NAD4L"], "ND5": ["ND5", "NAD5"], "ND6": ["ND6", "NAD6"],
        "CYTB": ["CYTB", "CYTOCHROME B", "COB"], "COX1": ["COX1", "COI", "COXI"],
        "COX2": ["COX2", "COII"], "COX3": ["COX3", "COIII"], "ATP6": ["ATP6", "ATPASE 6"], "ATP8": ["ATP8", "ATPASE 8"]
    }
    hedef_etiketler = gen_sozlugu.get(gen_adi, [gen_adi])
    for feature in record.features:
        if feature.type == "CDS":
            gene_tag = feature.qualifiers.get("gene", [""])[0].upper()
            product_tag = feature.qualifiers.get("product", [""])[0].upper()
            if any(e in gene_tag or e in product_tag for e in hedef_etiketler):
                if "translation" in feature.qualifiers:
                    return feature.qualifiers["translation"][0]
    return None

def dizileri_hizala(seq1, seq2):
    try:
        aligner = PairwiseAligner()
        aligner.mode = 'global'
        alignments = aligner.align(seq1, seq2)
        if len(alignments) > 0:
            aligned = alignments[0]
            s1_aligned = "".join([seq1[i] if i is not None else "-" for i in aligned.indices[0]])
            s2_aligned = "".join([seq2[j] if j is not None else "-" for j in aligned.indices[1]])
            return s1_aligned, s2_aligned
    except Exception:
        pass
    return seq1, seq2

def get_blosum_score(aa1, aa2):
    try:
        return blosum62[(aa1, aa2)]
    except KeyError:
        try:
            return blosum62[(aa2, aa1)]
        except KeyError:
            return -1

def gen_analiz_et(gene_code, rec1, rec2, crispr_mode):
    p1 = recorddan_protein_ayikla(rec1, gene_code)
    p2 = recorddan_protein_ayikla(rec2, gene_code)
    if not p1 or not p2:
        return None

    al1, al2 = dizileri_hizala(p1, p2)
    farklar = []
    toplam_risk = 0.0

    for idx in range(min(len(al1), len(al2))):
        aa1, aa2 = al1[idx], al2[idx]
        if aa1 != "-" and aa2 != "-" and aa1 != aa2:
            pr1 = AMINO_ACID_PROPERTIES.get(aa1, {'charge': 0, 'hydrophobicity': 0, 'size': 100, 'helix_propensity': 1.0})
            pr2 = AMINO_ACID_PROPERTIES.get(aa2, {'charge': 0, 'hydrophobicity': 0, 'size': 100, 'helix_propensity': 1.0})

            yuk_f = abs(pr1['charge'] - pr2['charge'])
            hid_f = abs(pr1['hydrophobicity'] - pr2['hydrophobicity'])
            hacim_f = abs(pr1['size'] - pr2['size'])
            helix_f = abs(pr1['helix_propensity'] - pr2['helix_propensity'])
            
            blosum_skor = get_blosum_score(aa1, aa2)
            
            lokal_r = (yuk_f * 3.5) + (hid_f * 2.0) + (hacim_f * 0.03) + (helix_f * 4.0) - (blosum_skor * 0.8)
            lokal_r = max(0.1, lokal_r)
            toplam_risk += lokal_r

            grna_target = f"5'-N20-NGG ({aa1}->{aa2} @Pos {idx+1})" if crispr_mode else "N/A"

            farklar.append({
                "Gen": gene_code, 
                "Pozisyon": idx + 1, 
                "Tür 1 (AA)": aa1, 
                "Tür 2 (AA)": aa2,
                "BLOSUM62 Skoru": blosum_skor,
                "Yük Değişimi": yuk_f, 
                "Hidrofobiklik Δ": round(hid_f, 2), 
                "Heliks Yapı Δ": round(helix_f, 2),
                "Lokal Risk Skoru": round(lokal_r, 2),
                "Mito-CRISPR gRNA": grna_target
            })

    genel_risk = min(100.0, (toplam_risk / (max(len(al1), 1) * 3.5)) * 100)
    return {
        "gene": gene_code, "seq_len": len(al1), "mutations": len(farklar), 
        "risk_score": genel_risk, "farklar": farklar, "seq1": p1, "seq2": p2
    }

# ==========================================
# TAB 1: BİYOİNFORMATİK ANALİZ ENGINE
# ==========================================
with tab_analiz:
    st.sidebar.markdown("### ⚙️ Analiz Parametreleri")
    user_email = st.sidebar.text_input("NCBI E-posta", "researcher@mgenanalysis.org")

    tur1_secim = st.sidebar.selectbox("1. Tür / Taşıyıcı Konak", list(HAZIR_TURLER.keys()), index=0)
    numbat_acc = st.sidebar.text_input("1. Tür NCBI Kodu", "NC_011949.1") if HAZIR_TURLER[tur1_secim] == "CUSTOM" else HAZIR_TURLER[tur1_secim]

    tur2_secim = st.sidebar.selectbox("2. Tür / Hedef Canlı", list(HAZIR_TURLER.keys()), index=1)
    thylacine_acc = st.sidebar.text_input("2. Tür NCBI Kodu", "NC_028319.1") if HAZIR_TURLER[tur2_secim] == "CUSTOM" else HAZIR_TURLER[tur2_secim]

    target_gene_raw = st.sidebar.selectbox("İncelenecek Mitokondriyal Gen", list(GEN_KOD_HARITASI.keys()), index=0)
    crispr_mode = st.sidebar.checkbox("🧬 Mito-CRISPR gRNA Adaylarını Hesapla", value=True)
    run_button = st.sidebar.button("🚀 BİYOİNFORMATİK ANALİZİ BAŞLAT")

    if run_button:
        selected_code = GEN_KOD_HARITASI[target_gene_raw]
        
        with st.spinner("Genom verileri NCBI üzerinden çekiliyor ve BLOSUM62 matrisleri işleniyor..."):
            rec1 = ncbi_genom_getir(numbat_acc, user_email)
            rec2 = ncbi_genom_getir(thylacine_acc, user_email)

        if not rec1 or not rec2:
            st.error("❌ NCBI üzerinden GenBank kayıtları çekilemedi. Bağlantınızı veya Accession kodlarını kontrol edin.")
        else:
            if selected_code == "ALL_13":
                all_genes = ["ND1", "ND2", "ND3", "ND4", "ND4L", "ND5", "ND6", "CYTB", "COX1", "COX2", "COX3", "ATP6", "ATP8"]
                all_results, all_farklar = [], []

                for g in all_genes:
                    res = gen_analiz_et(g, rec1, rec2, crispr_mode)
                    if res:
                        all_results.append(res)
                        all_farklar.extend(res["farklar"])

                if all_results:
                    tot_mutations = sum(r["mutations"] for r in all_results)
                    avg_risk = sum(r["risk_score"] for r in all_results) / len(all_results)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Analiz Edilen Gen", f"{len(all_results)} / 13 CDS")
                    c2.metric("Toplam İkame Mutasyonu", f"{tot_mutations} AA")
                    c3.metric("Mitonükleer Risk Endeksi", f"%{avg_risk:.1f}")
                    c4.metric("Uyum Durumu", "YÜKSEK BİYOFİZİKSEL FARK" if avg_risk > 35 else "UYUMLU")

                    st.markdown("---")
                    st.markdown("### 📊 Proteom Düzeyinde Mutasyon Biyofizik Haritası")
                    df_all = pd.DataFrame(all_farklar)
                    
                    if not df_all.empty:
                        fig = px.scatter(
                            df_all, x="Pozisyon", y="Lokal Risk Skoru", color="Gen", size="Hidrofobiklik Δ",
                            hover_data=["Tür 1 (AA)", "Tür 2 (AA)", "BLOSUM62 Skoru"],
                            template="plotly_dark", height=450
                        )
                        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown("### 📋 Mito-CRISPR Rehber RNA (gRNA) ve Mutasyon Dökümü")
                        st.dataframe(df_all, use_container_width=True)

                    st.session_state["son_analiz"] = f"Tüm 13 gen taranmıştır. Mutasyon: {tot_mutations} AA, Genel Risk Skoru: %{avg_risk:.1f}."
            else:
                res = gen_analiz_et(selected_code, rec1, rec2, crispr_mode)
                if res:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Gen", selected_code)
                    c2.metric("Dizi Uzunluğu", f"{res['seq_len']} AA")
                    c3.metric("Mutasyon Sayısı", f"{res['mutations']}")
                    c4.metric("Spesifik Gen Riski", f"%{res['risk_score']:.1f}")

                    st.session_state["son_sequence"] = res["seq2"]

                    st.markdown("---")
                    df_fark = pd.DataFrame(res['farklar'])
                    if not df_fark.empty:
                        fig = px.scatter(
                            df_fark, x="Pozisyon", y="Lokal Risk Skoru", size="Hidrofobiklik Δ", color="BLOSUM62 Skoru",
                            hover_data=["Tür 1 (AA)", "Tür 2 (AA)"], template="plotly_dark", height=400
                        )
                        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(df_fark, use_container_width=True)

                    st.session_state["son_analiz"] = f"{selected_code} geni analiz edildi. Mutasyon: {res['mutations']} AA, Risk: %{res['risk_score']:.1f}."
    else:
        st.info("👈 Analizi başlatmak için sol panelden türleri seçip **'BİYOİNFORMATİK ANALİZİ BAŞLAT'** butonuna tıklayın.")

# ==========================================
# TAB 2: py3Dmol PDB GÖRSELLEŞTİRİCİ
# ==========================================
with tab_3d_view:
    st.markdown("### 🧬 py3Dmol İnteraktif 3D Protein Yapı Analizörü")
    st.caption("RCSB PDB veritabanından PDB ID ile veya kendi yükleyeceğiniz .pdb dosyaları ile etkileşimli 3D görselleştirme.")

    col_view1, col_view2 = st.columns([1, 3])

    with col_view1:
        st.markdown("#### 🎨 3D Ayarlar")
        render_style = st.selectbox("Gösterim Modu", ["cartoon", "stick", "sphere"], index=0)
        color_scheme = st.selectbox("Renk Şeması", ["spectrum", "chain", "secondary structure"], index=0)
        show_surface = st.checkbox("Moleküler Yüzey (VDW Surface) Ekle", value=False)
        
        source_type = st.radio("PDB Veri Kaynağı", ["RCSB PDB kütüphanesi", "PDB Dosyası Yükle (.pdb)"])

    with col_view2:
        if source_type == "RCSB PDB kütüphanesi":
            pdb_id_input = st.text_input("RCSB PDB Accession Code", value="1A23", help="Örn: 1A23, 6VXX, 2RH1").strip()
            if st.button("3D Yapıyı Çiz (PDB ID)"):
                view = py3Dmol.view(width=800, height=500)
                view.addModel(f'pdb:{pdb_id_input}', 'pdb')
                
                style_dict = {color_scheme: {}} if render_style != "cartoon" else {'color': color_scheme}
                view.setStyle({render_style: style_dict})
                    
                if show_surface:
                    view.addSurface(py3Dmol.VDW, {'opacity': 0.5})
                    
                view.zoomTo()
                showmol(view, height=500, width=800)
        else:
            uploaded_pdb = st.file_uploader("Özel .pdb Dosyası Seçin", type=["pdb"])
            if uploaded_pdb is not None:
                pdb_str = uploaded_pdb.getvalue().decode("utf-8")
                view = py3Dmol.view(width=800, height=500)
                view.addModel(pdb_str, 'pdb')
                
                style_dict = {color_scheme: {}} if render_style != "cartoon" else {'color': color_scheme}
                view.setStyle({render_style: style_dict})
                    
                if show_surface:
                    view.addSurface(py3Dmol.VDW, {'opacity': 0.5})
                    
                view.zoomTo()
                showmol(view, height=500, width=800)

# ==========================================
# TAB 3: ESMFold ANLIK 3D YAPI TAHMİNİ
# ==========================================
with tab_esmfold:
    st.markdown("### 🔮 ESMFold API (Yapay Zeka ile Anlık 3D Protein Katlanması)")
    st.caption("Epitop veya amino asit dizisinden 3 saniye içinde yapay zeka ile PDB üretimi ve py3Dmol entegrasyonu.")

    varsayilan_dizi = st.session_state.get("son_sequence", "MTPMRTINPLMKLINHSFIDLPTPSNISAWWNFGSLLGACLILQITTGLFLAMHYSPDASTAFSSIAHITRDVNYGWIRYLHANGASMFFICLFLHIGRGLYYGSFLYSETWNIGIILLLATMATAFMGYVLPWGQMSFWGATVITNLFSAIPYIGTNLVEWIWGGFSVDKATLTRFFAFHFILPFIMALAAVHLTFLHETGSNNPTGLNSDSDKIPFHPYYTIKDILGALLLLALLLLTLFSPDLLGDPDNYTLANPLNTPPHIKPEWYFLFAYTILRSVPNKLGGVLALLLSILILIVIPILHTSKQRSMMFRPLSQCLFWLLTADLLTLTWIGGQPVEHPYITIGQMASILYFSIILILMPISGIIEDKMLKWN")
    
    user_fasta = st.text_area("Amino Asit Dizisi (FASTA / Düz Metin)", value=varsayilan_dizi, height=120)

    if st.button("✨ ESMFold Yapay Zeka ile 3D PDB Yapısını Tahmin Et"):
        clean_seq = "".join(user_fasta.split()).upper()
        
        if len(clean_seq) > 400:
            st.warning("⚠️ Hızlı API tahmini için dizi uzunluğu 400 amino asitten az olmalıdır.")
        elif len(clean_seq) == 0:
            st.error("❌ Lütfen geçerli bir amino asit dizisi girin.")
        else:
            with st.spinner("Meta ESMFold API üzerinden 3D atomik koordinatlar hesaplanıyor..."):
                try:
                    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
                    response = requests.post(url, data=clean_seq, timeout=60)
                    
                    if response.status_code == 200:
                        pdb_data = response.text
                        st.success("✅ 3D Katlanma Yapısı Başarıyla Hesaplandı!")
                        
                        view = py3Dmol.view(width=850, height=500)
                        view.addModel(pdb_data, 'pdb')
                        view.setStyle({'cartoon': {'color': 'spectrum'}})
                        view.zoomTo()
                        showmol(view, height=500, width=850)

                        st.download_button(
                            label="📥 Üretilen .PDB Dosyasını İndir",
                            data=pdb_data,
                            file_name="esmfold_predicted_structure.pdb",
                            mime="chemical/x-pdb"
                        )
                    else:
                        st.error(f"❌ ESMFold Servis Hatası (HTTP {response.status_code}). Lütfen dizinizi kontrol edin.")
                except Exception as e:
                    st.error(f"❌ Bağlantı Hatası: {str(e)}")

# ==========================================
# TAB 4: EVRENSEL GPT-4o BİYO-DANIŞMAN
# ==========================================
with tab_ai_bot:
    st.markdown("### 🤖 Mgen Analysis Evrensel Biyo-LLM Danışmanı")
    st.caption("Genetik mühendisliği, CRISPR, de-extinction veya genel moleküler biyoloji sorularınızı sorabilirsiniz.")

    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        pass

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {"role": "assistant", "content": "Merhaba! Ben Mgen Analysis Biyo-Danışmanıyım. Biyoinformatik analizleriniz, ESMFold 3D modelleriniz, mitokondriyal genetik veya de-extinction projeleriniz hakkındaki sorularınızı yanıtlamaya hazırım."}
        ]

    for msg in st.session_state["chat_messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    if user_prompt := st.chat_input("Sorunuzu yazın (Örn: ESMFold pLDDT skoru ne anlama gelir?)"):
        st.session_state["chat_messages"].append({"role": "user", "content": user_prompt})
        st.chat_message("user").write(user_prompt)

        analiz_ozet = st.session_state.get("son_analiz", "Kullanıcı henüz bir biyoinformatik analiz çalıştırmadı.")

        if api_key:
            try:
                client = openai.OpenAI(api_key=api_key)
                system_instruction = (
                    "Sen Mgen Analysis platformunda görev yapan senior seviye bir Biyoinformatik Uzmanı ve Sentetik Biyoloji Danışmanısın. "
                    "Kullanıcı sana genetik, biyofizik, de-extinction, CRISPR, protein katlanması, mitokondriyal DNA veya genel bilim ile ilgili HER TÜRLÜ soruyu sorabilir. "
                    "Yanıtların son derece akademik, detaylı, yapıcı ve bilimsel olarak doğru olmalıdır. "
                    f"Kullanıcının mevcut analiz bağlamı: '{analiz_ozet}'"
                )

                messages_for_api = [{"role": "system", "content": system_instruction}]
                for m in st.session_state["chat_messages"]:
                    messages_for_api.append({"role": m["role"], "content": m["content"]})

                with st.spinner("Mgen Analysis AI düşünüyor ve yanıtı hazırlıyor..."):
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages_for_api,
                        temperature=0.2
                    )
                    reply = response.choices[0].message.content

            except Exception as e:
                reply = f"❌ API Hatası: {str(e)}"
        else:
            reply = (
                "⚠️ **OpenAI API Anahtarı Bulunamadı!**\n\n"
                "Yapay zekanın evrensel sorulara eksiksiz yanıt verebilmesi için `.streamlit/secrets.toml` "
                "dosyanıza veya Streamlit Cloud üzerindeki Secrets paneline API anahtarınızı tanımlamalısınız:\n\n"
                "```toml\nOPENAI_API_KEY = 'sk-proj-...' \n```\n\n"
                f"**Mevcut Analiz Durumunuz:** {analiz_ozet}"
            )

        st.session_state["chat_messages"].append({"role": "assistant", "content": reply})
        st.chat_message("assistant").write(reply)
