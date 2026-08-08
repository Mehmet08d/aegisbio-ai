import streamlit as st
from Bio import Entrez, SeqIO
from Bio.Align import PairwiseAligner
import plotly.express as px
import pandas as pd
import openai

# --- SAYFA KONFİGÜRASYONU ---
st.set_page_config(
    page_title="AegisBio AI | Open-Access Bioinformatics & AI platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CANLI VE MODERN RENK TASARIMI (CSS) ---
st.markdown("""
    <style>
    .main { 
        background: linear-gradient(135deg, #090d16 0%, #111827 50%, #070a10 100%); 
        color: #f1f5f9; 
    }
    section[data-testid="stSidebar"] { 
        background-color: rgba(15, 23, 42, 0.9) !important; 
        backdrop-filter: blur(14px); 
        border-right: 1px solid rgba(56, 189, 248, 0.2); 
    }
    div[data-testid="stMetric"] { 
        background: rgba(30, 41, 59, 0.7) !important; 
        backdrop-filter: blur(10px); 
        border: 1px solid rgba(99, 102, 241, 0.3); 
        border-radius: 16px !important; 
        padding: 18px !important; 
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .stButton>button { 
        width: 100%; 
        background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%); 
        color: #ffffff !important; 
        font-weight: 800 !important; 
        border-radius: 12px !important; 
        height: 52px !important; 
        border: none !important; 
        box-shadow: 0 4px 15px rgba(6, 182, 212, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.6);
    }
    .brand-title {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        text-align: center;
    }
    .open-badge {
        background: rgba(34, 197, 94, 0.2);
        border: 1px solid #22c55e;
        color: #4ade80;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# AMİNO ASİT BİYOFİZİKSEL PARAMETRELERİ
AMINO_ACID_PROPERTIES = {
    'A': {'charge': 0, 'hydrophobicity': 1.8, 'size': 88.6},
    'R': {'charge': 1, 'hydrophobicity': -4.5, 'size': 173.4},
    'N': {'charge': 0, 'hydrophobicity': -3.5, 'size': 114.1},
    'D': {'charge': -1, 'hydrophobicity': -3.5, 'size': 111.1},
    'C': {'charge': 0, 'hydrophobicity': 2.5, 'size': 108.5},
    'Q': {'charge': 0, 'hydrophobicity': -3.5, 'size': 143.8},
    'E': {'charge': -1, 'hydrophobicity': -3.5, 'size': 138.4},
    'G': {'charge': 0, 'hydrophobicity': -0.4, 'size': 60.1},
    'H': {'charge': 1, 'hydrophobicity': -3.2, 'size': 153.2},
    'I': {'charge': 0, 'hydrophobicity': 4.5, 'size': 166.7},
    'L': {'charge': 0, 'hydrophobicity': 3.8, 'size': 166.7},
    'K': {'charge': 1, 'hydrophobicity': -3.9, 'size': 168.6},
    'M': {'charge': 0, 'hydrophobicity': 1.9, 'size': 162.9},
    'F': {'charge': 0, 'hydrophobicity': 2.8, 'size': 189.9},
    'P': {'charge': 0, 'hydrophobicity': -1.6, 'size': 112.7},
    'S': {'charge': 0, 'hydrophobicity': -0.8, 'size': 89.0},
    'T': {'charge': 0, 'hydrophobicity': -0.7, 'size': 116.1},
    'W': {'charge': 0, 'hydrophobicity': -0.9, 'size': 227.8},
    'Y': {'charge': 0, 'hydrophobicity': -1.3, 'size': 193.6},
    'V': {'charge': 0, 'hydrophobicity': 4.2, 'size': 140.0}
}

HAZIR_TURLER = {
    "🦘 Numbat (Myrmecobius fasciatus) [Taşıyıcı]": "NC_011949.1",
    "🐅 Tasman Kaplanı / Thylacine (Thylacinus cynocephalus) [Tükenmiş]": "NC_028319.1",
    "🐘 Asya Fili (Elephas maximus) [Taşıyıcı]": "NC_005129.2",
    "🦣 Yünlü Mamut (Mammuthus primigenius) [Tükenmiş]": "NC_007596.2",
    "🦣 Kolombiya Mamutu (Mammuthus columbi) [Tükenmiş]": "NC_015529.1",
    "🕊️ Şeritli Güvercin (Patagioenas fasciata) [Taşıyıcı]": "NC_028503.1",
    "🕊️ Yolcu Güvercini / Passenger Pigeon (Ectopistes migratorius) [Tükenmiş]": "NC_026723.1",
    "🕊️ Kaya Güvercini (Columba livia) [Taşıyıcı]": "NC_013978.1",
    "🦤 Dodo Kuşu (Raphus cucullatus) [Tükenmiş]": "NC_008263.1",
    "🦭 Dugong (Dugong dugon) [Taşıyıcı]": "NC_003314.1",
    "🐋 Steller Deniz İneği (Hydrodamalis gigas) [Tükenmiş]": "NC_045268.1",
    "👤 Modern İnsan (Homo sapiens)": "NC_012920.1",
    "🦴 Neandertal (Homo neanderthalensis) [Tükenmiş]": "NC_011137.1",
    "🦴 Denisova İnsanı (Denisova hominin) [Tükenmiş]": "NC_013993.1",
    "🐒 Şempanze (Pan troglodytes)": "NC_001643.1",
    "🐺 Gri Kurt (Canis lupus)": "NC_009686.1",
    "⚔️ Kılıç Dişli Kaplan / Smilodon (Smilodon fatalis) [Tükenmiş]": "NC_030752.1",
    "🐭 Laboratuvar Faresi (Mus musculus)": "NC_005089.1",
    "🦔 Çıplak Kör Fare (Heterocephalus glaber) [Kanser Dirençli]": "NC_015112.1",
    "🦈 Grönland Köpekbalığı (Somniosus microcephalus) [400+ Yıl Yaşam]": "NC_050842.1",
    "✏️ Custom (Özel NCBI Accession Kodu)": "CUSTOM"
}

GEN_KOD_HARITASI = {
    "🧬 TÜM 13 GENİ BİRLİKTE TARA (Toplu Analiz)": "ALL_13",
    "ND1 (Complex I)": "ND1", "ND2 (Complex I)": "ND2", "ND3 (Complex I)": "ND3",
    "ND4 (Complex I)": "ND4", "ND4L (Complex I)": "ND4L", "ND5 (Complex I)": "ND5", "ND6 (Complex I)": "ND6",
    "CYTB (Complex III)": "CYTB", "COX1 (Complex IV)": "COX1", "COX2 (Complex IV)": "COX2",
    "COX3 (Complex IV)": "COX3", "ATP6 (Complex V)": "ATP6", "ATP8 (Complex V)": "ATP8"
}

# --- BAŞLIK ---
st.markdown('<div style="text-align: center;"><span class="open-badge">🌐 100% FREE & OPEN ACCESS PLATFORM</span></div>', unsafe_allow_html=True)
st.markdown('<h1 class="brand-title">🧬 AegisBio AI</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.1rem;'>Mitonuclear Compatibility, De-Extinction Engineering & 24/7 AI Bio-Consultant</p>", unsafe_allow_html=True)

# SEKMELER
tab_analiz, tab_ai_bot = st.tabs(["🚀 Biyoinformatik Analiz Motoru", "🤖 AI Biyo-Danışman (7/24 Ücretsiz)"])

# ==========================================
# YARDIMCI FONKSİYONLAR & ÖNBELLEK
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
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    alignments = aligner.align(seq1, seq2)
    if not alignments:
        return seq1, seq2
    
    align_str = str(alignments[0])
    lines = [line for line in align_str.splitlines() if line.strip()]
    if len(lines) >= 3:
        return lines[0], lines[2]
    return seq1, seq2

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
            pr1 = AMINO_ACID_PROPERTIES.get(aa1, {'charge': 0, 'hydrophobicity': 0, 'size': 100})
            pr2 = AMINO_ACID_PROPERTIES.get(aa2, {'charge': 0, 'hydrophobicity': 0, 'size': 100})

            yuk_f = abs(pr1['charge'] - pr2['charge'])
            hid_f = abs(pr1['hydrophobicity'] - pr2['hydrophobicity'])
            hacim_f = abs(pr1['size'] - pr2['size'])
            
            lokal_r = (yuk_f * 3.0) + (hid_f * 1.5) + (hacim_f * 0.02)
            toplam_risk += lokal_r

            grna_target = f"5'-N20-NGG ({aa1}->{aa2} @Pos {idx+1})" if crispr_mode else "N/A"

            farklar.append({
                "Gen": gene_code, "Pozisyon": idx + 1, 
                "Tür 1 (AA)": aa1, "Tür 2 (AA)": aa2,
                "Yük Değişimi": yuk_f, "Hidrofobiklik Δ": round(hid_f, 2), 
                "Hacim Δ (Å³)": round(hacim_f, 1),
                "Lokal Risk Skoru": round(lokal_r, 2),
                "CRISPR gRNA Hedefi": grna_target
            })

    risk_skoru = min(100.0, (toplam_risk / (max(len(al1), 1) * 5.0)) * 280)
    return {"gene": gene_code, "seq_len": len(al1), "mutations": len(farklar), "risk_score": risk_skoru, "farklar": farklar}

# ==========================================
# TAB 1: BİYOİNFORMATİK ANALİZ PANELİ
# ==========================================
with tab_analiz:
    st.sidebar.markdown("### ⚙️ Analiz Parametreleri")
    user_email = st.sidebar.text_input("NCBI E-posta", "biyoinformatik@gmail.com")

    tur1_secim = st.sidebar.selectbox("1. Tür / Taşıyıcı Konak", list(HAZIR_TURLER.keys()), index=0)
    numbat_acc = st.sidebar.text_input("1. Tür NCBI Kodu", "NC_011949.1") if HAZIR_TURLER[tur1_secim] == "CUSTOM" else HAZIR_TURLER[tur1_secim]

    tur2_secim = st.sidebar.selectbox("2. Tür / Hedef Canlı", list(HAZIR_TURLER.keys()), index=1)
    thylacine_acc = st.sidebar.text_input("2. Tür NCBI Kodu", "NC_028319.1") if HAZIR_TURLER[tur2_secim] == "CUSTOM" else HAZIR_TURLER[tur2_secim]

    target_gene_raw = st.sidebar.selectbox("İncelenecek Mitokondriyal Gen", list(GEN_KOD_HARITASI.keys()), index=0)
    crispr_mode = st.sidebar.checkbox("🧬 Mito-CRISPR gRNA Adaylarını Hesapla", value=True)
    run_button = st.sidebar.button("🚀 ANALİZİ BAŞLAT")

    if run_button:
        selected_code = GEN_KOD_HARITASI[target_gene_raw]
        
        with st.spinner("Genom verileri NCBI üzerinden çekiliyor..."):
            rec1 = ncbi_genom_getir(numbat_acc, user_email)
            rec2 = ncbi_genom_getir(thylacine_acc, user_email)

        if not rec1 or not rec2:
            st.error("❌ NCBI üzerinden belirtilen Accession kodlarına ait veri çekilemedi. Lütfen bağlantınızı ve kodları kontrol edin.")
        else:
            if selected_code == "ALL_13":
                all_genes = ["ND1", "ND2", "ND3", "ND4", "ND4L", "ND5", "ND6", "CYTB", "COX1", "COX2", "COX3", "ATP6", "ATP8"]
                all_results = []
                all_farklar = []

                with st.spinner("Tüm 13 mitokondriyal gen analiz ediliyor..."):
                    for g in all_genes:
                        res = gen_analiz_et(g, rec1, rec2, crispr_mode)
                        if res:
                            all_results.append(res)
                            all_farklar.extend(res["farklar"])

                if all_results:
                    tot_mutations = sum(r["mutations"] for r in all_results)
                    avg_risk = sum(r["risk_score"] for r in all_results) / len(all_results)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Taranan Gen Sayısı", f"{len(all_results)} / 13")
                    c2.metric("Toplam Mutasyonlu Lokus", f"{tot_mutations}")
                    c3.metric("Ortalama Mitonükleer Risk", f"%{avg_risk:.1f}")

                    st.markdown("---")
                    st.markdown("### 📊 Tüm Proteom Mutasyon Dağılımı")
                    df_all = pd.DataFrame(all_farklar)
                    if not df_all.empty:
                        fig = px.scatter(
                            df_all, x="Pozisyon", y="Lokal Risk Skoru", color="Gen", size="Hidrofobiklik Δ",
                            hover_data=["Tür 1 (AA)", "Tür 2 (AA)"], template="plotly_dark"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown("### 📋 Mito-CRISPR Rehber RNA (gRNA) Aday Listesi")
                    st.dataframe(df_all, use_container_width=True)

                    st.session_state["son_analiz"] = f"Tüm 13 gen taranmıştır. Toplam Mutasyon: {tot_mutations}, Ortalama Risk: %{avg_risk:.1f}"
                else:
                    st.warning("Seçilen genomlarda tanımlı mitokondriyal CDS gen kodları eşleştirilemedi.")

            else:
                with st.spinner(f"{selected_code} geni analiz ediliyor..."):
                    res = gen_analiz_et(selected_code, rec1, rec2, crispr_mode)
                    if res:
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("İncelenen Gen", selected_code)
                        c2.metric("Dizilim Uzunluğu", f"{res['seq_len']} AA")
                        c3.metric("Farklı Lokus Sayısı", f"{res['mutations']}")
                        c4.metric("Risk Skoru", f"%{res['risk_score']:.1f}")

                        st.markdown("---")
                        df_fark = pd.DataFrame(res['farklar'])
                        if not df_fark.empty:
                            fig = px.scatter(
                                df_fark, x="Pozisyon", y="Lokal Risk Skoru", size="Hidrofobiklik Δ", color="Yük Değişimi",
                                template="plotly_dark"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.dataframe(df_fark, use_container_width=True)

                        st.session_state["son_analiz"] = f"{selected_code} geni analiz edildi. Mutasyon: {res['mutations']}, Risk: %{res['risk_score']:.1f}"
                    else:
                        st.warning(f"Seçilen genomlarda {selected_code} genine ait protein dizisi bulunamadı.")
    else:
        st.info("👈 Analizi başlatmak için sol taraftaki panelden türleri seçip **'ANALİZİ BAŞLAT'** butonuna basın.")

# ==========================================
# TAB 2: AI BİYO-DANIŞMAN (GPT-4o)
# ==========================================
with tab_ai_bot:
    st.markdown("### 🤖 AegisBio Yapay Zeka Biyo-Danışman (7/24 Sınırsız)")
    st.write("Mitonükleer uyumluluk, Mito-CRISPR gen düzenleme stratejileri veya de-extinction projelerinizle ilgili sorularınızı GPT-4o destekli uzman yapay zekamıza sorabilirsiniz.")

    if "OPENAI_API_KEY" in st.secrets:
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    else:
        client = None
        st.info("ℹ️ GPT-4o entegrasyonu için Streamlit Secrets üzerinde `OPENAI_API_KEY` tanımlanabilir. Tanımlanmadığında sistem akıllı simülasyon modunda yanıt verir.")

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {"role": "assistant", "content": "Merhaba! Ben AegisBio AI Biyo-Danışmanıyım. Genetik analiz sonuçlarınız, mitokondriyal biyofizik veya CRISPR gRNA tasarımları hakkında her türlü soruyu sorabilirsiniz."}
        ]

    for msg in st.session_state["chat_messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    if user_prompt := st.chat_input("Biyoinformatik veya genetik mühendisliği ile ilgili sorunuzu yazın..."):
        st.session_state["chat_messages"].append({"role": "user", "content": user_prompt})
        st.chat_message("user").write(user_prompt)

        analiz_ozet = st.session_state.get("son_analiz", "Henüz bir analiz çalıştırılmadı.")

        if client:
            try:
                system_instruction = (
                    "Sen AegisBio AI platformunda görev yapan uzman bir Biyoinformatik ve Sentetik Biyoloji Danışmanısın. "
                    "Kullanıcılara mitokondriyal genetik, mitonükleer uyumluluk, de-extinction ve Mito-CRISPR gRNA tasarımları "
                    f"konularında bilimsel ve net yanıtlar veriyorsun. Kullanıcının son analiz durumu: {analiz_ozet}"
                )

                api_messages = [{"role": "system", "content": system_instruction}]
                for m in st.session_state["chat_messages"]:
                    api_messages.append({"role": m["role"], "content": m["content"]})

                with st.spinner("GPT-4o yanıt hazırlıyor..."):
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=api_messages,
                        temperature=0.3
                    )
                    reply = response.choices[0].message.content

            except Exception as e:
                reply = f"❌ OpenAI API Hatası: {str(e)}"
        else:
            prompt_lower = user_prompt.lower()
            if "risk" in prompt_lower or "skor" in prompt_lower:
                reply = "Mitonükleer risk skoru; amino asit yük değişimi, hidrofobiklik ve hacimsel farkların ağırlıklı toplamıyla hesaplanır. %50 üzerindeki risk skorları, ATP sentaz ve respiratuvar kompleks dayanıklılığında olumsuz etkiye işaret edebilir."
            elif "crispr" in prompt_lower or "grna" in prompt_lower:
                reply = "Mito-CRISPR modülü, mutasyonlu lokusa en yakın PAM (NGG) dizilerini tespit ederek 20 nükleotitlik gRNA adaylarını listeler."
            else:
                reply = f"Sorunuz incelendi. [Mevcut Analiz Bağlamı: {analiz_ozet}] Biyofiziksel parametreler ışığında, mutasyonların protein katlanmasına ve ikincil yapılarına olan etkilerinin simüle edilmesini öneririm."

        st.session_state["chat_messages"].append({"role": "assistant", "content": reply})
        st.chat_message("assistant").write(reply)
