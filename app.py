import math
import re
import time
from Bio import Entrez, SeqIO
from Bio.Align import PairwiseAligner, substitution_matrices
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import py3Dmol
import requests
import streamlit as st
from st_mol import showmol  # 'stmol' yerine 'st_mol' olarak düzeltildi

# --- SAYFA KONFİGÜRASYONU ---
st.set_page_config(
    page_title="Mgen Analysis | Enterprise Bio-Suite & De-Extinction Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- SESSION STATE BAŞLATMA ---
if "chat_messages" not in st.session_state:
  st.session_state["chat_messages"] = [{
      "role": "assistant",
      "content": (
          "Merhaba! Ben Mgen Analysis Biyo-Danışmanıyım. Biyoinformatik"
          " analizleriniz, ESMFold 3D modelleriniz, Mito-CRISPR tasarımlarınız"
          " veya mitokondriyal genetik hakkındaki sorularınızı yanıtlamaya"
          " hazırım."
      ),
  }]
if "son_analiz" not in st.session_state:
  st.session_state["son_analiz"] = (
      "Kullanıcı henüz bir biyoinformatik analiz çalıştırmadı."
  )
if "son_sequence" not in st.session_state:
  st.session_state["son_sequence"] = (
      "MTPMRTINPLMKLINHSFIDLPTPSNISAWWNFGSLLGACLILQITTGLFLAMHYSPDASTAFSSIAHITRDVNYGWIRYLHANGASMFFICLFLHIGRGLYYGSFLYSETWNIGIILLLATMATAFMGYVLPWGQMSFWGATVITNLFSAIPYIGTNLVEWIWGGFSVDKATLTRFFAFHFILPFIMALAAVHLTFLHETGSNNPTGLNSDSDKIPFHPYYTIKDILGALLLLALLLLTLFSPDLLGDPDNYTLANPLNTPPHIKPEWYFLFAYTILRSVPNKLGGVLALLLSILILIVIPILHTSKQRSMMFRPLSQCLFWLLTADLLTLTWIGGQPVEHPYITIGQMASILYFSIILILMPISGIIEDKMLKWN"
  )
if "highlight_res_list" not in st.session_state:
  st.session_state["highlight_res_list"] = []

# --- GLASSMORPHISM UI TASARIMI ---
st.markdown(
    """
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
        height: 48px !important; 
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
""",
    unsafe_allow_html=True,
)

# BLOSUM62 MATRİSİ VE BİYOFİZİKSEL PARAMETRELER
blosum62 = substitution_matrices.load("BLOSUM62")

AMINO_ACID_PROPERTIES = {
    "A": {
        "charge": 0,
        "hydrophobicity": 1.8,
        "size": 88.6,
        "helix_propensity": 1.42,
    },
    "R": {
        "charge": 1,
        "hydrophobicity": -4.5,
        "size": 173.4,
        "helix_propensity": 0.98,
    },
    "N": {
        "charge": 0,
        "hydrophobicity": -3.5,
        "size": 114.1,
        "helix_propensity": 0.67,
    },
    "D": {
        "charge": -1,
        "hydrophobicity": -3.5,
        "size": 111.1,
        "helix_propensity": 1.01,
    },
    "C": {
        "charge": 0,
        "hydrophobicity": 2.5,
        "size": 108.5,
        "helix_propensity": 0.70,
    },
    "Q": {
        "charge": 0,
        "hydrophobicity": -3.5,
        "size": 143.8,
        "helix_propensity": 1.11,
    },
    "E": {
        "charge": -1,
        "hydrophobicity": -3.5,
        "size": 138.4,
        "helix_propensity": 1.51,
    },
    "G": {
        "charge": 0,
        "hydrophobicity": -0.4,
        "size": 60.1,
        "helix_propensity": 0.57,
    },
    "H": {
        "charge": 1,
        "hydrophobicity": -3.2,
        "size": 153.2,
        "helix_propensity": 1.00,
    },
    "I": {
        "charge": 0,
        "hydrophobicity": 4.5,
        "size": 166.7,
        "helix_propensity": 1.08,
    },
    "L": {
        "charge": 0,
        "hydrophobicity": 3.8,
        "size": 166.7,
        "helix_propensity": 1.21,
    },
    "K": {
        "charge": 1,
        "hydrophobicity": -3.9,
        "size": 168.6,
        "helix_propensity": 1.16,
    },
    "M": {
        "charge": 0,
        "hydrophobicity": 1.9,
        "size": 162.9,
        "helix_propensity": 1.45,
    },
    "F": {
        "charge": 0,
        "hydrophobicity": 2.8,
        "size": 189.9,
        "helix_propensity": 1.13,
    },
    "P": {
        "charge": 0,
        "hydrophobicity": -1.6,
        "size": 112.7,
        "helix_propensity": 0.57,
    },
    "S": {
        "charge": 0,
        "hydrophobicity": -0.8,
        "size": 89.0,
        "helix_propensity": 0.77,
    },
    "T": {
        "charge": 0,
        "hydrophobicity": -0.7,
        "size": 116.1,
        "helix_propensity": 0.83,
    },
    "W": {
        "charge": 0,
        "hydrophobicity": -0.9,
        "size": 227.8,
        "helix_propensity": 1.08,
    },
    "Y": {
        "charge": 0,
        "hydrophobicity": -1.3,
        "size": 193.6,
        "helix_propensity": 0.69,
    },
    "V": {
        "charge": 0,
        "hydrophobicity": 4.2,
        "size": 140.0,
        "helix_propensity": 1.06,
    },
}

HAZIR_TURLER = {
    # --- DE-EXTINCTION & ELEPHANTS ---
    "🦣 Yünlü Mamut (Mammuthus primigenius) [Tükenmiş]": "NC_007596.2",
    "🦣 Kolombiya Mamutu (Mammuthus columbi) [Tükenmiş]": "NC_015529.1",
    "🐘 Asya Fili (Elephas maximus) [Taşıyıcı Konak]": "NC_005129.2",
    "🐘 Afrika Çalı Fili (Loxodonta africana)": "NC_000934.1",
    "🐘 Afrika Orman Fili (Loxodonta cyclotis)": "NC_013693.1",
    "🦣 Düz Dişli Fil (Palaeoloxodon antiquus) [Tükenmiş]": "NC_034250.1",
    "🦣 Amerikan Mastodonu (Mammut americanum) [Tükenmiş]": "NC_009601.1",

    # --- DE-EXTINCTION FOCUS & THYLACINES ---
    "🐅 Tasman Kaplanı (Thylacinus cynocephalus) [Tükenmiş]": "NC_028319.1",
    "🦘 Numbat (Myrmecobius fasciatus) [Taşıyıcı Konak]": "NC_011949.1",
    "🦤 Dodo Kuşu (Raphus cucullatus) [Tükenmiş]": "NC_008263.1",
    "🐦 Rodrigues Solitaire (Pezophaps solitaria) [Tükenmiş]": "NC_013893.1",
    "🕊️ Nişobar Güvercini (Caloenas nicobarica) [Taşıyıcı Konak]": "NC_023968.1",
    "⚔️ Kılıç Dişli Kaplan (Smilodon fatalis) [Tükenmiş]": "NC_030752.1",
    "🦅 Haast Kartalı (Hieraaetus moorei) [Tükenmiş]": "NC_013805.1",
    "🕊️ Göçmen Güvercin (Ectopistes migratorius) [Tükenmiş]": "NC_026720.1",
    "🦜 Carolina Papağanı (Conuropsis carolinensis) [Tükenmiş]": "NC_025582.1",
    "🦆 Labrador Ördeği (Camptorhynchus labradorius) [Tükenmiş]": "NC_029471.1",
    "🦤 Dev Moa (Dinornis robustus) [Tükenmiş]": "NC_002674.1",
    "🦤 Küçük Çalı Moası (Anomalopteryx didiformis) [Tükenmiş]": "NC_002772.1",
    "🦤 Kivi Kuşu (Apteryx mantelli) [Taşıyıcı Konak]": "NC_002782.1",
    "🦭 Steller Deniz İneği (Hydrodamalis gigas) [Tükenmiş]": "NC_028624.1",
    "🦛 Madagaskar Cüce Hipopotamı (Hippopotamus lemerlei) [Tükenmiş]": "NC_023225.1",

    # --- HOMINIDS & PRIMATES ---
    "🦴 Neandertal İnsanı (Homo neanderthalensis) [Tükenmiş]": "NC_011137.1",
    "🦴 Denisova İnsanı (Denisova hominin) [Tükenmiş]": "NC_013993.1",
    "👤 Modern İnsan (Homo sapiens)": "NC_012920.1",
    "🐒 Batı Gorili (Gorilla gorilla)": "NC_011120.1",
    "🐒 Doğu Gorili (Gorilla beringei)": "NC_001645.1",
    "🐒 Şempanze (Pan troglodytes)": "NC_001643.1",
    "🐒 Bonobo (Pan paniscus)": "NC_001644.1",
    "🦧 Sumatra Orangutanı (Pongo abelii)": "NC_002083.1",
    "🦧 Borneo Orangutanı (Pongo pygmaeus)": "NC_001646.1",
    "🐒 Şebek / Rhesus Macaque (Macaca mulatta)": "NC_005943.1",
    "🐒 Yeşil Maymun (Chlorocebus sabaeus)": "NC_008066.1",
    "🐒 Halka Kuyruklu Lemur (Lemur catta)": "NC_004025.1",

    # --- FELIDS & CARNIVORA ---
    "🦁 Mağara Aslanı (Panthera spelaea) [Tükenmiş]": "NC_028308.1",
    "🐆 Amerikan Aslanı (Panthera atrox) [Tükenmiş]": "NC_028309.1",
    "🦁 Aslan (Panthera leo)": "NC_009692.1",
    "🐅 Kaplan (Panthera tigris)": "NC_006835.1",
    "🐆 Leopar (Panthera pardus)": "NC_010641.1",
    "🐆 Kar Leoparı (Panthera uncia)": "NC_010638.1",
    "🐆 Jaguar (Panthera onca)": "NC_009694.1",
    "🐆 Çita (Acinonyx jubatus)": "NC_005212.1",
    "🐈 Evcil Kedi (Felis catus)": "NC_001700.1",
    "🐺 Korkunç Kurt / Dire Wolf (Aenocyon dirus) [Tükenmiş]": "NC_058728.1",
    "🐺 Gri Kurt (Canis lupus)": "NC_009686.1",
    "🐕 Evcil Köpek (Canis lupus familiaris)": "NC_002008.4",
    "🦊 Kızıl Tilki (Vulpes vulpes)": "NC_008434.1",
    "🦊 Kutup Tilkisi (Vulpes lagopus)": "NC_009255.1",
    "🐻 Mağara Ayısı (Ursus spelaeus) [Tükenmiş]": "NC_011112.1",
    "🐻 Kutup Ayısı (Ursus maritimus)": "NC_003426.1",
    "🐻 Boz Ayı (Ursus arctos)": "NC_003427.1",
    "🐼 Dev Panda (Ailuropoda melanoleuca)": "NC_009484.1",
    "🦝 Kızıl Panda (Ailurus fulgens)": "NC_009691.1",

    # --- UNGULATES & MEGAFAUNA ---
    "🦣 Yünlü Gergedan (Coelodonta antiquitatis) [Tükenmiş]": "NC_012680.1",
    "🦏 Beyaz Gergedan (Ceratotherium simum)": "NC_001808.1",
    "🦏 Siyah Gergedan (Diceros bicornis)": "NC_012681.1",
    "🦏 Hint Gergedanı (Rhinoceros unicornis)": "NC_001779.1",
    "🦤 Aurochs / Yaban Öküzü (Bos primigenius) [Tükenmiş]": "NC_013996.1",
    "🐂 Sığır / İnek (Bos taurus)": "NC_006853.1",
    "🦬 Bizon (Bison bison)": "NC_012346.1",
    "🦬 Avrupa Bizonu (Bison bonasus)": "NC_014044.1",
    "🐴 Yaban Atı / Tarpan (Equus ferus) [Tükenmiş]": "NC_022268.1",
    "🐴 Quagga (Equus quagga quagga) [Tükenmiş]": "NC_000892.1",
    "🐴 Evcil At (Equus caballus)": "NC_001640.1",
    "🦓 Zebra (Equus quagga)": "NC_008103.1",
    "🦌 Dev İrlanda Geyiği (Megaloceros giganteus) [Tükenmiş]": "NC_023223.1",
    "🦌 Kızıl Geyik (Cervus elaphus)": "NC_007704.1",
    "🦌 Sığın / Sığın Geyiği (Alces alces)": "NC_020683.1",
    "🐪 Baktriyan Devesi (Camelus bactrianus)": "NC_009628.1",
    "🦒 Zürafa (Giraffa camelopardalis)": "NC_023961.1",
    "🦛 Su Aygırı (Hippopotamus amphibius)": "NC_000889.1",

    # --- CETACEANS & MARINE MAMMALS ---
    "🐋 Mavi Balina (Balaenoptera musculus)": "NC_001601.1",
    "🐋 Fin Balinası (Balaenoptera physalus)": "NC_001321.1",
    "🐋 Kambur Balina (Megaptera novaeangliae)": "NC_006927.1",
    "Whale İspermecet Balinası (Physeter macrocephalus)": "NC_002504.1",
    "🐬 Katil Balina / Orka (Orcinus orca)": "NC_023889.1",
    "🐬 Afalina / Yunus (Tursiops truncatus)": "NC_012059.1",
    "🦭 Karayip Keşiş Foku (Neomonachus tropicalis) [Tükenmiş]": "NC_028625.1",
    "🦭 Akdeniz Keşiş Foku (Monachus monachus)": "NC_008419.1",
    "🦭 Deniz Fili (Mirounga leonina)": "NC_008422.1",
    "🦦 Deniz Samuru (Enhydra lutris)": "NC_010298.1",

    # --- BIRDS ---
    "🦅 Kel Kartal (Haliaeetus leucocephalus)": "NC_008550.1",
    "🦅 Şahin (Buteo buteo)": "NC_003128.1",
    "🦅 Gökdoğan / Peregrine Falcon (Falco peregrinus)": "NC_029348.1",
    "🦉 Bayağı Puhu (Bubo bubo)": "NC_028441.1",
    "🐧 İmparator Penguen (Aptenodytes forsteri)": "NC_008432.1",
    "🐧 Adélie Pengueni (Pygoscelis adeliae)": "NC_000879.1",
    "🦩 Şili Flamingo (Phoenicopterus chilensis)": "NC_010089.1",
    "🦜 Gece Papağanı / Kakapo (Strigops habroptila)": "NC_005951.1",
    "🐓 Evcil Tavuk (Gallus gallus)": "NC_006856.1",
    "🦆 Yeşilbaş Ördek (Anas platyrhynchos)": "NC_009684.1",
    "🦚 Bayağı Tavuskuşu (Pavo cristatus)": "NC_015243.1",
    "🦃 Hindi (Meleagris gallopavo)": "NC_010195.1",

    # --- REPTILES & AMPHIBIANS ---
    "🦎 Komodo Ejderi (Varanus komodoensis)": "NC_008775.1",
    "🐊 Nil Timsahı (Crocodylus niloticus)": "NC_002744.1",
    "🐊 Amerikan Aligatörü (Alligator mississippiensis)": "NC_001922.1",
    "🐢 Galápagos Dev Kaplumbağası (Chelonoidis niger)": "NC_000888.1",
    "🐢 Çalı Deri Sırtlı Deniz Kaplumbağası (Dermochelys coriacea)": "NC_000886.1",
    "🐍 Kral Kobra (Ophiophagus hannah)": "NC_011394.1",
    "🐍 Yeşil Piton (Morelia viridis)": "NC_012708.1",
    "🐸 Boğa Kurbağası (Lithobates catesbeianus)": "NC_002805.1",
    "🦎 Aksolotl (Ambystoma mexicanum)": "NC_000868.1",

    # --- MARSUPIALS & MONOTREMES ---
    "🦘 Kızıl Kanguru (Osphranter rufus)": "NC_018360.1",
    "🐨 Koala (Phascolarctos cinereus)": "NC_002831.1",
    "🦇 Tazmanya Canavarı (Sarcophilus harrisii)": "NC_018359.1",
    "Ornithorhynchus anatinus (Ornithorhynchus anatinus)": "NC_000891.1",
    "🦔 Dikenli Karıncayiyan (Tachyglossus aculeatus)": "NC_008291.1",

    # --- INSECTS & INVERTEBRATES ---
    "🐝 Bal Arısı (Apis mellifera)": "NC_001566.1",
    "🪰 Meyve Sinek (Drosophila melanogaster)": "NC_001709.1",
    "🦋 İpek Böceği (Bombyx mori)": "NC_002355.1",
    "🦗 Göçmen Çekirge (Locusta migratoria)": "NC_001712.1",
    "🐙 Bayağı Ahtapot (Octopus vulgaris)": "NC_006353.1",
    "🦑 Dev Mürekkep Balığı (Architeuthis dux)": "NC_021464.1",

    # --- CUSTOM ENTRY ---
    "✏️ Custom (Özel NCBI Accession ID)": "CUSTOM",                            
}
HAZIR_PDBLER = {
    "Serbest Seçim / Manuel PDB Girişi": "MANUAL",
    "Kompleks IV: Sitokrom c Oksidaz (Mito-COX)": "1OCC",
    "Kompleks III: Sitokrom bc1 Kompleksi": "1BGY",
    "Kompleks I: NADH Dehidrojenaz Kompleksi": "5XTD",
    "Kompleks V: ATP Sentaz F1 Sektörü": "1E79",
    "SpCas9-gRNA-DNA Trikompleks Yapısı": "4OO8",
}

GEN_KOD_HARITASI = {
    "🧬 TÜM 13 GENİ BİRLİKTE TARA (Toplu Analiz)": "ALL_13",
    "ND1 (Complex I)": "ND1",
    "ND2 (Complex I)": "ND2",
    "ND3 (Complex I)": "ND3",
    "ND4 (Complex I)": "ND4",
    "ND4L (Complex I)": "ND4L",
    "ND5 (Complex I)": "ND5",
    "ND6 (Complex I)": "ND6",
    "CYTB (Complex III)": "CYTB",
    "COX1 (Complex IV)": "COX1",
    "COX2 (Complex IV)": "COX2",
    "COX3 (Complex IV)": "COX3",
    "ATP6 (Complex V)": "ATP6",
    "ATP8 (Complex V)": "ATP8",
}

CRISPR_PAM_TIPI = {
    "SpCas9 (Mito-Cas9) [PAM: NGG | 20 nt gRNA]": {
        "pam_regex": r"(?=(.{20}([ATCG]GG)))",
        "grna_len": 20,
        "pam_len": 3,
    },
    "SaCas9 [PAM: NNGRRT | 21 nt gRNA]": {
        "pam_regex": r"(?=(.{21}([ATCG]{2}[AG]GG[AG][ATCG])))",
        "grna_len": 21,
        "pam_len": 6,
    },
    "Mito-CBE / DddA Base Editor [PAM: T-Rich Context | 20 nt]": {
        "pam_regex": r"(?=(.{20}(T[ATCG]{2})))",
        "grna_len": 20,
        "pam_len": 3,
    },
}

st.markdown(
    """
    <div class="brand-header">
        <span class="academic-badge">Institutional Genomic Suite v4.5 Enterprise</span>
        <h1 class="brand-title">Mgen Analysis</h1>
        <p style='color: #94a3b8; font-size: 1.1rem; max-width: 850px; margin: 0 auto;'>
            Mito-CRISPR / Base Editing Engine, Mitonuclear Incompatibility Analytics, Interactive py3Dmol & ESMFold pLDDT Structural Suite
        </p>
    </div>
""",
    unsafe_allow_html=True,
)

# ANA SEKMELER
tab_analiz, tab_crispr, tab_3d_view, tab_esmfold, tab_simulasyon = st.tabs([
    "🚀 Biyoinformatik Analiz Engine",
    "🧬 Mito-CRISPR & Base Editing",
    "🧊 py3Dmol PDB Görselleştirici",
    "🔮 ESMFold 3D & pLDDT Analizi",
    "🧬 Evrim Simülatörü"
])


# ==========================================
# YARDIMCI VE HESAPLAMA FONKSİYONLARI
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def ncbi_genom_getir(accession_id, email):
  Entrez.email = email
  try:
    handle = Entrez.efetch(
        db="nucleotide", id=accession_id, rettype="gb", retmode="text"
    )
    record = SeqIO.read(handle, "genbank")
    handle.close()
    return record
  except Exception:
    return None


def recorddan_protein_ve_dna_ayikla(record, gen_adi):
  if not record:
    return None, None
  gen_sozlugu = {
      "ND1": ["ND1", "NAD1"],
      "ND2": ["ND2", "NAD2"],
      "ND3": ["ND3", "NAD3"],
      "ND4": ["ND4", "NAD4"],
      "ND4L": ["ND4L", "NAD4L"],
      "ND5": ["ND5", "NAD5"],
      "ND6": ["ND6", "NAD6"],
      "CYTB": ["CYTB", "CYTOCHROME B", "COB"],
      "COX1": ["COX1", "COI", "COXI"],
      "COX2": ["COX2", "COII"],
      "COX3": ["COX3", "COIII"],
      "ATP6": ["ATP6", "ATPASE 6"],
      "ATP8": ["ATP8", "ATPASE 8"],
  }
  hedef_etiketler = gen_sozlugu.get(gen_adi, [gen_adi])
  for feature in record.features:
    if feature.type == "CDS":
      gene_tag = feature.qualifiers.get("gene", [""])[0].upper()
      product_tag = feature.qualifiers.get("product", [""])[0].upper()
      if any(e in gene_tag or e in product_tag for e in hedef_etiketler):
        protein_seq = feature.qualifiers.get("translation", [""])[0]
        dna_seq = str(feature.extract(record.seq))
        return protein_seq, dna_seq
  return None, None


def dizileri_hizala(seq1, seq2):
  try:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    alignments = aligner.align(seq1, seq2)
    if len(alignments) > 0:
      aligned = alignments[0]
      aligned_seq1 = "".join(
          [seq1[i] if i != -1 else "-" for i in aligned.indices[0]]
      )
      aligned_seq2 = "".join(
          [seq2[j] if j != -1 else "-" for j in aligned.indices[1]]
      )
      return aligned_seq1, aligned_seq2
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


def gc_icerigi_hesapla(dna_seq):
  if not dna_seq:
    return 0.0
  gc_count = sum(1 for n in dna_seq.upper() if n in ["G", "C"])
  return (gc_count / len(dna_seq)) * 100


def melting_temp_hesapla(dna_seq):
  if not dna_seq:
    return 0.0
  seq = dna_seq.upper()
  a_t = sum(1 for n in seq if n in ["A", "T"])
  g_c = sum(1 for n in seq if n in ["G", "C"])
  if len(seq) < 14:
    return (a_t * 2) + (g_c * 4)
  return round(64.9 + 41 * (g_c - 16.4) / (a_t + g_c), 1)


def crispr_grna_tara(dna_sequence, system_type):
  cfg = CRISPR_PAM_TIPI[system_type]
  grna_len = cfg["grna_len"]
  matches = []
  seq_upper = dna_sequence.upper()

  for i in range(len(seq_upper) - grna_len - 6):
    window = seq_upper[i : i + grna_len + 6]
    grna_candidate = window[:grna_len]
    pam_candidate = window[grna_len : grna_len + cfg["pam_len"]]

    # Basit SpCas9 NGG veya Pam Kuralları
    if ("SpCas9" in system_type and pam_candidate.endswith("GG")) or (
        "SaCas9" in system_type and "GG" in pam_candidate
    ):
      gc = gc_icerigi_hesapla(grna_candidate)
      tm = melting_temp_hesapla(grna_candidate)
      # Skorlama (GC %40-60 arası yüksek skor alır)
      quality_score = 100 - abs(50 - gc) * 2 - (0 if 50 <= tm <= 65 else 10)
      quality_score = max(10, min(99, round(quality_score, 1)))

      matches.append({
          "Pozisyon (nt)": i + 1,
          "gRNA Dizisi (5'->3')": grna_candidate,
          "PAM": pam_candidate,
          "GC %": round(gc, 1),
          "Tm (°C)": tm,
          "Verimlilik Skoru": quality_score,
      })
  return matches


def pdb_plDDT_ayikla(pdb_text):
  plDDT_list = []
  lines = pdb_text.splitlines()
  current_res = None
  for line in lines:
    if line.startswith("ATOM") or line.startswith("HETATM"):
      try:
        res_id = int(line[22:26].strip())
        b_factor = float(line[60:76].strip())
        if res_id != current_res:
          plDDT_list.append((res_id, b_factor))
          current_res = res_id
      except ValueError:
        continue
  return pd.DataFrame(plDDT_list, columns=["Residue", "pLDDT"])


def gen_analiz_et(gene_code, rec1, rec2):
  p1, dna1 = recorddan_protein_ve_dna_ayikla(rec1, gene_code)
  p2, dna2 = recorddan_protein_ve_dna_ayikla(rec2, gene_code)
  if not p1 or not p2:
    return None

  al1, al2 = dizileri_hizala(p1, p2)
  farklar = []
  toplam_risk = 0.0

  for idx in range(min(len(al1), len(al2))):
    aa1, aa2 = al1[idx], al2[idx]
    if aa1 != "-" and aa2 != "-" and aa1 != aa2:
      pr1 = AMINO_ACID_PROPERTIES.get(
          aa1,
          {"charge": 0, "hydrophobicity": 0, "size": 100, "helix_propensity": 1.0},
      )
      pr2 = AMINO_ACID_PROPERTIES.get(
          aa2,
          {"charge": 0, "hydrophobicity": 0, "size": 100, "helix_propensity": 1.0},
      )

      yuk_f = abs(pr1["charge"] - pr2["charge"])
      hid_f = abs(pr1["hydrophobicity"] - pr2["hydrophobicity"])
      hacim_f = abs(pr1["size"] - pr2["size"])
      helix_f = abs(pr1["helix_propensity"] - pr2["helix_propensity"])

      blosum_skor = get_blosum_score(aa1, aa2)

      lokal_r = (
          (yuk_f * 3.5)
          + (hid_f * 2.0)
          + (hacim_f * 0.03)
          + (helix_f * 4.0)
          - (blosum_skor * 0.8)
      )
      lokal_r = max(0.1, lokal_r)
      toplam_risk += lokal_r

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
      })

  genel_risk = min(100.0, (toplam_risk / (max(len(al1), 1) * 3.5)) * 100)
  return {
      "gene": gene_code,
      "seq_len": len(al1),
      "mutations": len(farklar),
      "risk_score": genel_risk,
      "farklar": farklar,
      "seq1": p1,
      "seq2": p2,
      "dna1": dna1,
      "dna2": dna2,
  }


# ==========================================
# TAB 1: BİYOİNFORMATİK ANALİZ ENGINE
# ==========================================
with tab_analiz:
  st.sidebar.markdown("### ⚙️ Analiz Parametreleri")
  user_email = st.sidebar.text_input(
      "NCBI E-posta", "researcher@mgenanalysis.org", key="ncbi_email_input"
  )

  tur1_secim = st.sidebar.selectbox(
      "1. Tür / Taşıyıcı Konak",
      list(HAZIR_TURLER.keys()),
      index=0,
      key="tur1_select",
  )
  numbat_acc = (
      st.sidebar.text_input(
          "1. Tür NCBI Kodu", "NC_007596.2", key="tur1_acc_input"
      )
      if HAZIR_TURLER[tur1_secim] == "CUSTOM"
      else HAZIR_TURLER[tur1_secim]
  )

  tur2_secim = st.sidebar.selectbox(
      "2. Tür / Hedef Canlı",
      list(HAZIR_TURLER.keys()),
      index=1,
      key="tur2_select",
  )
  thylacine_acc = (
      st.sidebar.text_input(
          "2. Tür NCBI Kodu", "NC_005129.2", key="tur2_acc_input"
      )
      if HAZIR_TURLER[tur2_secim] == "CUSTOM"
      else HAZIR_TURLER[tur2_secim]
  )

  target_gene_raw = st.sidebar.selectbox(
      "İncelenecek Mitokondriyal Gen",
      list(GEN_KOD_HARITASI.keys()),
      index=0,
      key="gene_select",
  )
  run_button = st.sidebar.button(
      "🚀 BİYOİNFORMATİK ANALİZİ BAŞLAT", key="run_analysis_btn"
  )

  if run_button:
    selected_code = GEN_KOD_HARITASI[target_gene_raw]

    with st.spinner(
        "Genom verileri NCBI üzerinden çekiliyor ve matrisler işleniyor..."
    ):
      rec1 = ncbi_genom_getir(numbat_acc, user_email)
      rec2 = ncbi_genom_getir(thylacine_acc, user_email)

    if not rec1 or not rec2:
      st.error(
          "❌ NCBI üzerinden GenBank kayıtları çekilemedi. Accession kodlarını"
          " kontrol edin."
      )
    else:
      all_genes = [
          "ND1",
          "ND2",
          "ND3",
          "ND4",
          "ND4L",
          "ND5",
          "ND6",
          "CYTB",
          "COX1",
          "COX2",
          "COX3",
          "ATP6",
          "ATP8",
      ]
      target_list = (
          all_genes if selected_code == "ALL_13" else [selected_code]
      )

      sonuclar = []
      tum_farklar = []

      for g_code in target_list:
        res = gen_analiz_et(g_code, rec1, rec2)
        if res:
          sonuclar.append(res)
          tum_farklar.extend(res["farklar"])

      if sonuclar:
        df_res = pd.DataFrame(sonuclar)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("İncelenen Gen Sayısı", len(sonuclar))
        m2.metric("Toplam Mutasyon Noktası", len(tum_farklar))
        ort_risk = df_res["risk_score"].mean()
        m3.metric("Ortalama Uyumsuzluk Riski", f"%{ort_risk:.1f}")
        m4.metric(
            "Kritik Uyumsuz Gen",
            df_res.loc[df_res["risk_score"].idxmax()]["gene"],
        )

        st.markdown("---")

        col_g1, col_g2 = st.columns(2)
        with col_g1:
          fig_bar = px.bar(
              df_res,
              x="gene",
              y="risk_score",
              color="risk_score",
              title="Gen Bazlı Mitokondriyal Uyumsuzluk Skoru (%)",
              color_continuous_scale="Reds",
              labels={"risk_score": "Risk (%)", "gene": "Gen"},
          )
          fig_bar.update_layout(
              paper_bgcolor="rgba(0,0,0,0)",
              plot_bgcolor="rgba(0,0,0,0)",
              font_color="#f8fafc",
          )
          st.plotly_chart(fig_bar, use_container_width=True)

        with col_g2:
          fig_mut = px.scatter(
              df_res,
              x="gene",
              y="mutations",
              size="seq_len",
              color="risk_score",
              title="Mutasyon Yükü vs Gen Uzunluğu",
              color_continuous_scale="Viridis",
              labels={"mutations": "Mutasyon Sayısı", "gene": "Gen"},
          )
          fig_mut.update_layout(
              paper_bgcolor="rgba(0,0,0,0)",
              plot_bgcolor="rgba(0,0,0,0)",
              font_color="#f8fafc",
          )
          st.plotly_chart(fig_mut, use_container_width=True)

        if tum_farklar:
          st.markdown("### 📋 Mutasyon ve Biyofiziksel Değişim Detayları")
          df_farklar = pd.DataFrame(tum_farklar)
          st.dataframe(df_farklar, use_container_width=True)

          # State güncelleme
          mut_positions = [f["Pozisyon"] for f in tum_farklar]
          st.session_state["highlight_res_list"] = mut_positions

        st.session_state["son_sequence"] = sonuclar[0]["seq1"]
        st.session_state["son_dna"] = sonuclar[0]["dna1"]
        st.session_state["son_analiz"] = (
            f"NCBI Analiz Özeti:\nTür 1: {rec1.description}\nTür 2:"
            f" {rec2.description}\nToplam {len(tum_farklar)} amino asit"
            " mutasyonu tespit edildi. Ortalama Mitokondriyal Risk Skoru:"
            f" %{ort_risk:.2f}."
        )
        st.success(
            "✅ Biyoinformatik analizi tamamlandı. Mutasyonlar py3Dmol ve"
            " Mito-CRISPR sekmelerine aktarıldı."
        )


# ==========================================
# TAB 2: MITO-CRISPR & BASE EDITING ENGINE
# ==========================================
with tab_crispr:
  st.markdown("### 🧬 Mito-CRISPR & Base Editing Tasarım Engine")
  st.caption(
      "Mitokondriyal DNA dizilerinde spesifik gen düzenleme, nokta mutasyonu"
      " onarımı ve gRNA adaylarının taranması."
  )

  c_c1, c_c2 = st.columns([2, 1])
  with c_c1:
    default_dna = st.session_state.get(
        "son_dna",
        "ATGACCCCAATACGCAAAATTAACCCCCTAATAAAATTAATTAACCACTCATTCATCGACCTCCCCACACCATCCAACATCTCCGCATGATGAAACTTCGGCTCACTCCTAGGCGCCTGCCTGATCCTACAAATCACCACAGGACTATTCCTAGCCATACACTACTCACCCGACGCCTCAACCGCATTTTCATCAATCGCCCACATCACTCGAGACGTAAATTATGGCTGAATCCGCTATCTACACGCCAACGGAGCCTCAATATTCTTTATCTGCCTCTTCCTACACATCGGACGAGGCCTATACTACGGCTCCTTCCTATAC",
    )
    crispr_dna_input = st.text_area(
        "Hedef Mitokondriyal DNA Dizisi (5' -> 3')",
        default_dna,
        height=140,
        key="crispr_dna_area",
    )
  with c_c2:
    selected_crispr_system = st.selectbox(
        "CRISPR / Cas Sistem Tipi",
        list(CRISPR_PAM_TIPI.keys()),
        index=0,
        key="crispr_sys_select",
    )
    min_gc = st.slider(
        "Minimum GC İçeriği (%)", 20, 50, 35, key="min_gc_slider"
    )
    max_gc = st.slider(
        "Maksimum GC İçeriği (%)", 50, 80, 65, key="max_gc_slider"
    )

  if st.button("🔎 Mito-gRNA Adaylarını Tara", key="scan_grna_btn"):
    clean_dna = "".join(crispr_dna_input.split()).upper()

    if not clean_dna or len(clean_dna) < 30:
      st.warning(
          "⚠️ Lütfen gRNA taraması için en az 30 nükleotit uzunluğunda DNA"
          " dizisi girin."
      )
    else:
      results = crispr_grna_tara(clean_dna, selected_crispr_system)
      if results:
        df_grna = pd.DataFrame(results)

        # GC filtresi
        df_filtered = df_grna[
            (df_grna["GC %"] >= min_gc) & (df_grna["GC %"] <= max_gc)
        ]

        m_g1, m_g2, m_g3 = st.columns(3)
        m_g1.metric("Toplam Bulunan Aday gRNA", len(df_grna))
        m_g2.metric("Filtreye Uygun gRNA", len(df_filtered))
        top_score = (
            df_filtered["Verimlilik Skoru"].max()
            if not df_filtered.empty
            else 0
        )
        m_g3.metric("En Yüksek Verimlilik Skoru", f"{top_score} / 100")

        st.markdown("#### 🎯 Uygun Mito-gRNA Aday Listesi")
        st.dataframe(df_filtered, use_container_width=True)

        # Grafik Görselleştirme
        if not df_filtered.empty:
          fig_grna = px.scatter(
              df_filtered,
              x="Pozisyon (nt)",
              y="Verimlilik Skoru",
              size="GC %",
              color="Tm (°C)",
              hover_name="gRNA Dizisi (5'->3')",
              title="Genom Boyunca gRNA Dağılımı ve Kalite Metrikleri",
              color_continuous_scale="Plasma",
          )
          fig_grna.update_layout(
              paper_bgcolor="rgba(0,0,0,0)",
              plot_bgcolor="rgba(0,0,0,0)",
              font_color="#f8fafc",
          )
          st.plotly_chart(fig_grna, use_container_width=True)

          # İndirme Butonu
          csv_data = df_filtered.to_csv(index=False).encode("utf-8")
          st.download_button(
              label="📥 gRNA Adaylarını CSV Olarak İndir",
              data=csv_data,
              file_name="mito_crispr_grna_candidates.csv",
              mime="text/csv",
              key="dl_grna_csv",
          )
      else:
        st.error(
            "❌ Seçilen sistem parametrelerine uygun PAM veya gRNA dizisi"
            " bulunamadı."
        )


# ==========================================
# TAB 3: GELİŞMİŞ py3Dmol PDB GÖRSELLEŞTİRİCİ
# ==========================================
with tab_3d_view:
  st.markdown("### 🧊 py3Dmol Etkileşimli Yapısal Görselleştirici")
  st.caption(
      "Gelişmiş yüzey gösterimi, mutasyonlu rezidü vurgulama ve özelleştirilebilir"
      " 3D moleküler kamera."
  )

  v_c1, v_c2, v_c3, v_c4 = st.columns([2, 2, 2, 2])

  with v_c1:
    preset_choice = st.selectbox(
        "Hazır Kompleks Seçimi",
        list(HAZIR_PDBLER.keys()),
        index=0,
        key="pdb_preset_select",
    )
    if HAZIR_PDBLER[preset_choice] != "MANUAL":
      target_pdb = HAZIR_PDBLER[preset_choice]
    else:
      target_pdb = st.text_input("PDB Kodu", "1OCC", key="manual_pdb_input")

  with v_c2:
    style_3d = st.selectbox(
        "Gösterim Stili",
        ["cartoon", "stick", "sphere", "line"],
        index=0,
        key="style_3d_select",
    )
    color_3d = st.selectbox(
        "Renk Paleti",
        ["spectrum", "chain", "secondary structure", "residue"],
        index=0,
        key="color_3d_select",
    )

  with v_c3:
    surface_enable = st.checkbox(
        "Moleküler Yüzey Eklesin mi?", value=False, key="surf_check"
    )
    surface_opacity = st.slider(
        "Yüzey Şeffaflığı", 0.1, 1.0, 0.6, key="surf_opacity"
    )

  with v_c4:
    bg_color = st.selectbox(
        "Arka Plan Rengi",
        ["#020617", "#000000", "#ffffff"],
        index=0,
        key="bg_color_select",
    )
    spin_enable = st.checkbox(
        "3D Otomatik Döndürme", value=False, key="spin_check"
    )

  res_highlight_input = st.text_input(
      "Vurgulanacak Rezidü Pozisyonları (Virgülle Ayırın)",
      value=", ".join(map(str, st.session_state.get("highlight_res_list", []))),
      key="res_highlight_input",
  )

  if target_pdb:
    try:
      view = py3Dmol.view(query=f"pdb:{target_pdb.lower()}")
      view.setBackgroundColor(bg_color)

      # Temel Stil
      if style_3d == "cartoon":
        view.setStyle({"cartoon": {"color": color_3d}})
      elif style_3d == "stick":
        view.setStyle({"stick": {"colorscheme": color_3d}})
      elif style_3d == "sphere":
        view.setStyle({"sphere": {"colorscheme": color_3d}})
      else:
        view.setStyle({"line": {}})

      # Mutasyon Vurgulama
      if res_highlight_input.strip():
        try:
          res_ids = [
              int(r.strip())
              for r in res_highlight_input.split(",")
              if r.strip().isdigit()
          ]
          for r_id in res_ids:
            view.addStyle({
                "resi": r_id
            }, {
                "sphere": {"color": "#ef4444", "radius": 1.2},
                "stick": {"color": "#f59e0b", "radius": 0.4},
            })
            view.addLabel(
                f"Mut:{r_id}",
                {
                    "fontColor": "#ffffff",
                    "backgroundColor": "#ef4444",
                    "fontSize": 12,
                },
                {"resi": r_id},
            )
        except Exception:
          pass

      # Yüzey Eklemeleri
      if surface_enable:
        view.addSurface(
            py3Dmol.VDW,
            {"opacity": surface_opacity, "color": "white"},
            {"hetflag": False},
        )

      if spin_enable:
        view.spin(True)

      view.zoomTo()
      showmol(view, height=550, width=950)
    except Exception as e:
      st.error(f"PDB görselleştirilirken bir hata oluştu: {e}")


# ==========================================
# TAB 4: ESMFold 3D & pLDDT GÜVENİLİRLİK ANALİZİ
# ==========================================
with tab_esmfold:
  st.markdown("### 🔮 Meta ESMFold 3D Yapı & pLDDT Kalite Analizi")
  st.caption(
      "Diziden 3D katlanmış yapı tahmini yapın ve renklendirilmiş B-factor"
      " (pLDDT) güvenilirlik profilini inceleyin."
  )

  col_e1, col_e2 = st.columns([3, 1])

  with col_e1:
    default_fasta = st.session_state.get(
        "son_sequence",
        "MTPMRTINPLMKLINHSFIDLPTPSNISAWWNFGSLLGACLILQITTGLFLAMHYSPDASTAFSSIA",
    )
    input_fasta = st.text_area(
        "Amino Asit Dizisi (FASTA / Düz Metin)",
        default_fasta,
        height=120,
        key="esmfold_fasta_area",
    )

  with col_e2:
    st.markdown("#### Quick Tools")
    if st.button("📋 Analiz Edilen Proteini Getir", key="fetch_last_seq"):
      st.rerun()

    color_by_pLDDT = st.checkbox(
        "pLDDT Güvenilirliğine Göre Renklendir", value=True, key="plddt_color_chk"
    )

  if st.button("🔮 ESMFold 3D Yapıyı Tahmin Et & Analiz Et", key="predict_esm_btn"):
    clean_seq = "".join(input_fasta.split()).upper()

    if not clean_seq:
      st.error("Lütfen geçerli bir amino asit dizisi girin.")
    elif len(clean_seq) > 400:
      st.warning(
          "⚠️ ESMFold sunucu performansı için dizi ilk 400 amino asit ile"
          " sınırlandırılıyor."
      )
      clean_seq = clean_seq[:400]

    with st.spinner("Meta ESMFold API'sine bağlanılıyor ve 3D yapı türetiliyor..."):
      try:
        url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
        response = requests.post(
            url, data=clean_seq, headers={"Content-Type": "text/plain"}
        )

        if response.status_code == 200:
          pdb_data = response.text
          st.success("✅ 3D Protein Katlanması Başarıyla Tamamlandı!")

          # pLDDT Ayrıştırma
          df_plddt = pdb_plDDT_ayikla(pdb_data)

          if not df_plddt.empty:
            avg_plddt = df_plddt["pLDDT"].mean()

            me1, me2, me3 = st.columns(3)
            me1.metric("Ortalama pLDDT Güvenilirlik Skoru", f"{avg_plddt:.1f} / 100")
            me2.metric(
                "Yüksek Güvenilirlikli Rezidü (% >70)",
                f"%{(sum(df_plddt['pLDDT'] > 70)/len(df_plddt))*100:.1f}",
            )
            me3.metric("Toplam Amino Asit Uzunluğu", len(df_plddt))

            # pLDDT Plotly Grafiği
            fig_plddt = px.line(
                df_plddt,
                x="Residue",
                y="pLDDT",
                title="Amino Asit Pozisyonuna Göre ESMFold Güvenilirlik (pLDDT) Profili",
                labels={"Residue": "Amino Asit Pozisyonu", "pLDDT": "pLDDT Skoru"},
            )
            fig_plddt.add_hline(
                y=90,
                line_dash="dash",
                line_color="#10b981",
                annotation_text="Çok Yüksek (>90)",
            )
            fig_plddt.add_hline(
                y=70,
                line_dash="dash",
                line_color="#3b82f6",
                annotation_text="Yüksek (70-90)",
            )
            fig_plddt.add_hline(
                y=50,
                line_dash="dash",
                line_color="#f59e0b",
                annotation_text="Düşük (50-70)",
            )
            fig_plddt.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#f8fafc",
            )
            st.plotly_chart(fig_plddt, use_container_width=True)

          # py3Dmol Görselleştirme
          view_esm = py3Dmol.view(width=900, height=520)
          view_esm.addModel(pdb_data, "pdb")

          if color_by_pLDDT:
            # B-factor / pLDDT Spektrumu ile Renklendirme
            view_esm.setStyle({
                "cartoon": {
                    "colorscheme": {
                        "prop": "b",
                        "gradient": "roygb",
                        "min": 50,
                        "max": 90,
                    }
                }
            })
          else:
            view_esm.setStyle({"cartoon": {"color": "spectrum"}})

          view_esm.zoomTo()
          showmol(view_esm, height=520, width=900)

          st.download_button(
              label="💾 Üretilen PDB Dosyasını İndir",
              data=pdb_data,
              file_name="esmfold_predicted_structure.pdb",
              mime="chemical/x-pdb",
              key="download_pdb_btn",
          )
        else:
          st.error(
              f"ESMFold API Hatası (Durum Kodu: {response.status_code}). Lütfen"
              " diziyi kontrol edin."
          )
      except Exception as err:
        st.error(f"Bağlantı sırasında bir hata oluştu: {err}")


with tab_simulasyon:
    # --- 3D PROTEİN DESTEKLİ SİMÜLATÖR KODU BURADAN BAŞLIYOR ---
    st.header("🧬 Sentetik Genom Sandbox & Evrim Simülatörü v2.5")
    st.caption(
        "Türkiye'nin ilk interaktif web tabanlı genetik stres, mutasyon ve 3D protein görselleştirmeli evrim motoru."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("1. Sentetik Genom Tasarımı")

        # Temel Genler
        has_rad_resistance = st.checkbox("🛡️ Radyasyon Direnç Geni (PprA)")
        has_antifreeze = st.checkbox("❄️ Antifriz Proteini (AFP)")
        has_heat_shock = st.checkbox("🔥 Isı Şoku Proteini (HSP70)")
        has_metabolism_boost = st.checkbox("⚡ Yüksek Metabolizma (ATP+)")
        
        # Sentetik Genler
        has_piezophile = st.checkbox("🌊 Yüksek Basınç Direnç Geni (Piezo-Protein)")
        has_heavy_metal = st.checkbox("🧪 Ağır Metal/Asit Nötralizasyonu (Chelatin)")
        has_bioluminescence = st.checkbox("💡 Biyolüminesans / Işık Saçma (GFP-Luciferase)")
        has_biofilm = st.checkbox("🛡️ Koruyucu Biyofilm Katmanı (Exopolysaccharide)")

        # Çevresel Stres Faktörü
        st.subheader("2. Çevresel Koşullar")
        environment = st.selectbox(
            "Simülasyon Ortamı Seçin:",
            [
                "Çernobil Tipi Yüksek Radyasyon",
                "Antarktika Derin Buzulu (-30°C)",
                "Volkanik Termal Kaynak (+95°C)",
                "Mariana Çukuru Derin Deniz Basıncı (1100 atm)",
                "Asidik Toksik Atık Gölü (pH 1.5)",
                "Standart Laboratuvar Biyo-Reaktörü",
            ],
        )

        start_sim = st.button("🚀 Evrim Simülasyonunu Başlat", type="primary", use_container_width=True)

    with col2:
        st.subheader("3. Gerçek Zamanlı Analiz & Popülasyon Dinamiği")

        if start_sim:
            # Başlangıç Parametreleri
            generations = 50
            population = 100
            pop_history = []
            mutation_events = []

            # Biyolojik Basit Uyum (Fitness) Puanı Hesaplama
            fitness = 1.0

            if "Radyasyon" in environment:
                fitness += 2.5 if has_rad_resistance else -2.0
            elif "Buzul" in environment:
                fitness += 2.5 if has_antifreeze else -2.0
            elif "Volkanik" in environment:
                fitness += 2.5 if has_heat_shock else -2.0
            elif "Mariana" in environment:
                fitness += 2.5 if has_piezophile else -2.0
            elif "Asidik" in environment:
                fitness += 2.5 if has_heavy_metal else -2.0

            # İkincil Avantajlar
            if has_metabolism_boost:
                fitness += 0.4
            if has_biofilm:
                fitness += 0.3
            if has_bioluminescence:
                fitness += 0.1

            # Simülasyon Döngüsü
            chart_holder = st.empty()
            status_holder = st.empty()

            current_pop = population
            for gen in range(1, generations + 1):
                growth_rate = (fitness * 0.1) + np.random.normal(0, 0.05)
                current_pop = int(
                    current_pop * (1 + growth_rate) * (1 - current_pop / 1000)
                )

                if np.random.rand() < 0.12:
                    delta_fitness = np.random.choice([-0.4, 0.6])
                    fitness += delta_fitness
                    mut_type = "Pozitif (Adaptasyon Arttı)" if delta_fitness > 0 else "Zararlı (Sistemik Bozulma)"
                    mutation_events.append(f"Nesil {gen}: Nokta Mutasyonu ({mut_type})")

                if current_pop <= 0:
                    current_pop = 0
                    pop_history.append({"Nesil": gen, "Popülasyon": current_pop, "Adaptasyon Skoru": round(fitness, 2)})
                    break

                pop_history.append({"Nesil": gen, "Popülasyon": current_pop, "Adaptasyon Skoru": round(fitness, 2)})

                df_sim = pd.DataFrame(pop_history)
                fig = px.line(
                    df_sim,
                    x="Nesil",
                    y="Popülasyon",
                    title=f"Popülasyon Grafiği - {environment}",
                    line_shape="spline",
                )
                fig.update_traces(line_color="#00FF88", line_width=3)
                chart_holder.plotly_chart(fig, use_container_width=True)

                status_holder.info(
                    f"⏳ **Nesil:** {gen}/{generations} | 🦠 **Canlı Popülasyon:** {current_pop} | 📊 **Adaptasyon Skoru:** {round(fitness, 2)}"
                )
                time.sleep(0.04)

            # Sonuçlar ve Ek Özellikler
            if current_pop > 0:
                st.success(
                    "🎉 **ORGANİZMA HAYATTA KALDI!** Sentetik canlı seçilen ortama tam adaptasyon sağladı."
                )
                
                # --- SENTETİK BİYO-PASAPORT VE FASTA ÜRETİCİ ---
                st.divider()
                st.subheader("📄 Sentetik Canlı Pasaportu & FASTA Genetik Kodu")
                
                nucleotides = ["A", "C", "G", "T"]
                fasta_sequence = "".join(np.random.choice(nucleotides, size=180))
                formatted_fasta = f">AegisBio_SynOrganism_Gen{generations}_{environment[:4].upper()}\n"
                formatted_fasta += "\n".join([fasta_sequence[i:i+60] for i in range(0, len(fasta_sequence), 60)])

                c_info1, c_info2 = st.columns(2)
                with c_info1:
                    st.write(f"**Biyo-Güvenlik Seviyesi:** `BSL-1 (Sentetik Contained)`")
                    st.write(f"**Fenotip:** {'Işık Saçan ' if has_bioluminescence else ''}{'Dirençli ' if has_biofilm else ''}Sentetik Bakteri")
                    st.write(f"**Uyum Başarı Oranı:** `%{(current_pop/1000)*100:.1f}`")
                
                with c_info2:
                    st.download_button(
                        label="📥 Genom Kodu İndir (.FASTA)",
                        data=formatted_fasta,
                        file_name="synthetic_organism_genome.fasta",
                        mime="text/plain",
                    )

                # --- 3D PROTEİN YAPISI GÖRSELLEŞTİRİCİ ---
                st.divider()
                st.subheader("🧬 Eksprese Edilen Sentetik Protein Yapısı (3D)")
                st.caption("Organizmanın hayatta kalmasını sağlayan baskın enzimin 3D interaktif modeli (Sürükleyip döndürebilirsiniz):")

                # Seçilen gene göre PDB ID eşleştirme
                pdb_id = "1BNA"  # Varsayılan B-DNA
                if has_bioluminescence:
                    pdb_id = "1EMA"  # GFP (Yeşil Floresan Proteini)
                elif has_antifreeze:
                    pdb_id = "1W1I"  # Antifriz Proteini
                elif has_heat_shock:
                    pdb_id = "2KHO"  # Heat Shock Proteini (HSP)
                elif has_rad_resistance:
                    pdb_id = "1PPR"  # PprA Proteini

                try:
                    # py3Dmol Görünümü Oluşturma
                    xyzview = py3Dmol.view(query=f"pdb:{pdb_id}")
                    xyzview.setStyle({"cartoon": {"color": "spectrum"}})
                    xyzview.addSurface(py3Dmol.VDW, {"opacity": 0.4, "color": "white"})
                    xyzview.setBackgroundColor("#0D1117")
                    xyzview.zoomTo()
                    
                    # Streamlit üzerinde gösterme
                    showmol(xyzview, height=400, width=700)
                except Exception as e:
                    st.warning(f"3D Model Yüklenirken Hata Oluştu: {e}")

            else:
                st.error(
                    "💀 **POPÜLASYON TÜKENDİ.** Genom tasarımı çevresel stres faktörüne dayanamadı."
                )

            if mutation_events:
                with st.expander("🧬 Simülasyon Sırasında Oluşan Mutasyon Günlüğü"):
                    for m in mutation_events:
                        st.write(f"- {m}")
        else:
            st.info(
                "Simülasyonu başlatmak için sol panelden gen kombinasyonlarını seçin ve butona basın."
            )
    # --- 3D PROTEİN DESTEKLİ SİMÜLATÖR KODU BURADA BİTİYOR ---
