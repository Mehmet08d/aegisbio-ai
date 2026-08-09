import streamlit as st
from Bio import Entrez, SeqIO
from Bio.Align import PairwiseAligner
from Bio.Align import substitution_matrices
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import requests
import py3Dmol
from stmol import showmol
import time

# Google Generative AI SDK Güvenli İçe Aktarımı
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# --- SAYFA KONFİGÜRASYONU ---
st.set_page_config(
    page_title="Mgen Analysis | Enterprise Bio-Suite & De-Extinction Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE BAŞLATMA ---
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [
        {"role": "assistant", "content": "Merhaba! Ben Mgen Analysis Biyo-Danışmanıyım. Biyoinformatik analizleriniz, ESMFold 3D modelleriniz veya mitokondriyal genetik hakkındaki sorularınızı yanıtlamaya hazırım."}
    ]
if "son_analiz" not in st.session_state:
    st.session_state["son_analiz"] = "Kullanıcı henüz bir biyoinformatik analiz çalıştırmadı."
if "son_sequence" not in st.session_state:
    st.session_state["son_sequence"] = "MTPMRTINPLMKLINHSFIDLPTPSNISAWWNFGSLLGACLILQITTGLFLAMHYSPDASTAFSSIAHITRDVNYGWIRYLHANGASMFFICLFLHIGRGLYYGSFLYSETWNIGIILLLATMATAFMGYVLPWGQMSFWGATVITNLFSAIPYIGTNLVEWIWGGFSVDKATLTRFFAFHFILPFIMALAAVHLTFLHETGSNNPTGLNSDSDKIPFHPYYTIKDILGALLLLALLLLTLFSPDLLGDPDNYTLANPLNTPPHIKPEWYFLFAYTILRSVPNKLGGVLALLLSILILIVIPILHTSKQRSMMFRPLSQCLFWLLTADLLTLTWIGGQPVEHPYITIGQMASILYFSIILILMPISGIIEDKMLKWN"

# --- GLASSMORPHISM UI TASARIMI ---
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

HAZIR_TURLER = {
    # ==========================================
    # 🦖 1. NESLİ TÜKENMİŞ & DE-EXTINCTION ADAYLARI
    # ==========================================
    "🦣 Yünlü Mamut (Mammuthus primigenius) [Tükenmiş]": "NC_007596.2",
    "🦣 Kolombiya Mamutu (Mammuthus columbi) [Tükenmiş]": "NC_015529.1",
    "🦣 Step Mamutu (Mammuthus trogontherii) [Tükenmiş]": "NC_025796.1",
    "🦴 Amerika Mastodonu (Mammut americanum) [Tükenmiş]": "NC_009592.1",
    "🐅 Tasman Kaplanı / Thylacine (Thylacinus cynocephalus) [Tükenmiş]": "NC_028319.1",
    "⚔️ Kılıç Dişli Kaplan (Smilodon fatalis) [Tükenmiş]": "NC_030752.1",
    "🦤 Dodo Kuşu (Raphus cucullatus) [Tükenmiş]": "NC_008263.1",
    "🕊️ Yolcu Güvercini (Ectopistes migratorius) [Tükenmiş]": "NC_026723.1",
    "🦅 Dev Moa (Dinornis robustus) [Tükenmiş]": "NC_002674.1",
    "🦅 Küçük Çalı Moası (Anomalopteryx didiformis) [Tükenmiş]": "NC_002773.1",
    "🦅 Haast Kartalı (Hieraaetus moorei) [Tükenmiş]": "NC_041113.1",
    "🦴 Neandertal İnsanı (Homo neanderthalensis) [Tükenmiş]": "NC_011137.1",
    "🦴 Denisova İnsanı (Denisova hominin) [Tükenmiş]": "NC_013993.1",
    "🦛 Madagaskar Cüce Hipopotamı (Hippopotamus lemerlei) [Tükenmiş]": "NC_025287.1",
    "🦬 Mağara Ayısı (Ursus spelaeus) [Tükenmiş]": "NC_011112.1",
    "🦁 Mağara Aslanı (Panthera spelaea) [Tükenmiş]": "NC_028320.1",
    "🦏 Yünlü Gergedan (Coelodonta antiquitatis) [Tükenmiş]": "NC_012681.1",
    "🦬 Steller Deniz İneği (Hydrodamalis gigas) [Tükenmiş]": "NC_031350.1",
    "🦡 Dev Tembel Hayvan (Mylodon darwinii) [Tükenmiş]": "NC_037466.1",
    "🦡 Glyptodon (Glyptodon reticulatus) [Tükenmiş]": "NC_028520.1",
    "🦓 Quagga (Equus quagga quagga) [Tükenmiş]": "NC_008129.1",
    "🦌 Dev İrlanda Geyiği (Megaloceros giganteus) [Tükenmiş]": "NC_025780.1",
    "🐂 Aurochs / Yaban Öküzü (Bos primigenius) [Tükenmiş]": "NC_013996.1",
    "🦆 Labrador Ördeği (Camptorhynchus labradorius) [Tükenmiş]": "NC_039655.1",
    "🦜 Carolina Papağanı (Conuropsis carolinensis) [Tükenmiş]": "NC_037930.1",
    "🐧 Büyük Auk / Penguen (Pinguinus impennis) [Tükenmiş]": "NC_038234.1",
    "🦡 Falkland Tilkisi (Dusicyon australis) [Tükenmiş]": "NC_028542.1",
    "🐺 Japon Kurdu (Canis lupus hodophilax) [Tükenmiş]": "NC_039804.1",
    "🦅 Fil Kuşu (Aepyornis maximus) [Tükenmiş]": "NC_026821.1",
    "🦅 Muller Fil Kuşu (Mullerornis agilis) [Tükenmiş]": "NC_026820.1",

    # ==========================================
    # 👤 2. PRİMATLAR & İNSANSILAR
    # ==========================================
    "👤 Modern İnsan (Homo sapiens)": "NC_012920.1",
    "🐒 Şempanze (Pan troglodytes)": "NC_001643.1",
    "🐒 Bonobo (Pan paniscus)": "NC_001644.1",
    "🦍 Batan Kıyı Gorili (Gorilla gorilla)": "NC_001645.1",
    "🦍 Doğu Gorili (Gorilla beringei)": "NC_011120.1",
    "🦧 Sumatra Orangutanı (Pongo abelii)": "NC_002083.1",
    "🦧 Borneo Orangutanı (Pongo pygmaeus)": "NC_001646.1",
    "🐒 Ak Yanaklı Gibon (Nomascus leucogenys)": "NC_002082.1",
    "🐒 Lar Gibonu (Hylobates lar)": "NC_002081.1",
    "🐒 Şebek / Rhesus Maymunu (Macaca mulatta)": "NC_005943.1",
    "🐒 Şebek / Yengeç Yiyen Maymun (Macaca fascicularis)": "NC_000868.1",
    "🐒 Anubis Babunu (Papio anubis)": "NC_001992.1",
    "🐒 Hamadryas Babunu (Papio hamadryas)": "NC_008218.1",
    "🐒 Yeşil Maymun (Chlorocebus sabaeus)": "NC_008066.1",
    "🐒 Altın Maymun (Rhinopithecus roxellana)": "NC_008219.1",
    "🐒 Ortak İpek Maymun / Marmoset (Callithrix jacchus)": "NC_002556.1",
    "🐒 Sincap Maymunu (Saimiri sciureus)": "NC_008217.1",
    "🐒 Beyaz Yüzlü Capuchin (Cebus capucinus)": "NC_028540.1",
    "🐒 Halka Kuyruklu Lemur (Lemur catta)": "NC_004025.1",
    "🐒 Ay-ay / Aye-aye (Daubentonia madagascariensis)": "NC_000898.1",

    # ==========================================
    # 🦁 3. ETÇİLLER (ETÇİL MEMELİLER)
    # ==========================================
    "🦁 Aslan (Panthera leo)": "NC_028321.1",
    "🐅 Kaplan (Panthera tigris)": "NC_010642.1",
    "🐆 Leopar (Panthera pardus)": "NC_010641.1",
    "🐆 Kar Leoparı (Panthera uncia)": "NC_010638.1",
    "🐆 Yaguar (Panthera onca)": "NC_010640.1",
    "🐆 Çita (Acinonyx jubatus)": "NC_005212.1",
    "🐱 Puma (Puma concolor)": "NC_016470.1",
    "🐱 Vaşak (Lynx lynx)": "NC_028323.1",
    "🐱 Evcil Kedi (Felis catus)": "NC_001700.1",
    "🐱 Yaban Kedisi (Felis silvestris)": "NC_016433.1",
    "🐺 Gri Kurt (Canis lupus)": "NC_009686.1",
    "🐕 Evcil Köpek (Canis lupus familiaris)": "NC_002008.4",
    "🦮 Çakal (Canis aureus)": "NC_028328.1",
    "🦊 Kızıl Tilki (Vulpes vulpes)": "NC_008434.1",
    "🦊 Kutup Tilkisi (Vulpes lagopus)": "NC_026529.1",
    "🦮 Kır Kurdu / Kayot (Canis latrans)": "NC_013445.1",
    "🦮 Dingo (Canis lupus dingo)": "NC_011211.1",
    "🐻 Boz Ayı (Ursus arctos)": "NC_003427.1",
    "🐻 Kutup Ayısı (Ursus maritimus)": "NC_003428.1",
    "🐻 Amerikan Siyah Ayısı (Ursus americanus)": "NC_000892.1",
    "🐼 Dev Panda (Ailuropoda melanoleuca)": "NC_009484.1",
    "🦝 Rakun (Procyon lotor)": "NC_009101.1",
    "🦦 Su Samuru (Lutra lutra)": "NC_011358.1",
    "🦡 Porsuk (Meles meles)": "NC_010298.1",
    "🦡 Kokarca (Mephitis mephitis)": "NC_016462.1",
    "🦔 Benekli Sırtlan (Crocuta crocuta)": "NC_020638.1",
    "🦔 Çizgili Sırtlan (Hyaena hyaena)": "NC_020637.1",
    "🦡 Mirket (Suricata suricatta)": "NC_028327.1",

    # ==========================================
    # 🐘 4. OTLUÇILLAR, HOOFED & HORTUMLULAR
    # ==========================================
    "🐘 Asya Fili (Elephas maximus) [Taşıyıcı Konak]": "NC_005129.2",
    "🐘 Afrika Savan Fili (Loxodonta africana)": "NC_000934.1",
    "🐘 Afrika Orman Fili (Loxodonta cyclotis)": "NC_006613.1",
    "🦏 Beyaz Gergedan (Ceratotherium simum)": "NC_001808.1",
    "🦏 Siyah Gergedan (Diceros bicornis)": "NC_012682.1",
    "🦏 Hint Gergedanı (Rhinoceros unicornis)": "NC_001779.1",
    "🦛 Su Aygırı (Hippopotamus amphibius)": "NC_000889.1",
    "🦒 Zürafa (Giraffa camelopardalis)": "NC_020610.1",
    "🦒 Okapi (Okapia johnstoni)": "NC_020609.1",
    "ZEBRA Yaban Eşeği / Zebra (Equus quagga)": "NC_008133.1",
    "🐴 Evcil At (Equus caballus)": "NC_001640.1",
    "🐴 Przewalski Atı (Equus przewalskii)": "NC_008220.1",
    "🫏 Yaban Eşeği (Equus africanus)": "NC_007602.1",
    "🐫 Hörgüçlü Deve (Camelus bactrianus)": "NC_009628.1",
    "Tek Hörgüçlü Deve (Camelus dromedarius)": "NC_009849.1",
    "🦙 Lama (Lama glama)": "NC_012108.1",
    "🦙 Alpaka (Vicugna pacos)": "NC_012109.1",
    "🐂 Sığır / İnek (Bos taurus)": "NC_006853.1",
    "🐂 Yak (Bos mutus)": "NC_011221.1",
    "🦬 Amerikan Bizonu (Bison bison)": "NC_012346.1",
    "🐐 Yaban Keçisi (Capra aegagrus)": "NC_020680.1",
    "🐐 Evcil Keçi (Capra hircus)": "NC_005044.2",
    "🐑 Evcil Koyun (Ovis aries)": "NC_001941.1",
    "🦌 Kızıl Geyik (Cervus elaphus)": "NC_007704.1",
    "🦌 Sığın / Ren Geyiği (Rangifer tarandus)": "NC_007703.1",
    "🦌 Bataklık Geyiği (Alces alces)": "NC_020681.1",
    "🐖 Yaban Domuzu (Sus scrofa)": "NC_000845.1",

    # ==========================================
    # 🐋 5. DENİZ MEMELİLERİ (BALİNALAR & FOKLAR)
    # ==========================================
    "Mavi Balina (Balaenoptera musculus)": "NC_001601.1",
    "🐋 Fin Balinası (Balaenoptera physalus)": "NC_001321.1",
    "🐋 Kambur Balina (Megaptera novaeangliae)": "NC_006927.1",
    "🐋 Katil Balina / Orka (Orcinus orca)": "NC_023889.1",
    "🐬 Mutur / Şişe Burunlu Yunus (Tursiops truncatus)": "NC_012059.1",
    "🐬 Çizgili Yunus (Stenella coeruleoalba)": "NC_012061.1",
    "🐋 İspermecet Balinası (Physeter macrocephalus)": "NC_002504.1",
    "🐋 Deniz Tacir Balinası / Narval (Monodon monoceros)": "NC_005279.1",
    "🐋 Beyaz Balina / Beluga (Delphinapterus leucas)": "NC_001325.1",
    "🦭 Bıyıklı Fok (Erignathus barbatus)": "NC_002812.1",
    "🦭 Gri Fok (Halichoerus grypus)": "NC_001602.1",
    "🦭 Deniz Fili (Mirounga leonina)": "NC_002307.1",
    "🦭 Deniz Aslanı (Zalophus californianus)": "NC_004020.1",
    "🦭 Deniz Aygırı / Walrus (Odobenus rosmarus)": "NC_004021.1",
    "🦛 Manati (Trichechus manatus)": "NC_001291.1",

    # ==========================================
    # 🦘 6. KESELİLER & TEK DELİKLİLER
    # ==========================================
    "🦘 Numbat (Myrmecobius fasciatus) [Taşıyıcı Konak]": "NC_011949.1",
    "🦘 Kızıl Kanguru (Macropus rufus)": "NC_023522.1",
    "🦘 Doğu Gri Kangurusu (Macropus giganteus)": "NC_020600.1",
    "🐨 Koala (Phascolarctos cinereus)": "NC_021975.1",
    "🦡 Tasman Canavarı (Sarcophilus harrisii)": "NC_013770.1",
    "🦡 Kuzey Vombatı (Lasiorhinus latifrons)": "NC_003322.1",
    "🦡 Şeker Planörü (Petaurus breviceps)": "NC_024602.1",
    "🦡 Virginia Opossumu (Didelphis virginiana)": "NC_001610.1",
    "🥚 Ornitorenk (Ornithorhynchus anatinus)": "NC_000891.1",
    "🥚 Dikenli Karıncayiyen / Ekidne (Tachyglossus aculeatus)": "NC_003321.1",

    # ==========================================
    # 🦅 7. KUŞLAR
    # ==========================================
    "🦅 Kaya Kartalı (Aquila chrysaetos)": "NC_027656.1",
    "🦅 Kel Kartal (Haliaeetus leucocephalus)": "NC_008550.1",
    "🦅 Bayağı Doğan (Falco peregrinus)": "NC_029348.1",
    "🦉 Bayağı Baykuş (Tyto alba)": "NC_028435.1",
    "🦉 Puhu Baykuşu (Bubo bubo)": "NC_024285.1",
    "🦅 Bayağı Şahin (Buteo buteo)": "NC_028438.1",
    "🐧 İmparator Penguen (Aptenodytes forsteri)": "NC_025539.1",
    "🐧 Adélie Pengueni (Pygoscelis adeliae)": "NC_004418.1",
    "🦩 Flamingo (Phoenicopterus roseus)": "NC_028442.1",
    "🦆 Yeşilbaş Ördek (Anas platyrhynchos)": "NC_009684.1",
    "🦤 Deve Kuşu (Struthio camelus)": "NC_002785.1",
    "🦤 Emu (Dromaius novaehollandiae)": "NC_002784.1",
    "🦤 Kaspian Tepeli Devekuşu / Kasovar (Casuarius casuarius)": "NC_002783.1",
    "🦤 Kiwi Kuşu (Apteryx haastii)": "NC_002782.1",
    "🕊️ Kaya Güvercini (Columba livia)": "NC_013978.1",
    "🐓 Evcil Tavuk (Gallus gallus)": "NC_001323.1",
    "🪿 Yaban Kazı (Anser anser)": "NC_011196.1",
    "🦚 Bayağı Tavuskuşu (Pavo cristatus)": "NC_018043.1",
    "🦜 Kakapo Papağanı (Strigops habroptilus)": "NC_005932.1",
    "🦜 Kea Papağanı (Nestor notabilis)": "NC_029411.1",
    "🦅 Bayağı Kuzgun (Corvus corax)": "NC_029347.1",
    "🐦 Bayağı Serçe (Passer domesticus)": "NC_025612.1",
    "🐦 Sinek Kuşu / Kolibri (Archilochus alexandri)": "NC_010099.1",
    "🦤 Bayağı Pelikan (Pelecanus Onocrotalus)": "NC_028441.1",

    # ==========================================
    # 🐊 8. SÜRÜNGENLER & AMFİBİLER
    # ==========================================
    "🐊 Nil Timsahı (Crocodylus niloticus)": "NC_001920.1",
    "🐊 Amerikan Aligatörü (Alligator mississippiensis)": "NC_001922.1",
    "🐊 Gavial (Gavialis gangeticus)": "NC_021088.1",
    "Komodo Ejderi (Varanus komodoensis)": "NC_010974.1",
    "🦎 Bayağı Bukalemun (Chamaeleo calyptratus)": "NC_028434.1",
    "🦎 Yeşil İguana (Iguana iguana)": "NC_002793.1",
    "🐍 Kral Kobra (Ophiophagus hannah)": "NC_011394.1",
    "🐍 Bayağı Engerek (Vipera berus)": "NC_028439.1",
    "🐍 Piton Yılanı (Python bivittatus)": "NC_021479.1",
    "🐍 Yeşil Anakonda (Eunectes murinus)": "NC_028437.1",
    "🐢 Dev Galapagos Kaplumbağası (Chelonoidis niger)": "NC_028436.1",
    "🐢 Çorbacı Deri Sırtlı Deniz Kaplumbağası (Dermochelys coriacea)": "NC_028440.1",
    "🐢 Çorba Kaplumbağası (Chelonia mydas)": "NC_000886.1",
    "🦎 Tuatara (Sphenodon punctatus)": "NC_004815.1",
    "🐸 Afrika Pençeli Kurbağası (Xenopus laevis)": "NC_001573.1",
    "🐸 Boğa Kurbağası (Rana catesbeiana)": "NC_002805.1",
    "🐸 Aksolotl / Semender (Ambystoma mexicanum)": "NC_008229.1",
    "🦎 Lekeli Semender (Salamandra salamandra)": "NC_011538.1",

    # ==========================================
    # 🦈 9. BALIKLAR & DENİZ CANLILARI
    # ==========================================
    "🦈 Büyük Beyaz Köpekbalığı (Carcharodon carcharias)": "NC_022415.1",
    "🦈 Balina Köpekbalığı (Rhincodon typus)": "NC_023456.1",
    "🦈 Çekiç Başlı Köpekbalığı (Sphyrna lewini)": "NC_022830.1",
    "🐟 Bayağı Ton Balığı (Thunnus thynnus)": "NC_004386.1",
    "🐟 Atlantik Somonu (Salmo salar)": "NC_001960.1",
    "🐟 Bayağı Morina / Mezgit (Gadus morhua)": "NC_002081.2",
    "🐟 Zebra Balığı (Danio rerio)": "NC_002333.2",
    "🐟 Japon Balığı (Carassius auratus)": "NC_002811.1",
    "🐟 Bayağı Hamsi (Engraulis encrasicolus)": "NC_028080.1",
    "🐟 Bayağı Mersin Balığı (Acipenser sturio)": "NC_028122.1",
    "🐟 Latimeria / Sölakant (Latimeria chalumnae) [Canlı Fosil]": "NC_000829.1",
    "🐟 Avustralya Akciğerli Balığı (Neoceratodus forsteri)": "NC_000868.2",

    # ==========================================
    # 🐝 10. OMURGASIZLAR & BÖCEKLER
    # ==========================================
    "🐝 Bal Arısı (Apis mellifera)": "NC_001566.1",
    "🪰 Sirke Sineği (Drosophila melanogaster)": "NC_001709.1",
    "🦟 Sıtma Sivrisineği (Anopheles gambiae)": "NC_002084.1",
    "🦋 Kral Kelebeği (Danaus plexippus)": "NC_008230.1",
    "🪲 Bayağı Hamamböceği (Periplaneta americana)": "NC_006076.1",
    "🐙 Dev Pasifik Ahtapotu (Enteroctopus dofleini)": "NC_029703.1",
    "🦑 Dev Mürekkep Balığı (Architeuthis dux)": "NC_021147.1",
    "🦞 Istakoz (Homarus americanus)": "NC_015600.1",
    "🦐 Yaban Karidesi (Penaeus monodon)": "NC_002184.1",
    "🪱 Nematod / Solucan (Caenorhabditis elegans)": "NC_001328.1",

    # ==========================================
    # ✏️ ÖZEL KULLANICI GİRDİSİ
    # ==========================================
    "✏️ Custom (Özel NCBI Accession ID)": "CUSTOM"
}

GEN_KOD_HARITASI = {
    "🧬 TÜM 13 GENİ BİRLİKTE TARA (Toplu Analiz)": "ALL_13",
    "ND1 (Complex I)": "ND1", "ND2 (Complex I)": "ND2", "ND3 (Complex I)": "ND3",
    "ND4 (Complex I)": "ND4", "ND4L (Complex I)": "ND4L", "ND5 (Complex I)": "ND5", "ND6 (Complex I)": "ND6",
    "CYTB (Complex III)": "CYTB", "COX1 (Complex IV)": "COX1", "COX2 (Complex IV)": "COX2",
    "COX3 (Complex IV)": "COX3", "ATP6 (Complex V)": "ATP6", "ATP8 (Complex V)": "ATP8"
}

st.markdown("""
    <div class="brand-header">
        <span class="academic-badge">Institutional Genomic Suite v4.0 Enterprise</span>
        <h1 class="brand-title">Mgen Analysis</h1>
        <p style='color: #94a3b8; font-size: 1.1rem; max-width: 800px; margin: 0 auto;'>
            Mitonuclear Incompatibility Engine, De-Extinction Genomic Analytics, ESMFold 3D Prediction & Universal Gemini AI Bio-LLM
        </p>
    </div>
""", unsafe_allow_html=True)

# ANA SEKMELER
tab_analiz, tab_3d_view, tab_esmfold, tab_ai_bot = st.tabs([
    "🚀 Biyoinformatik Analiz Engine", 
    "🧬 py3Dmol PDB Görselleştirici", 
    "🔮 ESMFold Anlık 3D Yapı Tahmini",
    "🤖 Gemini AI Biyo-Danışman"
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
    """Biopython PairwiseAligner ile güvenli global hizalama"""
    try:
        aligner = PairwiseAligner()
        aligner.mode = 'global'
        alignments = aligner.align(seq1, seq2)
        if len(alignments) > 0:
            aligned = alignments[0]
            aligned_seq1 = "".join([seq1[i] if i != -1 else "-" for i in aligned.indices[0]])
            aligned_seq2 = "".join([seq2[j] if j != -1 else "-" for j in aligned.indices[1]])
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
    user_email = st.sidebar.text_input("NCBI E-posta", "researcher@mgenanalysis.org", key="ncbi_email_input")

    tur1_secim = st.sidebar.selectbox("1. Tür / Taşıyıcı Konak", list(HAZIR_TURLER.keys()), index=0, key="tur1_select")
    numbat_acc = st.sidebar.text_input("1. Tür NCBI Kodu", "NC_011949.1", key="tur1_acc_input") if HAZIR_TURLER[tur1_secim] == "CUSTOM" else HAZIR_TURLER[tur1_secim]

    tur2_secim = st.sidebar.selectbox("2. Tür / Hedef Canlı", list(HAZIR_TURLER.keys()), index=4, key="tur2_select")
    thylacine_acc = st.sidebar.text_input("2. Tür NCBI Kodu", "NC_028319.1", key="tur2_acc_input") if HAZIR_TURLER[tur2_secim] == "CUSTOM" else HAZIR_TURLER[tur2_secim]

    target_gene_raw = st.sidebar.selectbox("İncelenecek Mitokondriyal Gen", list(GEN_KOD_HARITASI.keys()), index=0, key="gene_select")
    crispr_mode = st.sidebar.checkbox("🧬 Mito-CRISPR gRNA Adaylarını Hesapla", value=True, key="crispr_check")
    run_button = st.sidebar.button("🚀 BİYOİNFORMATİK ANALİZİ BAŞLAT", key="run_analysis_btn")

    if run_button:
        selected_code = GEN_KOD_HARITASI[target_gene_raw]
        
        with st.spinner("Genom verileri NCBI üzerinden çekiliyor ve BLOSUM62 matrisleri işleniyor..."):
            rec1 = ncbi_genom_getir(numbat_acc, user_email)
            rec2 = ncbi_genom_getir(thylacine_acc, user_email)

        if not rec1 or not rec2:
            st.error("❌ NCBI üzerinden GenBank kayıtları çekilemedi. Bağlantınızı veya Accession kodlarını kontrol edin.")
        else:
            all_genes = ["ND1", "ND2", "ND3", "ND4", "ND4L", "ND5", "ND6", "CYTB", "COX1", "COX2", "COX3", "ATP6", "ATP8"]
            target_list = all_genes if selected_code == "ALL_13" else [selected_code]
            
            sonuclar = []
            tum_farklar = []

            for g_code in target_list:
                res = gen_analiz_et(g_code, rec1, rec2, crispr_mode)
                if res:
                    sonuclar.append(res)
                    tum_farklar.extend(res["farklar"])

            if sonuclar:
                df_res = pd.DataFrame(sonuclar)
                
                # Metrik Kartları
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("İncelenen Gen Sayısı", len(sonuclar))
                m2.metric("Toplam Mutasyon Noktası", len(tum_farklar))
                ort_risk = df_res["risk_score"].mean()
                m3.metric("Ortalama Uyumsuzluk Riski", f"%{ort_risk:.1f}")
                m4.metric("Kritik Uyumsuz Gen", df_res.loc[df_res['risk_score'].idxmax()]['gene'])

                st.markdown("---")

                # Görselleştirme Grafikleri
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    fig_bar = px.bar(
                        df_res, x="gene", y="risk_score", color="risk_score",
                        title="Gen Bazlı Mitokondriyal Uyumsuzluk Skoru (%)",
                        color_continuous_scale="Reds", labels={"risk_score": "Risk (%)", "gene": "Gen"}
                    )
                    fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f8fafc')
                    st.plotly_chart(fig_bar, use_container_width=True)

                with col_g2:
                    fig_mut = px.scatter(
                        df_res, x="gene", y="mutations", size="seq_len", color="risk_score",
                        title="Mutasyon Yükü vs Gen Uzunluğu",
                        color_continuous_scale="Viridis", labels={"mutations": "Mutasyon Sayısı", "gene": "Gen"}
                    )
                    fig_mut.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f8fafc')
                    st.plotly_chart(fig_mut, use_container_width=True)

                # Mutasyon Detay Tablosu
                if tum_farklar:
                    st.markdown("### 📋 Mutasyon ve Biyofiziksel Değişim Detayları")
                    df_farklar = pd.DataFrame(tum_farklar)
                    st.dataframe(df_farklar, use_container_width=True)

                # AI Analiz Özeti İçin State Kaydı
                st.session_state["son_sequence"] = sonuclar[0]["seq1"]
                st.session_state["son_analiz"] = f"NCBI Analiz Özeti:\nTür 1: {rec1.description}\nTür 2: {rec2.description}\nToplam {len(tum_farklar)} amino asit mutasyonu tespit edildi. Ortalama Mitokondriyal Risk Skoru: %{ort_risk:.2f}."
                st.success("✅ Biyoinformatik analizi başarıyla tamamlandı. Detaylı biyolojik değerlendirme için 'Gemini AI Biyo-Danışman' sekmesini kullanabilirsiniz.")
            else:
                st.warning("Seçilen gen(ler) için geçerli kodlama dizileri (CDS) eşleştirilemedi.")

# ==========================================
# TAB 2: py3Dmol PDB GÖRSELLEŞTİRİCİ
# ==========================================
with tab_3d_view:
    st.markdown("### 🧬 PDB Yapısal Moleküler Görselleştirici")
    st.caption("PDB veritabanından çekilen veya yerel olarak yüklenen protein yapılarını etkileşimli olarak inceleyin.")

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        pdb_code = st.text_input("PDB Kodunu Girin (ör. 1A2C, 6VXX)", "1A2C", key="pdb_code_input")
    with c2:
        style_type = st.selectbox("Görselleştirme Stili", ["cartoon", "stick", "sphere", "line"], index=0, key="3d_style_select")
    with c3:
        color_scheme = st.selectbox("Renk Şeması", ["spectrum", "chain", "secondary structure", "residue"], index=0, key="3d_color_select")

    if pdb_code:
        try:
            view = py3Dmol.view(query=f'pdb:{pdb_code.lower()}')
            if style_type == "cartoon":
                view.setStyle({'cartoon': {'color': color_scheme}})
            elif style_type == "stick":
                view.setStyle({'stick': {}})
            elif style_type == "sphere":
                view.setStyle({'sphere': {}})
            else:
                view.setStyle({'line': {}})

            view.zoomTo()
            showmol(view, height=500, width=800)
        except Exception as e:
            st.error(f"PDB yapısı yüklenirken hata oluştu: {e}")

# ==========================================
# TAB 3: ESMFold 3D YAPI TAHMİNİ
# ==========================================
with tab_esmfold:
    st.markdown("### 🔮 Meta ESMFold Anlık Yapı Tahmin Modülü")
    st.caption("Amino asit dizisini girin, yapay zekâ destekli ESMFold API'si ile katlanmış 3D protein yapısını anında simüle edin.")

    default_fasta = st.session_state.get("son_sequence", "MTPMRTINPLMKLINHSFIDLPTPSNISAWWNFGSLLGACLILQITTGLFLAMHYSPDASTAFSSIA")
    input_fasta = st.text_area("Amino Asit Dizisi (FASTA / Düz Metin)", default_fasta, height=120, key="esmfold_fasta_area")

    if st.button("🔮 3D PROTEİN YAPISINI TAHMİN ET (ESMFold)", key="btn_esmfold_predict"):
        cleaned_seq = "".join(input_fasta.split()).upper()
        if len(cleaned_seq) < 10:
            st.warning("Geçerli bir 3D tahmin için en az 10 amino asitlik bir dizi giriniz.")
        else:
            with st.spinner("Meta ESMFold API servisine bağlanılıyor, yapı simüle ediliyor..."):
                try:
                    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
                    res = requests.post(url, data=cleaned_seq, headers={"Content-Type": "text/plain"}, timeout=30)
                    if res.status_code == 200:
                        pdb_data = res.text
                        st.success("✅ ESMFold 3D Yapı Tahmini Başarıyla Oluşturuldu!")
                        
                        view_esm = py3Dmol.view(width=800, height=500)
                        view_esm.addModel(pdb_data, "pdb")
                        view_esm.setStyle({'cartoon': {'color': 'spectrum'}})
                        view_esm.zoomTo()
                        showmol(view_esm, height=500, width=800)

                        st.download_button(
                            label="💾 PDB Dosyasını İndir",
                            data=pdb_data,
                            file_name="esmfold_predicted_structure.pdb",
                            mime="chemical/x-pdb",
                            key="download_pdb_btn"
                        )
                    else:
                        st.error(f"ESMFold API Servisi Hata Döndürdü (Status Code: {res.status_code}). Lütfen tekrar deneyin.")
                except Exception as ex:
                    st.error(f"ESMFold servisine erişilirken bir hata oluştu: {ex}")

# ==========================================
# TAB 4: GEMINI AI BİYO-DANIŞMAN
# ==========================================
with tab_ai_bot:
    st.markdown("### 🤖  AI Biyo-Danışman")
    st.caption("Genomik analizleriniz, mutasyon riskleri, CRISPR stratejileri veya biyoinformatik kodları hakkında yapay zeka ile doğrudan sohbet edin.")

    api_key = st.secrets.get("GEMINI_API_KEY", None)

    if not api_key:
        api_key = st.text_input("Gemini API Key Giriniz (Streamlit Secrets'a Eklenmediyse):", type="password", key="user_gemini_key")

    if api_key and HAS_GENAI:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')

            for msg in st.session_state["chat_messages"]:
                st.chat_message(msg["role"]).write(msg["content"])

            if prompt := st.chat_input("Biyoinformatik analizinizi veya biyolojik sorularınızı sorun..."):
                st.session_state["chat_messages"].append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)

                context_prompt = f"Bağlam: {st.session_state['son_analiz']}\n\nKullanıcı Sorusu: {prompt}"
                
                with st.spinner("Gemini Biyo-Danışman yanıtlıyor..."):
                    response = model.generate_content(context_prompt)
                    bot_reply = response.text

                st.session_state["chat_messages"].append({"role": "assistant", "content": bot_reply})
                st.chat_message("assistant").write(bot_reply)

        except Exception as e:
            st.error(f"Gemini API Bağlantı Hatası: {e}")
    else:
        st.info("💡 AI Biyo-Danışmanı kullanabilmek için geçerli bir Gemini API Anahtarı tanımlanmış olmalıdır.")
