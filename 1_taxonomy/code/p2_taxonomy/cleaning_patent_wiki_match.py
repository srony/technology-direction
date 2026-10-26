"""
p2_taxonomy/analysis_matching.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FAISS-based patent matching: wiki pages --> top-K nearest patents.

Wraps the logic from ``02_match_patents.py`` into a clean module interface
that imports from ``shared/`` and delegates to the original functions.

Main entry point: ``run_matching()``.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from shared.cleaning import flag_noise_pages
from shared.patent_cache import device_auto
from shared.utils import ensure_dir
from shared.patent_cache import (
    _list_part_files as _list_part_files_shared,
    _fast_row_count as _fast_row_count_shared,
    load_embeddings as _load_patent_embeddings,
    CACHE_ROOT as PATENT_EMBED_CACHE_ROOT,
    CORPUS_ROOT as PATENT_CORPUS_ROOT,
)

# ------------------------------------------------------------------
# Settings (can be overridden via run_matching kwargs)
# ------------------------------------------------------------------

SEED = 42

DEFAULT_EMBED_MODEL = "AI-Growth-Lab/PatentSBERTa"

K_TOP = 5
K_CAND = 100

PATENT_EMBED_BATCH = 256
WIKI_EMBED_BATCH = 64

IVF_NLIST = 4096
PQ_M = 64
PQ_NBITS = 8
TRAIN_N = 200_000

MAX_TITLE_CHARS = 300

PATENT_TITLE_COL = "appln_title"
WIKI_ID_COL = "page_title"
WIKI_TITLE_COL = "page_title"
WIKI_DESC_COL = "description"

# Noise patterns matched against title + description text
NOISE_TITLE_PATTERNS = [
    r"\bdisambiguation\b", r"\bmay refer to\b",
    r"^list of\b", r"^outline of\b", r"^timeline of\b",
    r"^glossary of\b", r"^index of\b", r"^comparison of\b",
    r"^history of\b", r"^portal:", r"^wikipedia:",
    r"^wikimedia", r"\bchristmas\b",
    # Biology: these never appear in tech page descriptions
    r"\bspecies of\b", r"\bgenus of\b",
]
# Patterns matched ONLY against the page title (too broad for description text).
NOISE_TITLE_ONLY_PATTERNS = [
    r"\(software\)", r"\(unicode block\)",
    r"\bapps?\b", r"\brocks?\b",
    # Non-tech Wikipedia disambiguation markers
    r"\(podcast\)", r"\(typeface\)", r"\(magazine\)", r"\(newspaper\)",
    # --- Product instance brands: PHONES ---
    r"^Samsung Galaxy\b",
    r"^(?:IPhone|IPad|IPod)\s",
    r"^Oppo\s",
    r"^Huawei\s",
    r"^Vivo\s[A-Z]\d",
    r"^Honor\s(?:Magic|GT|X|V|Play|\d)",
    r"^Xiaomi\s",
    r"^Redmi\s",
    r"^Meizu\s",
    r"^Pixel\s+\d",
    r"^Ninetology\b",
    r"^Itel\s[A-Z]",
    r"^LG\s(?:G\d|V\d|K\d|Q\d|W\d|Stylo|Velvet|Wing|Optimus|Roboking)",
    r"^HTC\s",
    r"^Nokia\s(?:C\d|G\d|X\d|N\d|\d)",
    r"^Moto\s[A-Z]\b",
    r"^OnePlus\s",
    r"^ZTE\s",
    r"^Realme\s",
    r"^Tecno\s",
    r"^Infinix\s",
    r"^(?:Sony\s)?Xperia\b",
    r"^T-Mobile\smyTouch\b",
    r"^BlackBerry\s(?!$)",
    r"^(?:Palm\sCentro|Treo\s\d)",
    r"^Cubot\b",
    r"^Zinwa\b",
    r"^Unihertz\b",
    r"^GeeksPhone\b",
    r"^Umidigi\b",
    r"^Cat\sS\d",
    # --- Product instance brands: CARS ---
    r"^Audi\s",
    r"^BMW\s",
    r"^Mercedes-Benz\s",
    r"^Toyota\s",
    r"^Honda\s",
    r"^Nissan\s",
    r"^Volkswagen\s",
    r"^Ford\s(?:Focus|Fiesta|Mustang|Explorer|Escape|Ranger|F-\d|Fusion|Edge|Bronco|Puma|Kuga|Mondeo|Transit|EcoSport|Maverick|GT|Expedition)",
    r"^Chevrolet\s",
    r"^Hyundai\s",
    r"^Kia\s",
    r"^Tesla\s(?:Model|Cybercab|Cybertruck|Semi|Roadster)",
    r"^Citro[eë]n\s",
    r"^Renault\s",
    r"^VinFast\b",
    r"^Lucid\s(?:Air|Gravity)",
    r"^Daihatsu\s",
    r"^Hawtai\s",
    r"^Karry\s",
    r"^Denza\s",
    r"^Neta\s",
    r"^Tata\s(?:Tiago|Nexon|Punch|Harrier|Safari|Tigor|Altroz)",
    r"^BYD\s",
    r"^Nio\s",
    r"^Foxtron\b",
    r"^Strati\b",
    r"^Polestar\s",
    # --- Product instance brands: TRAINS / BUSES ---
    r"^British Rail Class\b",
    r"^Alexander Dennis\b",
    r"^Stadler\s(?:FLIRT|METRO|KISS|GTW|Dosto)",
    # --- Product instance brands: DRONES ---
    r"^AeroVironment\b",
    r"^TAI\s(?:Aksungur|Anka)",
    r"^Baykar\b",
    r"^DJI\s",
    r"^Parrot\s(?:Anafi|Bebop|AR|Disco)",
    r"^Skydio\b",
    # --- Product instance brands: COMPUTERS ---
    r"^Macintosh\s(?:\d|Classic|II|LC|SE|Portable|Plus|Quadra|Performa|PowerBook|Color|XL)",
    # --- OS version strings (not platform concepts like Windows IoT / Android TV) ---
    r"^Windows\s+\d+,\s+version\s",
    r"^Windows\s+(?:7|8|10|Vista|XP|95|2000)\s+(?:edition)",
    # --- Big tech product instances (not technology concepts) ---
    r"^Amazon\s",
    r"^AWS\s",
    r"^Microsoft\s",
    r"^Google\s",
    r"^Oracle\s",
    r"^Apple\s(?!II\b)",
    r"^Meta\s",
    r"^Facebook\s",
    r"^Samsung\s",
    r"^Qualcomm\s",
]
# Patterns matched ONLY against the description (not title).
# Company/org pages are not technologies — "Spotify is a Swedish company" etc.
NOISE_DESC_ONLY_PATTERNS = [
    r"^.{0,60}\bis a(?:n)?\s.{0,30}\b(?:company|corporation|startup|conglomerate|subsidiary|manufacturer|automaker|carmaker)\b",
    # Organizations — lookbehinds protect "data organization", "memory organization" etc.
    r"^.{0,60}\bis a(?:n)?\s.{0,30}\b(?:(?<!data )(?<!memory )(?<!file )(?<!code )(?<!information )(?:organization|organisation)|not-for-profit|non-profit|nonprofit)\b",
    # Associations — tempered token excludes "by" (prevents "organised by the Federation" FPs);
    #   (?!\s+football) protects "association football" (soccer)
    r"^.{0,60}\bis a(?:n)?\s(?:(?!\bby\b).){0,30}\b(?:association(?!\s+football)|society|institute|foundation|agency|commission|committee|council|union|federation|bureau|authority)\b",
    # Consortiums — tight gap (.{0,15}) so only "X is a consortium" matches, not mentions
    r"^.{0,60}\bis a(?:n)?\s.{0,15}\bconsortium\b(?!.{0,15}\b(?:standard|specification|protocol|format)\b)",
    # Specific software products, not technology concepts
    r"^.{0,60}\bis a(?:n)?\s.{0,50}\bword processor\b",
    r"^.{0,60}\bis a(?:n)?\s.{0,50}\bdigital audio workstation\b",
    # Smartphone/phone product instances — tight gap so only "X is a smartphone" matches
    r"^.{0,60}\bis a(?:n)?\s.{0,30}\b(?:smartphone|mobile phone|cell phone|feature phone|phablet)\b(?!\s+(?:architecture|technology|protocol|standard|platform|framework|concept))",
    # Specific personal computer/home computer instances
    r"^.{0,60}\bis a(?:n)?\s.{0,80}\b(?:personal computer|home computer)\b.{0,80}\b(?:designed|manufactured|produced|sold)\b",
    # --- Pharmaceutical drugs (NOT tech concepts) ---
    r"^.{0,60}\bis a(?:n)?\s.{0,80}\bpharmaceutical\b",
    # --- Video / computer / mobile games — tempered token excludes server/engine/library/framework/toolkit;
    #     negative lookahead after keyword excludes "video game engine/server/development" ---
    r"^.{0,60}\bis a(?:n)?\s(?:(?!\blibrary\b|\bframework\b|\btoolkit\b|\bserver\b|\bengine\b).){0,30}\b(?:video game|computer game|mobile game)\b(?!\s+(?:engine|server|development|programming|design|technology|platform|middleware))",
    # --- Streaming / VOD services — tight gap (.{0,15}) so concepts like "Media-on-demand" aren't caught ---
    r"^.{0,60}\bis a(?:n)?\s.{0,15}\b(?:streaming service|video on demand|over-the-top streaming)\b",
    # --- Social networks / social media platforms — tight gap ---
    r"^.{0,60}\bis a(?:n)?\s.{0,30}\b(?:social network|social media)\b",
    # --- Web browsers — tempered token excludes "accessed/using/via" before keyword ---
    r"^.{0,60}\bis a(?:n)?\s(?:(?!\baccessed\b|\busing\b|\bthrough\b|\bvia\b).){0,30}\bweb browser\b",
    # --- Text / code / hex editors ---
    r"^.{0,60}\bis a(?:n)?\s.{0,50}\b(?:text editor|code editor|source.code editor|hex editor|HTML editor)\b",
    # --- Media / video / audio / music players — tight gap ---
    r"^.{0,60}\bis a(?:n)?\s.{0,30}\b(?:media player|video player|audio player|music player)\b",
    # --- Email clients ---
    r"^.{0,60}\bis a(?:n)?\s.{0,30}\b(?:email client|e-mail client|mail client)\b",
    # --- File management utilities ---
    r"^.{0,60}\bis a(?:n)?\s.{0,30}\b(?:file manager|file archiver|file compression|backup software|password manager)\b",
    # --- Non-tech people (no "author"; gap .{0,30} prevents catching tech products via distant keywords) ---
    r"^.{0,60}\bis a(?:n)?\s.{0,30}\b(?:journalist|comedian|politician|activist|singer|actor|actress|novelist|poet|musician|athlete|television presenter|businessperson|entrepreneur|rapper|filmmaker|cartoonist)\b",
    # --- Cryptocurrency / digital currency ---
    r"^.{0,60}\bis a(?:n)?\s.{0,30}\b(?:cryptocurrency|digital currency|virtual currency|stablecoin|altcoin)\b",
    # --- Mobile apps (not tech concepts) ---
    r"^.{0,60}\bis a(?:n)?\s.{0,30}\b(?:mobile app|smartphone app|ios app|android app)\b",
]
# Hardcoded page titles to exclude (too specific for regex patterns)
# Includes agent-reviewed non-technology pages and brand-specific products.
HARDCODED_EXCLUSION_TITLES = {
    # --- # (5) ---
    "1 the Road",
    "100K Pathogen Genome Project",
    "3-Arylpropiolonitriles",
    "4DCT",
    "5-base sequencing",
    # --- A (94) ---
    "A-scan ultrasound biometry",
    "ADHD rating scale",
    "AMPHORA",
    "ARDB",
    "ATP-binding motif",
    "Abdominal ultrasonography",
    "Absorption refrigerator",
    "Accelerated aging",
    "Accession number (bioinformatics)",
    "Accounts payable",
    "Acoustic angiography",
    "Acoustic quieting",
    "Acrylonitrile butadiene styrene",
    "Action learning",
    "Action type approach",
    "Activating function",
    "Active Yaw Control",
    "Active redundancy",
    "Activity-based proteomics",
    "Acupuncture",
    "Adaptive coil combination",
    "Adaptive deep brain stimulation",
    "Adhesion molecule (immunoglobulin-like)",
    "Adiabatic MRI Pulses",
    "Advanced technology engine",
    "Advertising mail",
    "Aerial photographic and satellite image interpretation",
    "AgMES",
    "Agricultural Science and Technology Indicators",
    "Agroforestry",
    "AgustaWestland Project Zero",
    "Aileron",
    "Air Infiltration and Ventilation Centre",
    "Air conditioning",
    "Air handler",
    "Airborne wind turbine",
    "Airbus MAVERIC",
    "Aircraft flight dynamics",
    "Airflow window",
    "Airing",
    "Airplane scatter",
    "Aldehyde-stabilized cryopreservation",
    "Algae DNA barcoding",
    "Alkali-metal thermal to electric converter",
    "All-electric range",
    "Allocative efficiency",
    "Alternator (automotive)",
    "Aluminum polymer composite",
    "Alzheimer's Disease Neuroimaging Initiative",
    "Alzheimer's disease among East Asians",
    "Ambulatory glucose profile",
    "Ammeter",
    "Amplicon sequence variant",
    "Amplified magnetic resonance imaging",
    "Amplitude integrated electroencephalography",
    "Amyloid-related imaging abnormalities",
    "Analysis of Functional NeuroImages",
    "Analyte-specific reagent",
    "Anatomical model",
    "Ancient pathogen genomics",
    "Angiocardiography",
    "Angiography",
    "Angioplasty",
    "Angioscopy",
    "Angiosome",
    "Angle-resolved low-coherence interferometry",
    "AnimalBase",
    "Anomaly scan",
    "Anoscopy",
    "Antegrade pyelography",
    "Anthropometry",
    "Anthropometry of the upper arm",
    "Anti-scatter grid",
    "Aortography",
    "Apple Account",
    "Apple Home",
    "Apple Pay",
    "Apple ecosystem",
    "Arterial input function",
    "Arterial resistivity index",
    "Arthroscopy",
    "Asset/liability modeling",
    "Assured clear distance ahead",
    "Atherectomy",
    "Atmospheric temperature",
    "Atrial tachycardia",
    "Auditing (Scientology)",
    "Authentic assessment",
    "Auto-defrost",
    "Automated whole-breast ultrasound",
    "Automatic balancing valve",
    "Automatic tachycardia",
    "Automotive air conditioning",
    "Autothrottle",
    # --- B (83) ---
    "BAM (file format)",
    "BASFI",
    "BB-Cl-Amidine",
    "BGB-16673",
    "BLOSUM",
    "BLUE protocol",
    "BRAIN Initiative",
    "BRB-seq",
    "BT Smart Hub",
    "BacMap",
    "Bacterial phylodynamics",
    "Bacteriocin IId",
    "Bacteriophage scaffolding proteins",
    "Ballie",
    "Balloon-occluded retrograde transvenous obliteration",
    "Bank Verification Number",
    "Banknote",
    "Banner",
    "Barcode of Life Data System",
    "Bare-metal stent",
    "Barium titanate",
    "Barnes & Noble Nook 1st Edition",
    "Baroreceptor",
    "Bartercard",
    "Battery electric bus",
    "Battery electric vehicle",
    "Battery swapping",
    "Battle damage assessment",
    "Bebionic",
    "Bellows (photography)",
    "Bend radius",
    "Beta Technologies",
    "Bette Korber",
    "Billboard",
    "Billing and settlement plan",
    "Bimetallic strip",
    "Binding domain",
    "Binding potential",
    "BindingDB",
    "Bio-MEMS",
    "Bio-layer interferometry",
    "Biobank",
    "Biohybrid solar cell",
    "Bioinformatics discovery of non-coding RNAs",
    "Biological specimen",
    "Biomarker discovery",
    "Biomonitoring",
    "Bionics",
    "Biophone",
    "Bioprinting drug delivery",
    "Biopsy",
    "Biorepository",
    "Bioresorbable stent",
    "Bismuth telluride",
    "Bitcoin",
    "Blood alcohol content",
    "Body identification",
    "Boiler",
    "Bombinin",
    "Bone age",
    "Bone marrow examination",
    "Bone scintigraphy",
    "Bone segment navigation",
    "Bookkeeping",
    "Boston Diagnostic Aphasia Examination",
    "Bottom-up proteomics",
    "Botulinum toxin",
    "Bradford protein assay",
    "Brain atlas",
    "Brain implant",
    "Brain mapping",
    "BrainGate",
    "Brand protection",
    "Breast MRI",
    "Breast cancer screening",
    "Breast imaging",
    "Breast measurement",
    "Breast ultrasound",
    "Brequinar",
    "Bronchial brushing",
    "Building airtightness",
    "Bus HVAC System",
    "Bus rapid transit",
    # --- C (226) ---
    "CDPPB",
    "CERN Hadron Linacs",
    "CHIMERE chemistry-transport model",
    "CITE-Seq",
    "CLARITY",
    "COVID-19 pandemic cases",
    "COVID-19 rapid antigen test",
    "CRISPR",
    "CRISPR activation",
    "CRISPR gene editing",
    "CRISPR interference",
    "CT pulmonary angiogram",
    "CUBIC",
    "CUT&RUN sequencing",
    "CZTS",
    "Cadmium telluride photovoltaics",
    "Calcein",
    "Calibration gas",
    "California mastitis test",
    "Call centre",
    "Calypso Ichthyological Database",
    "Cambridge Structural Database",
    "Camouflage",
    "Cancer genome sequencing",
    "Cancer registry",
    "Candle warmer",
    "Canine cancer detection",
    "Cannabis drug testing",
    "Cannon-Caliber Electromagnetic Gun launcher",
    "Capsule endoscopy",
    "Car elevator",
    "Carbon accounting",
    "Carbon copy",
    "Carbon dioxide angiography",
    "Carbon nanotube nanomotor",
    "CardMobili",
    "Cardiac PET",
    "Cardiac conduction system",
    "Cardiac electrophysiology",
    "Cardiac imaging",
    "Cardiac magnetic resonance imaging",
    "Cardiac magnetic resonance imaging perfusion",
    "Cardiac resynchronization therapy",
    "Cardiac rhythmicity",
    "Cardiac ventriculography",
    "Carius halogen method",
    "Carotid ultrasonography",
    "Carwow",
    "Cas12a",
    "Cas13",
    "Cas9",
    "Cash concentration",
    "Cash flow",
    "Cash flow forecasting",
    "Cash flow statement",
    "Cash method of accounting",
    "Cash-flow diagram",
    "Catheterization laboratory",
    "Cell-free protein synthesis",
    "Cells Alive System",
    "Ceperognastat",
    "Cephalometry",
    "Cerebral angiography",
    "Cerebrospinal fluid flow MRI",
    "Certified check",
    "Cervical Cancer Gene Database",
    "Cervicography",
    "ChIL-sequencing",
    "ChIP-exo",
    "ChIP-on-chip",
    "Chainlink (blockchain)",
    "Chaotic scattering",
    "Character amnesia",
    "Charge card",
    "ChargePlace Scotland",
    "Chassis",
    "Chatbot psychosis",
    "Chatfishing",
    "Cheating",
    "Check washing",
    "Chemical space",
    "Chemogenetics",
    "Chemogenomics",
    "Chemotaxonomy",
    "Chest radiograph",
    "ChiRP-Seq",
    "Chiller",
    "Chiller boiler system",
    "Chinese character strokes",
    "Chinese herbology",
    "Chinese number gestures",
    "Cholangiography",
    "Cholecystography",
    "Cholescintigraphy",
    "Chou–Fasman method",
    "Chromatin immunoprecipitation",
    "Cigarette machine",
    "Cinematographer",
    "Circuit topology",
    "Circular consensus sequencing",
    "Cladding (fiber optics)",
    "Cleveland open-cup method",
    "Click farm",
    "Click fraud",
    "Click-through rate",
    "Clinical cardiac electrophysiology",
    "Clinical coder",
    "Clinical metagenomic sequencing",
    "Clinical prediction rule",
    "Clinical study design",
    "Clinical trial",
    "Clinicogenomics",
    "Clock position",
    "Coal burner",
    "Cobb angle",
    "Coded anti-piracy",
    "Coefficient of performance",
    "Coffee vending machine",
    "Cogeneration",
    "Cognitive test",
    "Coin",
    "Cold chain",
    "Colloidal gold protein assay",
    "Colonoscopy",
    "Colored Coins",
    "Colossal magnetoresistance",
    "Colposcopy",
    "Combat Fitness Test",
    "Combined storage tanks",
    "Combustion chemical vapor deposition",
    "Comet assay",
    "Commemorative plaque",
    "Commodity currency",
    "Communication with submarines",
    "Community Exchange System",
    "Comparative genomics",
    "Compartmental models (epidemiology)",
    "Competitive examination",
    "Complex system approach to peace and armed conflict",
    "Comprehensive Antibiotic Resistance Database",
    "Comprehensive Health Assessment Program",
    "Compression of genomic sequencing data",
    "Computed tomography dose index",
    "Computed tomography enterography",
    "Computed tomography laser mammography",
    "Computed tomography of the chest",
    "Computed tomography of the head",
    "Computed tomography of the thyroid",
    "Computer-induced medical problems",
    "Condenser (heat transfer)",
    "Cone beam computed tomography",
    "Cone-beam spiral computed tomography",
    "Connectogram",
    "Connectome",
    "Consensus sequence",
    "Conservation and restoration of rail vehicles",
    "Conservation genomics",
    "Consolidated financial statement",
    "Conspiracy",
    "Consumption map",
    "Contact order",
    "Contamination Indicator Decontamination Assurance System",
    "Continued process verification",
    "Contrast-enhanced ultrasound",
    "Control knob",
    "Coolant",
    "Cooling",
    "Cooling capacity",
    "Cooling center",
    "Cooling load",
    "Cophenetic",
    "Copper indium gallium selenide solar cell",
    "Copy number variation",
    "Copy trading",
    "Copyright infringement",
    "Corneal topography",
    "Coronary CT angiography",
    "Coronary CT calcium scan",
    "Coronary stent",
    "Coronavirus breathalyzer",
    "Corporate social media",
    "Corpus linguistics",
    "Corrective Optics Space Telescope Axial Replacement",
    "Correlation coefficient",
    "Correlation function (statistical mechanics)",
    "Correspondence analysis",
    "Cortana (virtual assistant)",
    "Cost",
    "Cost allocation",
    "Cost auditing",
    "Cost price",
    "Counterweight",
    "Counting",
    "Cpf1",
    "Cranial ultrasound",
    "Cranio-corpography",
    "Craniometry",
    "Creative Commons license",
    "Credit card",
    "Credit card fraud",
    "Credit card imprinter",
    "Crime concentration",
    "Criticisms of intellectual property",
    "Crop circle",
    "CropSyst",
    "Cross ventilation",
    "Cross-Border Interbank Payment System",
    "Cross-border E-commerce",
    "Crossbuck",
    "Crowdfunding",
    "Crowdshipping",
    "Crp domain",
    "Cryoablation",
    "Cryocooler",
    "Cryogenic current comparator",
    "Cryopreservation",
    "Cryptocurrency and crime",
    "Cryptocurrency exchange",
    "Crystalline silicon",
    "Culturomics (microbiology)",
    "Currency",
    "Currency money",
    "Currency symbol",
    "Current density imaging",
    "CyberCell",
    "Cystourethrography",
    # --- D (121) ---
    "D75-4590",
    "DAPI",
    "DECIPHER",
    "DIMPL",
    "DISC assessment",
    "DIY Kindle Scanner",
    "DNA and RNA codon tables",
    "DNA bank",
    "DNA barcoding",
    "DNA database",
    "DNA glycosylase",
    "DNA nanoball sequencing",
    "DNA profiling",
    "DNA sequencer",
    "DNA sequencing",
    "DNA teleportation",
    "DNA-templated organic synthesis",
    "DNSChanger",
    "DOTA (chelator)",
    "DTED",
    "DVD Decrypter",
    "DVR-MS",
    "Daisy Intelligence",
    "Damper (flow)",
    "Data Mining and Knowledge Discovery",
    "Data broker",
    "Data cap",
    "Data collaboratives",
    "Data defined storage",
    "Data literacy",
    "Data philanthropy",
    "Database of protein conformational diversity",
    "DbSNP",
    "De novo peptide sequencing",
    "Debits and credits",
    "Deceleron",
    "DecisionDx-UM",
    "Dedicated outdoor air system",
    "Deep brain stimulation",
    "Defibrillation",
    "Defibrillation threshold",
    "Defrosting (refrigeration)",
    "Dehumidifier",
    "Delta-V charging",
    "Demand controlled ventilation",
    "Demand draft",
    "Denitrifying bacteria",
    "Dense breast tissue",
    "Depolarizer (optics)",
    "Depth kymography",
    "Detective quantum efficiency",
    "Diagnostic medical sonography",
    "Dibutoxymethane",
    "Dichroism",
    "Didget",
    "Dielectric withstand test",
    "Diesel generator",
    "Diesel-electric powertrain",
    "Diesel–electric powertrain",
    "Differential dynamic microscopy",
    "Differential staining",
    "Diffuse correlation spectrometry",
    "Diffuse optical imaging",
    "Diffuse optical mammography",
    "Diffuser (thermodynamics)",
    "Digimask",
    "Digital 3D",
    "Digital Building Logbook",
    "Digital anthropology",
    "Digital billboard",
    "Digital calendar",
    "Digital card",
    "Digital collections selection criteria",
    "Digital commons",
    "Digital curation",
    "Digital distribution",
    "Digital empathy",
    "Digital materialization",
    "Digital on-screen graphic",
    "Digital photo frame",
    "Digital puppetry",
    "Digital scent technology",
    "Digital technologies and environmental sustainability",
    "Digital ticket",
    "Dille-Koppanyi reagent",
    "Dille–Koppanyi reagent",
    "Dilution refrigerator",
    "Direct bank",
    "Direct laser interference patterning",
    "Direct traffic control",
    "Direct-broadcast satellite",
    "Direct-to-garment printing",
    "Discharge printing",
    "Discovery shopping",
    "Disease burden",
    "Disease gene identification",
    "Disease registry",
    "Disease surveillance",
    "Disease vector",
    "Dishwasher",
    "Distributed power",
    "Docphin",
    "Doppler fetal monitor",
    "Doppler tomography",
    "Doppler ultrasonography",
    "Double-balloon enteroscopy",
    "Double-deck elevator",
    "Double-sided tape",
    "Double-strand breaks",
    "Downhole heat exchanger",
    "Driving test",
    "Drug-eluting stent",
    "Dual-energy X-ray absorptiometry",
    "Dual-mode bus",
    "Duct (flow)",
    "Duct leakage testing",
    "Ductless air conditioner",
    "Dump charging",
    "Duplex sequencing",
    "Dynamic angiothermography",
    "Dynamo (automotive)",
    # --- E (173) ---
    "E-accounting",
    "E-agriculture",
    "E-card",
    "E-meter",
    "E-patient",
    "E-scooter-sharing system",
    "ECall",
    "EDIM technology",
    "EFDC Explorer",
    "EHealth",
    "ENDURANCE",
    "EOS (medical imaging)",
    "ESI Group",
    "ESign (India)",
    "EU Cloud Code of Conduct",
    "EVolution",
    "Early Warning and Response System",
    "Earth inductor compass",
    "Earth-centered, Earth-fixed coordinate system",
    "Earth3D",
    "Earthquake Early Warning (Japan)",
    "EasyHDR",
    "Echinobase",
    "Echo Seeker",
    "Echo Voyager",
    "Echocardiography",
    "Echoencephalography",
    "Eclipse ERP",
    "Ectopic beat",
    "Edman degradation",
    "Educational measurement",
    "Eelume",
    "Effective (database)",
    "Effective dose (radiation)",
    "Efficiency map",
    "Efficient coding hypothesis",
    "Ehrlich's reagent",
    "Einstein (US-CERT program)",
    "Einstein refrigerator",
    "Elastography",
    "Election security",
    "Electric aircraft",
    "Electric boat",
    "Electric bus",
    "Electric car",
    "Electric car charging methods",
    "Electric heavy equipment",
    "Electric helicopter",
    "Electric motor",
    "Electric motor test stand",
    "Electric motorcycles and scooters",
    "Electric power conversion",
    "Electric road",
    "Electric tractor",
    "Electric truck",
    "Electric unicycle",
    "Electric vehicle",
    "Electric vehicle conversion",
    "Electrical brain stimulation",
    "Electrical load",
    "Electrical outlet tester",
    "Electrical resistivity measurement of concrete",
    "Electrical resistivity tomography",
    "Electrically conductive adhesive",
    "Electricity price forecasting",
    "Electro stimulation",
    "Electro-galvanic oxygen sensor",
    "Electroacupuncture",
    "Electroanatomic mapping",
    "Electrocar",
    "Electrochemical aptamer-based biosensors",
    "Electrochemical gas sensor",
    "Electrochemical skin conductance",
    "Electrocochleography",
    "Electroconvulsive therapy",
    "Electrodynamic wheel",
    "Electroencephalography",
    "Electroencephalography functional magnetic resonance imaging",
    "Electrogram",
    "Electrohydraulic servo valve",
    "Electromagnetic ion thruster",
    "Electromagnetic log",
    "Electromagnetic motor",
    "Electromagnetic navigation bronchoscopy",
    "Electromagnetic testing",
    "Electromyoneurography",
    "Electron beam computed tomography",
    "Electron holography",
    "Electrooculography",
    "Electropherogram",
    "Electrophysiology",
    "Electrophysiology study",
    "Electrostatic fieldmeter",
    "Electrostatic spray-assisted vapour deposition",
    "Electrotonic potential",
    "Elevator",
    "Elicitin",
    "Eliica",
    "Embedded atom model",
    "Embodied cognition",
    "Embolectomy",
    "Embolization",
    "Emergency brake (train)",
    "Emergency power system",
    "Emergency radio",
    "Emergency ultrasound",
    "Emergency vehicle lighting",
    "Emerging infectious disease",
    "Emissions reduction currency systems",
    "Endoanal ultrasound",
    "Endoclip",
    "Endoluminal capsule monitoring",
    "Endomicroscopy",
    "Endorectal coil magnetic resonance imaging",
    "Endoscopic carpal tunnel release",
    "Endoscopic ear surgery",
    "Endoscopic nasopharyngectomy",
    "Endoscopic retrograde cholangiopancreatography",
    "Endoscopic stenting",
    "Endoscopic thoracic sympathectomy",
    "Endoscopic ultrasound",
    "Endoscopy",
    "Endoscopy unit",
    "Endovascular coiling",
    "Energy signature",
    "Engineering biology",
    "Enhanced heat transfer",
    "Enteroscopy",
    "Environmental accounting",
    "Environmental impact of bitcoin",
    "Environmental monitoring",
    "Environmental testing",
    "Enzyme-activated MR contrast agents",
    "Epidemic models on lattices",
    "Epidemiology of autism",
    "Epidemiology of cancer",
    "Epidemiology of metabolic syndrome",
    "Epigenetic therapy",
    "Epigenome editing",
    "Epiphany Eyewear",
    "Epitranscriptomic sequencing",
    "Epson Micro Flying Robot",
    "Equity crowdfunding",
    "Ergonomic glove",
    "Ergonomic hazard",
    "Esophagogastroduodenoscopy",
    "Essential gene",
    "Estimated date of delivery",
    "Eukaryotic Pathogen Database",
    "Eurobalise",
    "European Nucleotide Archive",
    "European seasonal energy efficiency ratio",
    "Eutelsat Quantum",
    "Evaporative cooler",
    "Evaporative cooling chambers",
    "Eve Air Mobility",
    "Event generator",
    "Evoked potential",
    "Evolutionary Classification of Protein Domains",
    "Evolutionary epidemiology",
    "Exam",
    "Exchange Online Protection",
    "Exchange rate",
    "Exhaust gas analyzer",
    "Exhaust gas temperature gauge",
    "Expansion joint",
    "Expansion tank",
    "Experiment",
    "Explosive gas leak detector",
    "Expression Atlas",
    "Extreme value theory",
    "Extromatica Network Monitor",
    "EyeWriter",
    # --- F (102) ---
    "F-rating",
    "FARME",
    "FME connector",
    "FMRI lie detection",
    "FMRIB Software Library",
    "FORECAST (model)",
    "FaceApp",
    "Factor regression model",
    "Factorization of polynomials",
    "Factorization of polynomials over finite fields",
    "Fail-safe",
    "Failure analysis",
    "FarmBot",
    "Fastran",
    "Fat removal procedures",
    "Fat suppression",
    "Fatigue testing",
    "Fecal immunochemical test",
    "Fetoscopy",
    "Fiat Panda Hydrogen",
    "Fiberscope",
    "Fibrobacter succinogenes",
    "Field-induced polymer electroluminescent technology",
    "Field-theoretic simulation",
    "Filecoin",
    "Filling carousel",
    "Filtered Rayleigh scattering",
    "FinanceWorks",
    "Financial accounting",
    "Financial modeling",
    "Financial transaction",
    "Fine structure genetics",
    "Fine-tuning (physics)",
    "Fingerprint",
    "Fire pump",
    "Fire sprinkler system",
    "Fire suppression system",
    "Fire test",
    "First Warning",
    "Fish counter",
    "FishBase",
    "Fisheye lens",
    "Fitbit",
    "Fitness (Apple)",
    "Five-bar linkage",
    "Fixes that fail",
    "Flame supervision device",
    "FlashCopy",
    "Flexibility (engineering)",
    "Flexible-fuel vehicle",
    "Flight recorder",
    "Floristic Quality Assessment",
    "Flow control (fluid)",
    "Flow measurement",
    "Flow tracer",
    "Flow-through test",
    "Flue",
    "Flue-gas condensation",
    "Fluence response",
    "Fluorescein diacetate hydrolysis",
    "Fluorescence image-guided surgery",
    "Fluorescent glucose biosensor",
    "Fluorescent penetrant inspection",
    "Fluoroscopy",
    "Flux balance analysis",
    "Flux pumping",
    "Fluxomics",
    "Focused assessment with sonography for trauma",
    "Focused ultrasound",
    "Focused ultrasound-mediated diagnostics",
    "Fogging phenomenon",
    "Foghorn",
    "FoldX",
    "Foodomics",
    "Forced Rayleigh scattering",
    "Forced-air",
    "Forward scatter",
    "Fracture sonography",
    "Free cooling",
    "Frequency changer",
    "Frozen tissue array",
    "Frozen zoo",
    "Fuel cell auxiliary power unit",
    "Fuel cell bus",
    "Fuel cell vehicle",
    "Full motion racing simulator",
    "Full-body CT scan",
    "Fume hood",
    "Functional disconnection",
    "Functional element SNPs database",
    "Functional endoscopic sinus surgery",
    "Functional magnetic resonance imaging",
    "Functional magnetic resonance spectroscopy of the brain",
    "Functional molecular infection epidemiology",
    "Functional near-infrared spectroscopy",
    "Functional neuroimaging",
    "Functional ultrasound imaging",
    "Furnace (central heating)",
    "Fuse (electrical)",
    "Fusion Camera System",
    "Fusion Drive",
    "Future Combat Air System",
    # --- G (121) ---
    "G protein-coupled receptors database",
    "GB stroke-based order",
    "GEN2PHEN",
    "GENIUS Act",
    "GOR method",
    "GPS buoy",
    "GPS drawing",
    "GPS-based game",
    "GPU mining",
    "GRCh38",
    "GSK-971086",
    "GUIDE-Seq",
    "Gabor limit",
    "Gadolinium disilicide",
    "Gait analysis",
    "Galaxy AI",
    "Galaxy Avatar",
    "Gallery Project",
    "Gallium phosphate",
    "Gallium(II) selenide",
    "Game accessibility",
    "Gamma camera",
    "Ganimal",
    "Gantry (medical)",
    "Gantry (transport)",
    "Gap surface plasmon",
    "Garage door opener",
    "Garmin Forerunner",
    "Gas detector",
    "Gas pyelogram",
    "Gaseous ionization detector",
    "Gaussian orbital",
    "GenBank",
    "Gene Disease Database",
    "Gene Ontology Term Enrichment",
    "Gene bank",
    "Gene expression profiling",
    "Gene map",
    "Gene set enrichment analysis",
    "GeneReviews",
    "GeneTalk",
    "GeneXpert MTB/RIF",
    "Genetic editing",
    "Genetic marker",
    "Genetic testing",
    "Genetically encoded voltage indicator",
    "Geneva score",
    "Gene–environment interaction",
    "Genome",
    "Genome Reference Consortium",
    "Genome Taxonomy Database",
    "Genome diversity and karyotype evolution of mammals",
    "Genome editing",
    "Genome profiling",
    "Genome-based peptide fingerprint scanning",
    "Genome-wide CRISPR-Cas9 knockout screens",
    "Genomic and medical data",
    "Genomic library",
    "Genomics",
    "Genotype",
    "Genotyping by sequencing",
    "GeoTime",
    "Geodemographic segmentation",
    "Geological Strength Index",
    "Geotagged photograph",
    "Germline editing",
    "Germplasm",
    "Ghosting (medical imaging)",
    "Gleason grading system",
    "Global Explorer ROV",
    "Global Payments",
    "Global Public Health Intelligence Network",
    "Global Sea Level Observing System",
    "Global microbial identifier",
    "Global relief model",
    "Glove One",
    "Glow plate",
    "Glucoboy",
    "Glutamate-sensitive fluorescent reporter",
    "Glycol chiller",
    "Glycoproteomics",
    "Glyphs (font editor)",
    "GoTo Meeting",
    "Goal attainment scaling",
    "Gold–aluminium intermetallic",
    "Google AI Mode",
    "Google Account",
    "Google Fonts",
    "Google Home Hub",
    "Google Maps",
    "Google Maps Navigation",
    "Google Messages",
    "Google Nest (smart speakers)",
    "Google Nest Hub",
    "Google Pay (payment method)",
    "Google Pinyin",
    "Google Street View",
    "Google Tasks",
    "Google Traffic",
    "Google Translate",
    "Google Voice Search",
    "Gradient echo",
    "Gradient-index optics",
    "Grammarly",
    "Graphical models for protein structure",
    "Grating-coupled interferometry",
    "Gravatar",
    "GreeNC",
    "Grid-tie inverter",
    "Gridegalutamide",
    "Ground reconnaissance",
    "Ground segment",
    "Ground-coupled heat exchanger",
    "Ground-level power supply",
    "Group II pyridoxal-dependent decarboxylases",
    "Group III pyridoxal-dependent decarboxylases",
    "Guide RNA",
    "GuideGeek",
    "Gynecologic ultrasonography",
    "Gynography",
    "Gyro monorail",
    # --- H (107) ---
    "H2Ceramic cooling",
    "HAWK beacon",
    "HBV hydrology model",
    "HCL Commerce",
    "HDDerase",
    "HDR10+",
    "HIE of One",
    "HIV Drug Resistance Database",
    "HOSxP",
    "HPE OpenCall",
    "HR-pQCT",
    "HRDetect",
    "Hand geometry",
    "Handle",
    "Handwriting",
    "Harmonic",
    "Harmonic balance",
    "Harmonic spectrum",
    "Hashtag activism",
    "Hazard analysis",
    "Hazard ratio",
    "Hazard symbol",
    "Heading indicator",
    "HealthifyMe",
    "Heat exchanger",
    "Heat recovery ventilation",
    "Heat spreader",
    "Heated glass",
    "Heater core",
    "Heating pad",
    "Heating system",
    "Heating, ventilation, and air conditioning",
    "Heidelberg Retinal Tomography",
    "Height above ground level",
    "Helicopter noise reduction",
    "Heller's test",
    "Hello English",
    "Helmet camera",
    "Hemisphere GNSS",
    "HemoSpat",
    "Hepatic artery embolization",
    "Hepatogram",
    "Herbarium vivum",
    "Here One",
    "Heterojunction solar cell",
    "HeuristicLab",
    "Hexaminolevulinate",
    "Hi-C (genomic analysis technique)",
    "High performance positioning system",
    "High-definition fiber tracking",
    "High-performance Integrated Virtual Environment",
    "High-resolution computed tomography",
    "Highway location marker",
    "Hill-holder",
    "Hirtz compass",
    "Hockey tape",
    "Hodoscope",
    "Hoist (device)",
    "Holmes and Rahe stress scale",
    "HoloLens 2",
    "Hologram bracelet",
    "Hologram trademark",
    "Holographic direct sound printing",
    "Holographic grating",
    "Holter monitor",
    "Home audio",
    "Home ultrasound",
    "Homology modeling",
    "Honeyd",
    "Hook echo",
    "Hospi",
    "Hospital Records Database",
    "Hugo Wolf F/A-18C simulator",
    "Human Metabolome Database",
    "Human billboard",
    "Human body weight",
    "Human height",
    "Human information interaction",
    "Human microbiome",
    "Human-readable medium",
    "Humanitarian drone",
    "Human–electric hybrid vehicle",
    "Humidistat",
    "Hybrid Air Vehicles Airlander 10",
    "Hybrid electric aircraft",
    "Hybrid electric bus",
    "Hybrid electric truck",
    "Hybrid electric vehicle",
    "Hybrid electromagnetic suspension",
    "Hybrid ferry",
    "Hybrid genome assembly",
    "Hybrid heat",
    "Hybrid solar cell",
    "Hybrid taxi",
    "Hybrid vehicle",
    "Hydraulic cylinder",
    "HydroGeoSphere",
    "Hydrogen leak testing",
    "Hydrologic Evaluation of Landfill Performance",
    "Hydrologic unit system (United States)",
    "Hydrological model",
    "Hydrological optimization",
    "Hydronic balancing",
    "Hyperpolarization (physics)",
    "Hyperpolarized gas MRI",
    "Hypoxic air technology for fire prevention",
    "Hysteroscopy",
    # --- I (113) ---
    "I'm Watch",
    "I-III-VI semiconductors",
    "I-Pass",
    "I-TASSER",
    "IBM 1712",
    "IBM 2821 Control Unit",
    "IBM 4683",
    "IBM Intelligent Printer Data Stream",
    "IBM LPFK",
    "IBM RFID Information Center",
    "IBM SAN Volume Controller",
    "IBM TouchMobile",
    "IBM WebFountain",
    "ID Sniper rifle",
    "IEC 63119",
    "IEEE Cloud Computing",
    "IEEE Smart Grid",
    "ILO Classification",
    "IMARK",
    "INtime",
    "ISET Test",
    "IScanner",
    "Ice storage air conditioning",
    "Ideogram (text-to-image model)",
    "Ignition magneto",
    "Ignition timing",
    "Illumination (image)",
    "Image",
    "Image-guided radiation therapy",
    "Imageability",
    "Imagen (text-to-image model)",
    "Imagery intelligence",
    "Imaging phantom",
    "Immersion exhibit",
    "Immunological Genome Project",
    "Impedance analogy",
    "Impedance–pH monitoring",
    "Implantable cardioverter-defibrillator",
    "Implantable loop recorder",
    "Index card",
    "Indicator diagram",
    "Indium tin oxide",
    "Individual physical proficiency test",
    "Indocyanine green",
    "Indocyanine green angiography",
    "Induction heating",
    "Inductrack",
    "Industrial radiography",
    "Inertia wheel pendulum",
    "Infection",
    "Infection prevention and control",
    "Inferior vena cava filter",
    "Inferring horizontal gene transfer",
    "Information continuity",
    "Infrared heater",
    "Inhibitor of DNA-binding protein",
    "Initial coin offering",
    "Inkjet solar cell",
    "Innovation intermediary",
    "Innovative Clean Transit rule",
    "Inpaint",
    "Inspirator",
    "Instability index",
    "Installment sales method",
    "Instant camera",
    "Instrument landing system localizer",
    "Insulated pipe",
    "Insulated shipping container",
    "Insurance fraud",
    "IntFOLD",
    "Integrated Microbial Genomes System",
    "Integrated Ocean Observing System",
    "Intel Display Power Saving Technology",
    "Intel Modular Server System",
    "Intel Teach",
    "Intel Tera-Scale",
    "Intel Turbo Memory",
    "InterSystems Caché",
    "Interferometric visibility",
    "Intermediate band photovoltaics",
    "Intermediate-frequency amplifier",
    "Internal rotary inspection system",
    "International Classification of Childhood Cancer",
    "International Farming Systems Collection",
    "International Phonetic Alphabet",
    "International Society for Computer Aided Surgery",
    "International Subarachnoid Aneurysm Trial",
    "IntervalZero",
    "Interventional magnetic resonance imaging",
    "Interventional neuroradiology",
    "Interventional oncology",
    "Interventional radiology",
    "Intracardiac echocardiogram",
    "Intracortical encephalogram signal analysis",
    "Intraoperative MRI",
    "Intraoperative neurophysiological monitoring",
    "Intraoral camera",
    "Intravascular fluorescence",
    "Intravascular imaging",
    "Intravascular ultrasound",
    "Intravoxel incoherent motion",
    "Invasion percolation",
    "Inventory valuation",
    "Inversion recovery",
    "Invoice processing",
    "Iodinated contrast",
    "Ion beam",
    "Ion beam mixing",
    "Iris flower data set",
    "Irissometry",
    "Irreversible electroporation",
    "Ishiguro Storm Surge Computer",
    "Issuing bank",
    # --- J (6) ---
    "Joint replacement registry",
    "Joule heating",
    "Joule thief",
    "Jugaad",
    "Jukebox",
    "Jump camera",
    # --- K (16) ---
    "Kevin Ashton",
    "Kidney dialysis machine",
    "Kinematic chain",
    "Kinematic pair",
    "Kinematic synthesis",
    "Kinetic energy recovery system",
    "Kinetic theory of gases",
    "Kinome",
    "Kitchen stove",
    "Knee replacement",
    "Knitting machine",
    "Knowledge Navigator",
    "Knowledge crystal",
    "Knowledge economy",
    "Knowledge value chain",
    "Kymography",
    # --- L (92) ---
    "L-733,060",
    "LEAPER gene editing",
    "LI-RADS",
    "LIGO",
    "LINAC 1",
    "LINAC 2",
    "LINAC 3",
    "LING VR",
    "Label",
    "Laminar flow",
    "Lamp (fixture)",
    "Landed cost",
    "Landline",
    "Langley extrapolation",
    "Language resource",
    "Lap desk",
    "Laparoscopy",
    "Lapping",
    "Laptop charging trolley",
    "Large Hadron Collider",
    "LarvalBase",
    "Laser cooling",
    "Laser hair removal",
    "Laser lithotripsy",
    "Laser medicine",
    "Laser surgery",
    "Laser therapy",
    "Lateral flow test",
    "Lathe",
    "Lattice density functional theory",
    "Laundry ball",
    "Lead poisoning",
    "Lead scandium tantalate",
    "Lead-DBS",
    "Leather",
    "Lebanese loop",
    "Lederbergvirus P22",
    "Left atrial appendage occlusion",
    "Legibility",
    "Leiden Open Variation Database",
    "Lemon battery",
    "Lesion network mapping",
    "Leucine-rich repeat receptor like protein kinase",
    "Lever",
    "Lexile",
    "LibraryThing",
    "Life Cycle Climate Performance",
    "Life Image",
    "LifeBEAM",
    "Liftware",
    "Light booth",
    "Light characteristic",
    "Light therapy",
    "Lilium Jet",
    "Linac4",
    "Linear actuator",
    "Linear motor",
    "Linguistic categories",
    "Linkage (mechanical)",
    "Lipoprotein Insulin Resistance Index",
    "Liquid Fidelity",
    "Liquid biopsy",
    "Liquid breathing",
    "Liquid nitrogen",
    "Liquid-mirror space telescope",
    "Liquid–liquid phase separation sequence-based predictors",
    "Lithotripsy",
    "Live action",
    "Live blood analysis",
    "Live broadcast",
    "Livestreamed news",
    "Lobster-eye optics",
    "Local coordinates",
    "Locomotive",
    "Logo",
    "Loitering munition",
    "Long division",
    "Long-range restriction mapping",
    "Long-term video-EEG monitoring",
    "LongPen",
    "Loop modeling",
    "Lorentz force",
    "Lorenzo (electronic health record)",
    "Low-field magnetic resonance imaging",
    "Low-intensity pulsed ultrasound",
    "Lower gastrointestinal series",
    "Luciferase",
    "Luminescence",
    "Luminescent solar concentrator",
    "Lunar Laser Ranging experiments",
    "Lung cancer screening",
    "Lymphogram",
    # --- M (259) ---
    "M2-PK Test",
    "MAP test",
    "MARCbot",
    "MELD-Plus",
    "MG-RAST",
    "MIMIC (immunology)",
    "MODELLER",
    "MOPITT",
    "MRI contrast agent",
    "MUBII-TB-DB",
    "Machete order",
    "Magazine",
    "Maglev",
    "Magnet therapy",
    "Magnetic field oscillating amplified thruster",
    "Magnetic induction tomography",
    "Magnetic levitation",
    "Magnetic nano",
    "Magnetic nanoring",
    "Magnetic resonance",
    "Magnetic resonance angiography",
    "Magnetic resonance cholangiopancreatography",
    "Magnetic resonance elastography",
    "Magnetic resonance imaging of the brain",
    "Magnetic resonance microscopy",
    "Magnetic resonance myelography",
    "Magnetic resonance neurography",
    "Magnetic resonance spectroscopic imaging",
    "Magnetic resonance velocimetry",
    "Magnetic sail",
    "Magnetic-plasmonic bifunctional nanoparticles",
    "Magnetoencephalography",
    "Mail",
    "Mainline DHT",
    "MammaPrint",
    "Mammography",
    "Managed lane",
    "Mandatory sign",
    "Manifold alignment",
    "Manual fare collection",
    "Map communication model",
    "Map layout",
    "Map symbol",
    "MapDotNet",
    "Marijuana vending machine",
    "Marine optical buoy",
    "Marine radar",
    "Marine weather forecasting",
    "Maritime broadcast communications net",
    "MarkV-A1",
    "Marker beacon",
    "Markov partition",
    "Mars rover",
    "Marvin (robot)",
    "MasSpec Pen",
    "Mask",
    "Masking threshold",
    "Mass comparison",
    "Mass driver",
    "Mass flow controller",
    "Mass flow sensor",
    "Mass transfer",
    "Mast radiator",
    "Masterpass",
    "Matching (statistics)",
    "Material handling",
    "Materialise Mimics",
    "Materially engineered artificial pollinators",
    "Materiomics",
    "Mathematical Reviews",
    "Mathematical modelling of infectious diseases",
    "Matrix (numismatics)",
    "Maximum entropy thermodynamics",
    "Maximum life disparity",
    "Maya AUV India",
    "McDonnell Douglas X-36",
    "MeVisLab",
    "Mechanical calculator",
    "Mechanical device test stands",
    "Mechanical probe station",
    "Mechanical room",
    "Mechanical testing",
    "Mechanical ventilation",
    "Mechanography",
    "Mediastinoscopy",
    "Mediation (statistics)",
    "Medical Subject Headings",
    "Medical alarm",
    "Medical algorithm",
    "Medical guideline",
    "Medical identification tag",
    "Medical photography",
    "Medical radiography",
    "Medical record",
    "Medical test",
    "Medical transcription",
    "Medication Administration Record",
    "Medication package insert",
    "Medicinal chemistry",
    "Medium of exchange",
    "Medullary sponge kidney",
    "Megaphone",
    "Meibography",
    "Meissner effect",
    "Melamine resin",
    "Melatonin",
    "Membrane protein",
    "Membrane technology",
    "Meniscal cartilage replacement therapy",
    "Menksoft Mongolian IME",
    "Mental model",
    "Merchant account",
    "Merchant's mark",
    "Merck molecular force field",
    "Mercury cadmium telluride",
    "Mercury-atlas 9",
    "Mesomark assay",
    "Mesotherapy",
    "Messages (Apple)",
    "Metabarcoding",
    "Metabolic network modelling",
    "Metabolite Set Enrichment Analysis",
    "Metabolome",
    "Metabolomic Pathway Analysis",
    "Metabolomics",
    "Metabonomics",
    "Metacognition",
    "Metagenomics",
    "Metal casting simulation",
    "Metal detector",
    "Metal oxide adhesion",
    "Metal rubber",
    "Metal-induced gap states",
    "Metallurgical assay",
    "Metalorganic vapour-phase epitaxy",
    "Metascape",
    "Metatranscriptomics",
    "Metering pulse",
    "Metformin",
    "MethBase",
    "Methanol fuel",
    "Methyl cellulose",
    "Methylated DNA immunoprecipitation",
    "Methylation specific oligonucleotide microarray",
    "Metrebus Card",
    "MetroHero",
    "Mi-Pay",
    "Michelangelo Hand",
    "Micro heat exchanger",
    "Micro-Imaging Dust Analysis System",
    "Micro-spatially offset Raman spectroscopy",
    "MicroDicom",
    "MicroRNA and microRNA target database",
    "MicroRNA biosensors",
    "MicroRNA sequencing",
    "MicroStation",
    "Microangiography",
    "Microbial DNA barcoding",
    "Microbial art",
    "Microbial electrosynthesis",
    "Microbial fuel cell",
    "Microbial pathogenesis",
    "Microbial phylogenetics",
    "Microbolometer",
    "Microgravity bioprinting",
    "Microgrid",
    "Micropipette",
    "Microplastics",
    "Microscopy",
    "Microsoft Access",
    "Microsoft Baseline Security Analyzer",
    "Microsoft BizTalk Server",
    "Microsoft Cordless Phone System",
    "Microsoft Device Emulator",
    "Microsoft Drive Optimizer",
    "Microsoft Excel",
    "Microsoft Expression Encoder",
    "Microsoft Lumia",
    "Microsoft Messaging",
    "Microsoft Musical Instruments",
    "Microsoft Office Accounting",
    "Microsoft Site Server",
    "Microsoft Software Updater",
    "Microsoft Surface Hub",
    "Microsoft V-Chat",
    "Microsoft Voice Command",
    "Microsoft account",
    "Microsoft adCenter Analytics",
    "Microvasculature remodeling",
    "Microwave ablation",
    "Microwave chemistry",
    "Microwave chemistry sensor",
    "Microwave oven",
    "Microwave volumetric heating",
    "Millimeter cloud radar",
    "Mineral processing",
    "Mini-split air conditioner",
    "Miniature UHF connector",
    "Minibar",
    "Minimally invasive procedure",
    "Misinformation",
    "Mitra personal mover",
    "Mitsuba (software)",
    "Mixed-mode ventilation",
    "MobiDB",
    "Mobile shelving",
    "Mobility scooter",
    "Model for End-Stage Liver Disease",
    "Moiré pattern",
    "Moisture removal efficiency",
    "Molecular Inversion Probe",
    "Molecular diagnostics",
    "Molecular marker",
    "Molecular models of DNA",
    "Molecular xenomonitoring",
    "Molten salt reactor",
    "Momentum exchange tether",
    "Mondo spider",
    "Money laundering",
    "Monocrystalline silicon",
    "Monosodium glutamate",
    "Morphine",
    "Morris–Lecar model",
    "Mortality Medical Data System",
    "Mosquito control",
    "Motorola connector",
    "Mouse Models of Human Cancer database",
    "Moving field inductive power transfer",
    "Moving magnet actuator",
    "Moving-bed heat exchanger",
    "Multi Media Interface",
    "Multi-Choice TV",
    "Multi-Evaporator System",
    "Multi-fuel stove",
    "Multi-junction solar cell",
    "Multiangle light scattering",
    "Multichannel Television Sound",
    "Multiplane camera",
    "Multiple choice",
    "Multiple scattering low coherence interferometry",
    "Multiple sequence alignment",
    "Multiple unit",
    "Multiplex (sensor)",
    "Multiplexed point-of-care testing",
    "Multispectral optoacoustic tomography",
    "Multiway switching",
    "Muography",
    "Muon tomography",
    "Murray loop bridge",
    "Muscle car",
    "Museomics",
    "Musical cryptogram",
    "Mustard Database",
    "Mutational signatures",
    "Muzzle print",
    "Myelinoid",
    "Myelography",
    "Mynx vascular closure device",
    "Myocardial perfusion imaging",
    # --- N (78) ---
    "N-localizer",
    "N200 (neuroscience)",
    "NAPP (database)",
    "NH Predict",
    "NIAID ChemDB",
    "NOAA Weather Radio",
    "NOMe-seq",
    "Nacubactam",
    "Nalmefene",
    "Naltrexone",
    "Nano-tetherball sensor",
    "Nanobatteries",
    "Nanocrystal solar cell",
    "Nanoinverter",
    "Nanoionics",
    "Nanomotor",
    "Nanoneuroscience",
    "Nanoprecipitation",
    "Nanoscale plasmonic motor",
    "Nanoscale secondary ion mass spectrometry",
    "Nanosphere lithography",
    "Nanotomography",
    "Narrow-band imaging",
    "Nasal spray",
    "National Airspace System Voice Switch",
    "National Automated Fingerprint Identification System",
    "National Common Mobility Card",
    "National DNA database",
    "National Elevation Dataset",
    "National Health Information Infrastructure",
    "National Information Infrastructure",
    "National Warning System",
    "Natural gas",
    "Natural pacemaker",
    "Natural refrigerant",
    "Navigation light",
    "Near-infrared vein finder",
    "Neodymium magnet",
    "Neoepitope",
    "Nephroscopy",
    "Nephrostomy",
    "Net settlement",
    "Netgear DG834 (series)",
    "Network of Cancer Genes",
    "Neural backpropagation",
    "Neuroepidemiology",
    "Neuroergonomics",
    "Neuroimaging",
    "Neuromuscular monitoring",
    "Neuronal ensemble",
    "Neuronal sprouting",
    "Neuroplasticity",
    "Neuroproteomics",
    "Neurotherapy",
    "Neutron stimulated emission computed tomography",
    "New Austrian tunneling method",
    "News analyst",
    "Nickel niobate",
    "Nike+iPod",
    "Nikon VR",
    "Nitrogen trifluoride",
    "No-broadcasting theorem",
    "Nokia Mail for Exchange",
    "Non-contact thermography",
    "Non-contact ultrasound",
    "Non-invasive vagus nerve stimulation devices",
    "Non-lethal weapon",
    "Non-methane volatile organic compound",
    "Non-spiking neuron",
    "Noninvasive glucose monitor",
    "Norketamine",
    "Nuchal scan",
    "Nuclear medicine",
    "Nucleic acid modification databases",
    "Nucleic acid test",
    "Nucleotide universal IDentifier",
    "Nurse call button",
    "Nutrigenomics",
    # --- O (48) ---
    "OPTOS formalism",
    "Observer (weather)",
    "Obstetric ultrasonography",
    "Octreotide scan",
    "Oculus Quest 2",
    "Off-target genome editing",
    "Office chair",
    "Offset ink",
    "Oil shale",
    "Olfactometer",
    "Olfactory receptor",
    "Oligomorphic permutation group",
    "Oligonucleotide",
    "Oncotype DX Colon Cancer Assay",
    "Online electric vehicle",
    "Opaque projector",
    "Open innovation intermediary",
    "Operating cash flow",
    "Operational availability",
    "Operative temperature",
    "Optic crystals",
    "Optical beam smoke detector",
    "Optical brightener",
    "Optical granulometry",
    "Optical ground wire",
    "Optical mapping",
    "Optical mineralogy",
    "Optical table",
    "Optoelectric nuclear battery",
    "Oral rehydration therapy",
    "Organ printing",
    "Organic solar cell",
    "Organizational learning",
    "Organophosphorus chemistry",
    "Orifice plate",
    "Orthopaedic Studio",
    "Orthophoto",
    "Oscillator phase noise",
    "Osmotic power",
    "Osteocalcin",
    "Otoscope",
    "Outsourcing",
    "Overhead microphone",
    "Overhead projector",
    "Oxaliplatin",
    "Oxford Nanopore Technologies",
    "Oxygen sensor",
    "Ozoneweb",
    # --- P (246) ---
    "P wave (electrocardiography)",
    "PDBbind database",
    "PEDOT-TMA",
    "PET radiotracer",
    "PHI-base",
    "PI-RADS",
    "POCUS",
    "PRIAM enzyme-specific profiles",
    "PRINTS",
    "PSI Protein Classifier",
    "Pacemaker",
    "Pacemaker failure",
    "Packaging machinery",
    "Paint adhesion testing",
    "Paired-end tag",
    "Palm (PDA)",
    "Palm print",
    "Pantograph (transport)",
    "Pap test",
    "Paper money",
    "Paracentesis",
    "Paracetamol",
    "Paraffin wax",
    "Parallelogram of force",
    "Pariser–Parr–Pople method",
    "Particle accelerator",
    "Passive solar building design",
    "Passive ventilation",
    "Patch-sequencing",
    "PathoPhenoDB",
    "Pathogenomics",
    "Patient Specific Talus Spacer",
    "PayMe",
    "PayNow",
    "Pedestrian crash avoidance mitigation",
    "Peercoin",
    "Pen",
    "Pencil sharpener",
    "Pendulum-and-hydrostat control",
    "Pentosidine",
    "Peptide biosensor",
    "Peptide sequence",
    "Peptide sequence tag",
    "Peptide-mass fingerprint",
    "PeptideAtlas",
    "Percutaneous",
    "Percutaneous transhepatic cholangiography",
    "Performance status",
    "Perfusion MRI",
    "Perfusion scanning",
    "Peritoneal equilibration test",
    "Permanganate index",
    "Perovskite (structure)",
    "Perovskite solar cell",
    "Personal Genome Project",
    "Personal genomics",
    "Personalized genomics",
    "Perturb-seq",
    "Pfam",
    "PhIP-Seq",
    "Pharmaceutical bioinformatics",
    "Pharmaceutical formulation",
    "Pharmacodiagnostic testing",
    "Pharmacogenomics",
    "Pharmacogenomics annotation",
    "Pharmacomicrobiomics",
    "Pharmacophore",
    "Pharmacy",
    "Phase contrast magnetic resonance imaging",
    "Phenol formaldehyde resin",
    "Phenome",
    "PhenomicDB",
    "Phenomics",
    "Phenoscape",
    "Photo booth",
    "Photoacoustic imaging",
    "Photoactivated adenylyl cyclase",
    "Photoactivated localization microscopy",
    "Photoelectrochemical cell",
    "Photographic paper",
    "Photoreceptor cell",
    "Photovoltaic module analysis techniques",
    "Photovoltaic thermal hybrid solar collector",
    "Phylogenetics",
    "PhylomeDB",
    "Phylosymbiosis",
    "Physiognomy",
    "Physiomics",
    "PictoChat",
    "Pictometry",
    "Pierce Protein Assay",
    "Piezoelectricity",
    "Piezoelectrochemical transducer effect",
    "Piezotronics",
    "Pigment yellow 185",
    "Pillow-plate heat exchanger",
    "Pingping (payment)",
    "Pinhead mirror",
    "Pinhole camera",
    "Pinyin",
    "Pinyin alphabetical order",
    "Pitot tube",
    "Placard",
    "Plant Phenology Ontology",
    "Plant disease epidemiology",
    "Plant genome assembly",
    "Plasma magnet",
    "Plasma polymerization",
    "Plasmonic lens",
    "Plasmonic solar cell",
    "Plastic card",
    "Platoon (automobile)",
    "Plugless Power",
    "PlusEnergy",
    "Pneumatic actuator",
    "Point-of-care genetic testing",
    "Polar surface area",
    "Polarimeter",
    "Police body camera",
    "Political forecasting",
    "Polka dot sign",
    "Pollen DNA barcoding",
    "Poly(p-phenylene vinylene)",
    "Polyacetylene",
    "Polyaniline nanofibers",
    "Polycrystalline silicon",
    "Polyethylenimine",
    "Polygenic score",
    "Polymer Battery Experiment",
    "Polymer electrolytes",
    "Polymer sponge",
    "Polymer-fullerene bulk heterojunction solar cell",
    "Polynomial interpolation",
    "Polynomial long division",
    "Polyphenylene sulfide",
    "Polyphosphate kinase",
    "Polyposis registries",
    "Polyvinyl alcohol",
    "Polyvinylidene fluoride",
    "Pontis",
    "Population genomics",
    "Portable magnetic resonance imaging",
    "Portable ultrasound",
    "Portal vein embolization",
    "Portrait Professional",
    "Positional sequencing",
    "Positron emission mammography",
    "Positron emission tomography",
    "Postal Alpha Numeric Encoding Technique",
    "Postal stationery",
    "Postpaid mobile phone",
    "Postsynaptic potential",
    "Potentiometric sensor",
    "Pouchoscopy",
    "Pound sterling",
    "Powell's dog leg method",
    "Power Glove",
    "Power Wheels",
    "Power optimizer",
    "PowerNow!",
    "Powerlight Technologies",
    "Powermat Technologies",
    "Precision Lightweight GPS Receiver",
    "PrecisionFDA",
    "Preclinical SPECT",
    "Preclinical imaging",
    "Predicted Aligned Error",
    "Prediction game",
    "Predictive genomics",
    "Predictive methods for surgery duration",
    "Premature junctional contraction",
    "Premature ventricular contraction",
    "Prescription analytics",
    "Prescription monitoring program",
    "Pressure reduction regulator",
    "Pressure-sensitive adhesive",
    "Pressuretrol",
    "Pressurisation ductwork",
    "Pretargeting (imaging)",
    "Preview (Apple)",
    "Primer Premier",
    "Prior probability",
    "Priority signs",
    "Privacy4Cars",
    "Probabilistic Approach for Protein NMR Assignment Validation",
    "Process camera",
    "Process validation",
    "Programmable Cricket",
    "Programmed Airline Reservations System",
    "Progress testing",
    "Project Digits",
    "Project Iris",
    "Project Nightingale",
    "Pronunciation respelling",
    "Propaganda",
    "Propaganda through media",
    "Propagation of uncertainty",
    "Propensity score matching",
    "Proportional navigation",
    "Prostate cancer screening",
    "Prosthesis",
    "Protein Data Bank",
    "Protein Segment Finder",
    "Protein aggregation predictors",
    "Protein contact map",
    "Protein domain",
    "Protein fragment library",
    "Protein kinase domain",
    "Protein mass spectrometry",
    "Protein microarray",
    "Protein sequencing",
    "Protein structure database",
    "Protein subcellular localization prediction",
    "Protein tandem repeats",
    "Protein targeting",
    "Protein–protein interaction",
    "Protein–protein interaction prediction",
    "Protein–protein interaction screening",
    "Proteogenomics",
    "Proteome",
    "Proteome Analyst",
    "Proteomic profiling",
    "Proteomics",
    "Protospacer adjacent motif",
    "Proxflyer",
    "Proximity extension assay",
    "Pseudo amino acid composition",
    "Psychotherapy and social action model",
    "PubMed",
    "Public broadcasting",
    "Public health genomics",
    "Public health surveillance",
    "Public transport timetable",
    "Pulmonary angiography",
    "Pulmonary function testing",
    "Pulmonary wedge pressure",
    "Pulse tube refrigerator",
    "Pulsed field ablation",
    "Pulsed radiofrequency",
    "Pump thrombosis",
    "Pupillary distance",
    "Purchase-to-pay",
    "Pyelogram",
    "Pyrheliometer",
    "Pyrometer",
    "Pyroxasulfone",
    # --- Q (13) ---
    "Q-slope",
    "Q-system (geotechnical engineering)",
    "QRIS",
    "QRISK",
    "Quantitative computed tomography",
    "Quantitative investing",
    "Quantitative models of the action potential",
    "Quantitative proteomics",
    "Quantitative structure–activity relationship",
    "Quantitative susceptibility mapping",
    "Quantitative trait locus",
    "Queue jump",
    "Quill drive",
    # --- R (165) ---
    "RADAR (audio recorder)",
    "REE Automotive",
    "RELAP5-3D",
    "RFIQin",
    "RNA CoSSMos",
    "RNA Modification Base",
    "RNA immunoprecipitation chip",
    "RNA velocity",
    "RNA-Seq",
    "ROV KIEL 6000",
    "ROV PHOCA",
    "RXNO Ontology",
    "RaceCam",
    "Rack and pinion",
    "Rackspace Cloud",
    "Radar astronomy",
    "Radar detector",
    "Radar detector detector",
    "Radar speed gun",
    "Radar speed sign",
    "Radar tower",
    "Radial polarization",
    "Radiant cooling",
    "Radiation Exposure Monitoring",
    "Radiation lobectomy",
    "Radiation protection of patients",
    "Radiative cooling",
    "Radiator",
    "Radiator (engine cooling)",
    "Radiator (heating)",
    "Radiator cabinet",
    "Radiator reflector",
    "Radio beacon",
    "Radio beacon station",
    "Radio broadcasting",
    "Radio-controlled aircraft",
    "Radio-controlled boat",
    "Radio-controlled glider",
    "Radio-controlled model",
    "Radio-frequency induction",
    "Radio-frequency skin tightening",
    "Radio-frequency welding",
    "Radiofrequency ablation",
    "Radiofrequency coil",
    "Radiographic anatomy",
    "Radiographic classification of osteoarthritis",
    "Radiography",
    "Radiology",
    "Radiometer",
    "Radiometric calibration",
    "Radiometry",
    "Radioplayer",
    "Railway electric traction",
    "Rainbow hologram",
    "Raindrop size distribution",
    "Raman spectroelectrochemistry",
    "Randomization",
    "Range extender",
    "Rangefinder camera",
    "Rank correlation",
    "Ranking",
    "Rapid 3D Mapping",
    "Rapid Evolution",
    "Rapid amplification of cDNA ends",
    "Rapid antigen test",
    "Rapid strep test",
    "Rapid tooling",
    "Rapid transit",
    "Rapier Unmanned Helicopter",
    "Rav-Kav",
    "Ray (optics)",
    "Reach (advertising)",
    "Reaction video",
    "Reactive bonding",
    "Read (biology)",
    "Real-Time AudioSuite",
    "RealCall",
    "RealD 3D",
    "Realization (linguistics)",
    "Reassurance marker",
    "Reception report",
    "Rechargeable battery",
    "Recliner",
    "Recognition sequence",
    "Recognition signal",
    "RecoverPoint",
    "Recrystallization (chemistry)",
    "Rectal thermometry",
    "Rectilinear scanner",
    "Red light camera",
    "Reduced representation bisulfite sequencing",
    "Reel trailer",
    "Reference genome",
    "Reflectometry",
    "Reflex camera",
    "Refrigerant",
    "Refrigeration",
    "Refrigerator",
    "Refrigerator car",
    "Refrigerator truck",
    "Regenerative heat exchanger",
    "Registered fax",
    "Registration black",
    "Regulatory technology",
    "Reluctance motor",
    "Remote control locomotive",
    "Remote deposit",
    "Remote guidance",
    "Remote therapy",
    "Renal sympathetic denervation",
    "Renal ultrasonography",
    "Renewable heat",
    "Repeating coil",
    "Representative money",
    "Reserve currency",
    "Reserve requirement",
    "Resistance thermometer",
    "Resistome",
    "Resonance",
    "Resonance Raman spectroscopy",
    "Responsive neurostimulation device",
    "Resting state fMRI",
    "Restriction landmark genomic scanning",
    "Restriction site associated DNA markers",
    "Resuscitative endovascular balloon occlusion of the aorta",
    "Retinal birefringence scanning",
    "Retro screening",
    "Retrograde pyelogram",
    "Retrograde tracing",
    "Retroreflective sheeting",
    "Reverse SMS billing",
    "Reverse domain hijacking",
    "Reverse telephone directory",
    "Rheoencephalography",
    "Ribbon microphone",
    "Ribonomics",
    "Ribonuclease III",
    "Right to Internet access",
    "Rigid belt actuator",
    "Rigid chain actuator",
    "Ring circuit",
    "Ringback number",
    "Ringing (telephony)",
    "Ripple tank",
    "RoboTurb",
    "Robocall",
    "Robosaurus",
    "Robot football",
    "Robot tax",
    "Robot-sumo",
    "Robotaxi",
    "Robotic pet",
    "Robotis Bioloid",
    "Rocket sled launch",
    "Roentgen stereophotogrammetry",
    "Roller screw",
    "Romanization",
    "Rotary converter",
    "Rotary converter plant",
    "Rotary dial",
    "Rotary storage",
    "Rubber-tyred metro",
    "Rubber-tyred tram",
    "Rudder travel limiter",
    "Runoff model (reservoir)",
    # --- S (186) ---
    "S Voice",
    "S1850M",
    "SAP BTP",
    "SAP Cloud Infrastructure",
    "SAP ERP",
    "SAP Graphical User Interface",
    "SENSOR-Pesticides",
    "SETAR (model)",
    "SHAREit",
    "SIGMET",
    "SM-COVID-19",
    "SMART Health Card",
    "SMART Retainer",
    "SNP annotation",
    "SNPedia",
    "SPOT Satellite Messenger",
    "STR analysis",
    "SU(1,1) interferometry",
    "SY control cable",
    "Sacral nerve stimulation",
    "SafeSearch",
    "Safran Patroller",
    "Sales letter",
    "Salisbury screen",
    "Saliva hormone testing",
    "Saliva testing",
    "Salt spray test",
    "San Diego Supercomputer Center",
    "Satellite radio",
    "Satellite television",
    "Sawfish harvester",
    "Scanning fiber endoscope",
    "Scanning near-field ultrasound holography",
    "Scheimpflug principle",
    "Schiff test",
    "Schmidt hammer",
    "Schottky junction solar cell",
    "Schuller's view",
    "Scimitar antenna",
    "Secretomics",
    "Self-focusing transducers",
    "Sense (electronics)",
    "Sensory neuroscience",
    "Sequence motif",
    "Sequence-tagged site",
    "Sequencing by hybridization",
    "Sequential structure alignment program",
    "Series and parallel circuits",
    "Serum Metabolome Database",
    "Setright Machine",
    "Shear wave elastography",
    "Shell-and-tube heat exchanger",
    "Ship's wheel",
    "Shock mount",
    "Shockley–Queisser limit",
    "Shotgun proteomics",
    "Shotgun sequencing",
    "Sight glass",
    "Sigmoidoscopy",
    "Signature",
    "Significance analysis of microarrays",
    "Silverquant",
    "Simbionix",
    "Simulated patient",
    "Single cell epigenomics",
    "Single molecule fluorescent sequencing",
    "Single photon absorptiometry",
    "Single-cell DNA template strand sequencing",
    "Single-cell sequencing",
    "Single-cell transcriptomics",
    "Single-molecule magnetic sequencing",
    "Single-molecule real-time sequencing",
    "Single-photon emission computed tomography",
    "Skin electrode gel",
    "Skin temperature (atmosphere)",
    "Sleep spindle",
    "Slicer (guitar effect)",
    "Small Molecule Pathway Database",
    "Small RNA sequencing",
    "Small bowel faeces sign",
    "SnRNA-seq",
    "Social accounting",
    "Social distancing",
    "Sodium MRI",
    "Solar air conditioning",
    "Solar air heat",
    "Solar bus",
    "Solar cell",
    "Solar cell fabric",
    "Solar controller",
    "Solar generator",
    "Solar inverter",
    "Solar irradiance",
    "Solar lamp",
    "Solar panel",
    "Solar shingle",
    "Solar vehicle",
    "Solar-cell efficiency",
    "Solar-powered aircraft",
    "Solar-powered waste compacting bin",
    "Solid-state dye laser",
    "Somatosensory evoked potential",
    "Sonographer",
    "Sonophoresis",
    "Sound attenuator",
    "Source attribution",
    "Spark chamber",
    "Spatial frequency domain imaging",
    "Spatially offset Raman spectroscopy",
    "Speckle (interference)",
    "Speckle pattern",
    "Speckle tracking echocardiography",
    "Spectral imaging (radiography)",
    "Spin coating",
    "Spin engineering",
    "Spinal cord stimulator",
    "Spinal fMRI",
    "Spliceman",
    "Sprinkler fitting",
    "Sputum culture",
    "Stabilator",
    "Stacked thimble",
    "Stamping (metalworking)",
    "StarBase (biological database)",
    "Statistical model specification",
    "Statistical potential",
    "Statistical study of energy data",
    "Steady-state free precession imaging",
    "Stealth ship",
    "Steam jet cooling",
    "Stereoelectroencephalography",
    "Stereotactic surgery",
    "Stethoscope",
    "Stimulated Raman spectroscopy",
    "Storage heater",
    "Storage water heater",
    "Stove",
    "Strain–encoded magnetic resonance imaging",
    "Stretta procedure",
    "Stroke order",
    "Structural genomics",
    "Structure-based combinatorial protein engineering",
    "Structured illumination light sheet microscopy",
    "Subcutaneous implantable defibrillator",
    "Substitution matrix",
    "Superconcentrated electrolytes",
    "Superconducting electric machine",
    "Superconductivity",
    "Superparamagnetic iron–platinum particles",
    "Supramolecular electronics",
    "Sure Chill Technology",
    "Surface plasmon",
    "Surface plasmon resonance",
    "Surface plasmon resonance microscopy",
    "Surface weather observation",
    "Surface-enhanced laser desorption/ionization",
    "Surrogate data",
    "Surround sound",
    "Surveillance Camera Code of Practice",
    "Susceptibility weighted imaging",
    "Sustainability measurement",
    "Sustainable Slip Resistance",
    "Swathi Weapon Locating Radar",
    "Sweet Home 3D",
    "Swing-door operator",
    "Swing-lens camera",
    "Swiss-model",
    "Symbol",
    "Symbol (formal)",
    "Synaptic plasticity",
    "Synaptic potential",
    "Synchronization",
    "Synchronization of chaos",
    "Synchronize It",
    "Synchrotron light source",
    "Synthetic MRI",
    "Synthetic Organism Designer",
    "Synthetic division",
    "Synthetic genomes",
    "Synthetic genomics",
    "Synthetic microbial consortia",
    "Synthetic vaccine",
    "Systemness",
    "Systems biology",
    "Systems chemistry",
    "Systems immunology",
    # --- T (167) ---
    "T wave",
    "T wave alternans",
    "T-money",
    "T2*-weighted imaging",
    "TCL1 MTCP1 protein domain",
    "TIGRFAMs",
    "TIRA",
    "TRAFFICTHIEF",
    "TRANSFAC",
    "Tachometer",
    "Tachymetric anti-aircraft fire control system",
    "Tactical Control System",
    "Tag SNP",
    "Tailplane",
    "Take-back system",
    "Talairach coordinates",
    "Talk box",
    "TalkBack",
    "Tally counter",
    "Tantalum pentoxide",
    "Tape cartridge",
    "Tape dispenser",
    "Target acquisition",
    "Target drone",
    "TargetScan",
    "Targeted analysis sequencing",
    "Tax refund",
    "Taxonomy",
    "Tcr-seq",
    "Technology demonstration",
    "Techron",
    "Teleblock",
    "Telecentric lens",
    "Telecollaboration",
    "Teleconference",
    "Teledermatology",
    "Telegraph pole",
    "Telehealth",
    "Telehomecare",
    "Telemarketing",
    "Telemedicine",
    "Telenursing",
    "Telepathology",
    "Telepharmacy",
    "Telephone banking",
    "Telephone call",
    "Telephone card",
    "Telephone counseling",
    "Telephone country code",
    "Telephone directory",
    "Telephone interpreting",
    "Telephone line",
    "Telephone number",
    "Telephone switchboard",
    "Telepsychiatry",
    "Telepsychology",
    "Teleradiology",
    "Telerehabilitation",
    "Telestream pipeline",
    "Teletraffic engineering",
    "Television advertising workflow",
    "Television broadcaster",
    "Television news screen layout",
    "Television set",
    "Television transmitter",
    "Terminal aerodrome forecast",
    "Test drive",
    "Test preparation",
    "Test score",
    "TestComplete",
    "TestFlight",
    "Tetrakis(trimethylsilyl)silane",
    "Tetrode (biology)",
    "Thalamic stimulator",
    "The Algorithm Auction",
    "The Cancer Genome Atlas",
    "The Emotion Machine",
    "The Human-Induced Earthquake Database",
    "The National Map",
    "The long tailpipe",
    "The writing hand",
    "Therapeutic endoscopy",
    "Therapeutic ultrasound",
    "Thermal bag",
    "Thermal comfort",
    "Thermal conductance and resistance",
    "Thermal diode",
    "Thermal efficiency",
    "Thermal energy storage",
    "Thermal expansion valve",
    "Thermal mass refrigerator",
    "Thermoelectric cooling",
    "Thermoelectric heat pump",
    "Thermoelectric temperature control",
    "Thermosiphon",
    "Thermostat",
    "Theta model",
    "Theta wave",
    "Third rail",
    "Third-generation photovoltaic cell",
    "Threading (protein sequence)",
    "Thrombus perviousness",
    "TicketIQ",
    "Tiller",
    "Time temperature indicator",
    "Time-domain diffuse optics",
    "Time-of-flight detector",
    "Time-of-flight ultrasonic determination of 3D elastic constants",
    "Timehop",
    "Timetree",
    "Tin foil hat",
    "Tin-based perovskite solar cell",
    "Tinnitus masker",
    "Tip-enhanced Raman spectroscopy",
    "Tissue Doppler echocardiography",
    "Tissue cryopreservation",
    "Tomoelastography",
    "Top-down proteomics",
    "Torque vectoring",
    "Total air temperature",
    "Towel warmer",
    "Toxgnostics",
    "Toxic unit",
    "Toxicogenomics",
    "Toxicology testing",
    "Toxicovigilance",
    "Traction power network",
    "Traction substation",
    "Tractography",
    "Traffic collision avoidance system",
    "Traffic cone",
    "Traffic light",
    "Traffic warning sign",
    "Trainline",
    "Tram stop",
    "Transaction cost",
    "Transcatheter arterial chemoembolization",
    "Transcranial Doppler",
    "Transcranial direct-current stimulation",
    "Transcranial focused ultrasound",
    "Transcranial pulsed ultrasound",
    "Transcription factor",
    "Transcription factor binding site databases",
    "Transcriptomics technologies",
    "Transcutaneous electrical nerve stimulation",
    "Transcutaneous oxygen measurement",
    "Transcutaneous pacing",
    "Transesophageal echocardiogram",
    "Transit route network design problem",
    "Transmission electron microscopy DNA sequencing",
    "Transplant Benefit Score",
    "Transrectal ultrasonography",
    "Transthoracic echocardiogram",
    "Transvenous pacing",
    "Trinder glucose activity test",
    "Triple smear",
    "Tripod (photography)",
    "Trolley pole",
    "Trolleybus",
    "Trolleytruck",
    "Tuberculosis radiology",
    "Tumor-informed minimal residual disease",
    "TuneSat",
    "Tunebot",
    "Tunnel magnetoresistance",
    "Turning signal",
    "Twin registry",
    # --- U (25) ---
    "U-Report",
    "USDC (cryptocurrency)",
    "Ubuntu One",
    "Udio",
    "Ultra-high-purity steam for oxidation and annealing",
    "Ultra-low temperature freezer",
    "Ultrasonic pulse velocity test",
    "Ultrasonography of chronic venous insufficiency of the legs",
    "Ultrasound Localization Microscopy",
    "Ultrasound computer tomography",
    "Ultrasound energy",
    "Ultrasound-enhanced systemic thrombolysis",
    "Ultrasound-guided lumbar puncture",
    "Ultrasound-modulated optical tomography",
    "Uni-directional vehicle",
    "Unique molecular identifier",
    "United Kingdom Model for End-Stage Liver Disease",
    "United States National Grid",
    "Universal psychometrics",
    "Upper gastrointestinal series",
    "Urban rail transit",
    "Ureteroscopy",
    "Urethroscopy",
    "Urinary tract ultrasound",
    "Uterine artery embolization",
    # --- V (95) ---
    "VIOLIN vaccine database",
    "VRChat",
    "VU0514009",
    "Vaccine cooler",
    "Vacuum cooling",
    "Vaginal ultrasonography",
    "Vaginogram",
    "Vagus nerve stimulation",
    "Variometer",
    "Vascular closure device",
    "Vascular snare",
    "Vasography",
    "Vector incrimination",
    "Vehicle fire suppression system",
    "Vehicle identification number",
    "Vehicle-activated sign",
    "Vending machine",
    "Venography",
    "Venous translucence",
    "Ventilation (architecture)",
    "Ventilation shaft",
    "Ventilation/perfusion scan",
    "Ventilative cooling",
    "Ventricular dyssynchrony",
    "Ventricular fibrillation",
    "Ventricular flutter",
    "Ventricular tachycardia",
    "Veo (text-to-video model)",
    "Veritas Cluster File System",
    "Veritas Cluster Server",
    "Veritas File System",
    "Vertcoin",
    "Vertebrate Genome Annotation Project",
    "Very-long-baseline interferometry",
    "VetBact",
    "Vibration-powered generator",
    "Victor 3900",
    "Video camera",
    "Video projector",
    "Video-oculography",
    "VideoPoet",
    "VideoRay UROVs",
    "VideoScribe",
    "Videocassette recorder",
    "Videostroboscopy",
    "View camera",
    "Vimeo Livestream",
    "Vinegar test",
    "Vipivotide tetraxetan",
    "VirtuSphere",
    "Virtual Singapore",
    "Virtual assistant (occupation)",
    "Virtual audience",
    "Virtual body",
    "Virtual colonoscopy",
    "Virtual economy",
    "Virtual field trip",
    "Virtual goods",
    "Virtual pet",
    "Virtual reality cue reactivity",
    "Virtual reality pornography",
    "Virtual reality roller coaster",
    "Virtual reality sex",
    "Virtual reality sickness",
    "Virtual scientific community",
    "Virtual sex",
    "Virtual team",
    "Virtual zoo",
    "Virtuix Omni",
    "Virus",
    "Viscous vortex domains method",
    "Visibility",
    "Visible light imaging",
    "Vision therapy",
    "VisionMap A3 Digital Mapping System",
    "VisionOS",
    "Visual Components",
    "Visual Studio",
    "VisualDx",
    "Viverse",
    "Vodafone Mobile Connect USB Modem",
    "Voiding cystourethrography",
    "Volatile organic compound",
    "Volixibat",
    "Volocopter VoloRegion",
    "Volocopter Volocity",
    "Voltage clamp",
    "Volume CT",
    "Volume hologram",
    "Voluntary observing ship program",
    "Von Neumann stability analysis",
    "Vopium",
    "Voronoi deformation density",
    "Voronoi diagram",
    "Voucher",
    # --- W (78) ---
    "WAIFW matrix",
    "Walking vehicle",
    "Wallbox",
    "Walrasian auction",
    "Warburg–Christian method",
    "Warning triangle",
    "Washing machine",
    "Waste heat recovery unit",
    "Wastewater surveillance",
    "Water chiller",
    "Water cooling",
    "Water detector",
    "Water flow test",
    "Water heat recycling",
    "Water heating",
    "Water ionizer",
    "Water model",
    "Water transfer printing",
    "Water tunnel (hydrodynamic)",
    "WaterGAP",
    "Waterless coolant",
    "Waterlow score",
    "Watermark",
    "Wattmeter",
    "Wave buoy",
    "Wax motor",
    "Weak measurement",
    "Weapon target assignment problem",
    "Weather (Apple)",
    "Weather balloon",
    "Weather buoy",
    "Weather compensation",
    "Weather radio",
    "Weatherization",
    "Weighing scale",
    "Weight function",
    "Weight-shift control",
    "Weld tests for friction welding",
    "Welding power supply",
    "Well-Being Index",
    "Wells score (pulmonary embolism)",
    "Wet leakage current test",
    "What Is the Electric Car?",
    "Wheel hub motor",
    "Wheelchair power add-on",
    "White noise machine",
    "Whiteboard",
    "Whole genome bisulfite sequencing",
    "Whole genome sequencing",
    "Wide-angle lens",
    "Wii Remote",
    "Wildfire suppression",
    "Wildlife radio telemetry",
    "Wilson-Cowan model",
    "WinFax",
    "Wind controller",
    "Wind transducer",
    "Wind tunnel",
    "Wind turbine",
    "Wind turbine syndrome",
    "Windbelt",
    "Windcatcher",
    "Wire drawing",
    "Wireless device radiation and health",
    "Wireless home phone",
    "Wireless intercom",
    "Wireless microphone",
    "Wireless speaker",
    "Wireless telegraphy",
    "Wiring diagram",
    "Workouts (Apple)",
    "Workplace exposure monitoring",
    "Workplace health surveillance",
    "WorldRemit",
    "Worldspan",
    "Wound closure strip",
    "Wouzee",
    "Wrist rest",
    # --- X (20) ---
    "X-ray computed tomography",
    "X-ray detector",
    "X-ray diffraction computed tomography",
    "X-ray fluorescence holography",
    "X-ray image intensifier",
    "X-ray machine",
    "X-ray marker",
    "X-ray motion analysis",
    "X-ray telescope",
    "X-ray transform",
    "X-ray tube",
    "XCMS Online",
    "XFOIL",
    "XMind",
    "Xbox Avatar",
    "Xenon-enhanced CT scanning",
    "Xeroradiography",
    "Xiaoice",
    "Xiaowei",
    "XyloTron",
    # --- Y (5) ---
    "Yamaha MU-series",
    "Yamartino method",
    "Yandex Translate",
    "Yaw drive",
    "Youyang Gu COVID model",
    # --- Z (6) ---
    "Z-source inverter",
    "Zebrafish Information Network",
    "Zero differential overlap",
    "Zoetron therapy",
    "Zone melting",
    "Zoom lens",
}
MORE_BAD_PATH_PATTERNS = [r"\bculture\b"]


# ------------------------------------------------------------------
# Wiki embedding cache
# ------------------------------------------------------------------

def _file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _noise_fingerprint() -> str:
    payload = {
        "NOISE_TITLE_PATTERNS": NOISE_TITLE_PATTERNS,
        "NOISE_TITLE_ONLY_PATTERNS": NOISE_TITLE_ONLY_PATTERNS,
        "NOISE_DESC_ONLY_PATTERNS": NOISE_DESC_ONLY_PATTERNS,
        "MORE_BAD_PATH_PATTERNS": MORE_BAD_PATH_PATTERNS,
        "HARDCODED_EXCLUSION_TITLES": sorted(HARDCODED_EXCLUSION_TITLES),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_or_build_wiki_embeddings_filtered(
    *,
    wiki_csv_path: Path,
    cache_root: Path,
    embed_model: str,
    device: str,
    title_col: str = WIKI_TITLE_COL,
    desc_col: str = WIKI_DESC_COL,
    wiki_id_col: str = WIKI_ID_COL,
    mode: str = "digital",
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load or build filtered wiki embeddings with caching.

    The ``mode`` parameter controls whether the digital-curated
    ``HARDCODED_EXCLUSION_TITLES`` list is applied. For green-tech matching
    that list excludes ~hundreds of legitimate green-tech pages (microbial
    fuel cell, photoelectrochemical cell, yaw drive, etc.) flagged during
    digital-domain review. The downstream LLM validation in step 04
    (filter-matches with --validate-llm) handles per-page tech vs non-tech
    classification, so the structural HARDCODED list is unnecessary for
    green and only causes false negatives. For ``mode == 'green'`` we skip
    this list; the regex pattern filters (NOISE_TITLE_PATTERNS, etc.) and
    LLM step still apply.
    """
    from sentence_transformers import SentenceTransformer

    cache_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(wiki_csv_path, low_memory=False)
    for col in (wiki_id_col, title_col, desc_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not in wiki CSV. Columns: {list(df.columns)[:80]}")

    exclusion_titles = set() if mode == "green" else HARDCODED_EXCLUSION_TITLES
    if mode == "green":
        print(f"[wiki] mode=green: skipping HARDCODED_EXCLUSION_TITLES (digital-bias bypass)")

    df = flag_noise_pages(
        df,
        title_col=title_col,
        desc_col=desc_col,
        noise_title_patterns=NOISE_TITLE_PATTERNS,
        bad_path_patterns=MORE_BAD_PATH_PATTERNS,
        noise_title_only_patterns=NOISE_TITLE_ONLY_PATTERNS,
        noise_desc_only_patterns=NOISE_DESC_ONLY_PATTERNS,
        exclusion_titles=exclusion_titles,
    )
    df_f = df[~df["flag_noise_structural"]].copy()
    after = len(df_f)
    print(f"[wiki] Noise filtering: kept={after:,} removed={len(df) - after:,}")

    wiki_input_md5 = _file_md5(wiki_csv_path)
    noise_fp = _noise_fingerprint()

    raw = json.dumps(
        {
            "wiki_input_md5": wiki_input_md5,
            "embed_model": embed_model,
            "desc_col": desc_col,
            "title_col": title_col,
            "id_col": wiki_id_col,
            "n_rows_after_filter": int(after),
            "noise_fp": noise_fp,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    cid = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    emb_path = cache_root / f"wiki_{cid}.embeddings.npz"
    ids_path = cache_root / f"wiki_{cid}.ids.npy"
    meta_path = cache_root / f"wiki_{cid}.meta.json"

    if emb_path.exists() and ids_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ok = (
            meta.get("wiki_input_md5") == wiki_input_md5
            and meta.get("embed_model") == embed_model
            and meta.get("desc_col") == desc_col
            and meta.get("title_col") == title_col
            and meta.get("wiki_id_col") == wiki_id_col
            and int(meta.get("n_rows_after_filter", -1)) == after
            and meta.get("noise_fp") == noise_fp
            and bool(meta.get("normalized", False))
        )
        if ok:
            emb = np.load(emb_path)["emb"].astype(np.float32, copy=False)
            ids = np.load(ids_path, allow_pickle=False)
            if emb.shape[0] == after and len(ids) == after:
                print(f"[wiki-cache] HIT: {emb_path}")
                return emb, ids, meta

    print("[wiki-cache] MISS. Computing wiki embeddings (description-only).")
    model = SentenceTransformer(embed_model, device=device)
    texts = df_f[desc_col].astype(str).fillna("").tolist()

    emb = model.encode(
        texts,
        batch_size=WIKI_EMBED_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)

    ids = df_f[wiki_id_col].astype("string").fillna("").astype(str).to_numpy(dtype=np.str_)
    np.savez_compressed(emb_path, emb=emb)
    np.save(ids_path, ids, allow_pickle=False)

    meta = {
        "wiki_input_md5": wiki_input_md5,
        "embed_model": embed_model,
        "desc_col": desc_col,
        "title_col": title_col,
        "wiki_id_col": wiki_id_col,
        "n_rows_after_filter": int(after),
        "noise_fp": noise_fp,
        "normalized": True,
        "created_utc": datetime.utcnow().isoformat() + "Z",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[wiki-cache] Saved embeddings: {emb_path}")
    return emb, ids, meta


# ------------------------------------------------------------------
# FAISS index build / load
# ------------------------------------------------------------------

def build_or_load_faiss_index(
    patent_emb: np.ndarray,
    dim: int,
    index_path: Path,
) -> "faiss.Index":
    """Build an IVF-PQ FAISS index (or load from disk)."""
    import faiss

    if index_path.exists():
        print(f"[faiss] Loading index: {index_path}")
        return faiss.read_index(str(index_path))

    n = patent_emb.shape[0]
    print(f"[faiss] Building IVF-PQ index for {n:,} vectors (dim={dim})")

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(
        quantizer, dim, IVF_NLIST, PQ_M, PQ_NBITS, faiss.METRIC_INNER_PRODUCT,
    )

    train_n = min(TRAIN_N, n)
    rng = np.random.default_rng(SEED)
    train_idx = rng.choice(n, size=train_n, replace=False)
    train_vecs = np.asarray(patent_emb[train_idx], dtype=np.float32)
    print(f"[faiss] Training on {train_n:,} samples...")
    index.train(train_vecs)

    index.nprobe = 32
    add_bs = 200_000
    for start in tqdm(range(0, n, add_bs), desc="[faiss] adding", unit="chunk"):
        end = min(start + add_bs, n)
        xb = np.asarray(patent_emb[start:end], dtype=np.float32)
        index.add(xb)
        del xb
        gc.collect()

    print(f"[faiss] Saving index: {index_path}")
    faiss.write_index(index, str(index_path))
    return index


# ------------------------------------------------------------------
# Reusable similarity search
# ------------------------------------------------------------------

def search_top_similarities(
    wiki_emb: np.ndarray,
    wiki_ids: np.ndarray,
    index: "faiss.Index",
    patent_emb: Optional[np.ndarray] = None,
    k_top: int = K_TOP,
    k_cand: int = K_CAND,
) -> pd.DataFrame:
    """Search FAISS index and return top-K similarity per wiki page.

    Parameters
    ----------
    wiki_emb : np.ndarray
        Wiki page embeddings (n_pages, dim), float32, L2-normalized.
    wiki_ids : np.ndarray
        Page titles corresponding to rows of *wiki_emb*.
    index : faiss.Index
        Trained FAISS index of patent embeddings.
    patent_emb : np.ndarray or None
        Full patent embeddings for exact reranking.  If None, uses
        approximate FAISS scores directly.
    k_top : int
        Number of top matches to return per page.
    k_cand : int
        ANN candidates to retrieve before exact reranking.

    Returns
    -------
    pd.DataFrame
        Columns: wiki_page_title, top1_similarity, ..., topK_similarity,
        top1_patent_idx, ..., topK_patent_idx (integer indices into patent_emb).
    """
    n = wiki_emb.shape[0]
    top_sims = np.zeros((n, k_top), dtype=np.float32)
    top_idxs = np.full((n, k_top), -1, dtype=np.int64)

    for wi in tqdm(range(n), desc="[search] similarity", unit="page"):
        w_vec = np.asarray(wiki_emb[wi:wi + 1], dtype=np.float32)
        D, I = index.search(w_vec, k_cand)
        cand_idx = I[0]
        cand_idx = cand_idx[cand_idx >= 0]
        if len(cand_idx) == 0:
            continue

        if patent_emb is not None:
            # Exact reranking
            cand_vecs = np.asarray(patent_emb[cand_idx], dtype=np.float32)
            scores = cand_vecs @ w_vec[0]
        else:
            scores = D[0, :len(cand_idx)]

        topk = min(k_top, len(scores))
        best_pos = np.argpartition(scores, -topk)[-topk:]
        best_pos = best_pos[np.argsort(-scores[best_pos])]
        top_sims[wi, :topk] = scores[best_pos[:topk]]
        top_idxs[wi, :topk] = cand_idx[best_pos[:topk]]

    cols: dict = {"wiki_page_title": wiki_ids}
    for i in range(k_top):
        cols[f"top{i+1}_similarity"] = top_sims[:, i]
        cols[f"top{i+1}_patent_idx"] = top_idxs[:, i]
    return pd.DataFrame(cols)


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def run_matching(
    wiki_csv: Path,
    patent_parts_dir: Path,
    output_dir: Path,
    mode: str = "green",
    embed_model: str = DEFAULT_EMBED_MODEL,
    wiki_cache_root: Optional[Path] = None,
    k_top: int = K_TOP,
    k_cand: int = K_CAND,
) -> Path:
    """Run wiki-to-patent ANN matching and write a CSV of top-K matches.

    Parameters
    ----------
    wiki_csv : Path
        Cleaned wiki taxonomy CSV (must contain page_title, description).
    patent_parts_dir : Path
        Directory containing ``part_*.parquet`` patent files.
    output_dir : Path
        Output directory for CSV, FAISS index, checkpoint, and meta.
    mode : str
        ``"green"`` or ``"digital"`` -- selects the shared patent embedding cache.
    embed_model : str
        SentenceTransformer model id.
    wiki_cache_root : Path or None
        Cache directory for wiki embeddings; defaults to ``output_dir/_wiki_cache``.
    k_top : int
        Number of top matches per wiki page.
    k_cand : int
        ANN candidates to retrieve before exact reranking.

    Returns
    -------
    Path
        Path to the output CSV file.
    """
    import faiss  # noqa: F811 (re-import for type checker)

    output_dir = Path(output_dir)
    ensure_dir(str(output_dir))

    if wiki_cache_root is None:
        wiki_cache_root = output_dir / "_wiki_cache"

    dev = device_auto()
    print(f"[run_matching] device={dev}")

    parts = _list_part_files_shared(str(patent_parts_dir))
    n_pat = _fast_row_count_shared(parts)

    # 1) Patent embeddings (from shared cache)
    patent_emb, _patent_ids = _load_patent_embeddings(
        mode=mode, view="abstract", embed_model=embed_model,
        cache_root=PATENT_EMBED_CACHE_ROOT, corpus_root=PATENT_CORPUS_ROOT,
    )
    dim = int(patent_emb.shape[1])

    # 2) Wiki embeddings
    wiki_emb, wiki_ids, _ = load_or_build_wiki_embeddings_filtered(
        wiki_csv_path=wiki_csv,
        cache_root=wiki_cache_root,
        embed_model=embed_model,
        device=dev,
        mode=mode,
    )

    if wiki_emb.shape[1] != dim:
        raise RuntimeError(
            f"Dim mismatch: patents dim={dim} vs wiki dim={wiki_emb.shape[1]}"
        )

    print(f"[patent] n={n_pat:,} dim={dim}")
    print(f"[wiki] n={wiki_emb.shape[0]:,} dim={wiki_emb.shape[1]}")

    # 3) FAISS index
    faiss_path = output_dir / "faiss_patent.index"
    index = build_or_load_faiss_index(patent_emb, dim, faiss_path)

    # 4) Search (reusable function)
    df_matches = search_top_similarities(
        wiki_emb=wiki_emb,
        wiki_ids=wiki_ids,
        index=index,
        patent_emb=patent_emb,
        k_top=k_top,
        k_cand=k_cand,
    )

    # 5) Resolve patent titles for output CSV
    all_titles: list[str] = []
    for p in tqdm(parts, desc="[patent] loading titles"):
        df_t = pd.read_parquet(p, columns=[PATENT_TITLE_COL])
        all_titles.extend(
            df_t[PATENT_TITLE_COL].astype("string").fillna("").astype(str).tolist()
        )

    out_csv = output_dir / "wiki_top5_patents.csv"
    header = ["wiki_page_title"]
    for i in range(1, k_top + 1):
        header += [f"top{i}_similarity", f"top{i}_patent_title"]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for _, mrow in df_matches.iterrows():
            row = [mrow["wiki_page_title"]]
            for j in range(1, k_top + 1):
                sim = mrow[f"top{j}_similarity"]
                pidx = int(mrow[f"top{j}_patent_idx"])
                if pidx >= 0 and sim > 0:
                    row += [float(sim), all_titles[pidx]]
                else:
                    row += ["", ""]
            writer.writerow(row)

    print(f"[done] CSV: {out_csv}")
    return out_csv


# ------------------------------------------------------------------
# filter-matches: threshold + LLM taxonomy validation
# ------------------------------------------------------------------

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

_LLM_SYSTEM_PROMPT = "You are a precise taxonomy cleaner that returns strict JSON outputs."


def _build_llm_validation_prompt(
    page_title: str, description: str,
    mode: str = "digital",
) -> str:
    if mode == "digital":
        mode_label = "digital"
        mode_examples_1 = (
            '(e.g., "Transformer architecture," "TCP/IP," "CMOS sensor," '
            '"Blockchain protocol," "AES encryption," "PageRank")'
        )
        mode_examples_0 = (
            '     * A mathematical concept, theorem, or problem (e.g., "Granger causality," "Travelling salesman\n'
            '       problem," "Four-valued logic," "Steiner problem")\n'
            '     * A physics phenomenon, property, or model (e.g., "Six degrees of freedom," "Inverted pendulum,"\n'
            '       "Shot noise," "Gimbal lock," "Mode collapse")\n'
            '     * A theoretical framework, abstract principle, or classification system\n'
            '     * A material or substance (e.g., "Indium tin oxide") rather than a device or system that uses it\n'
            '     * A category or class of things rather than a specific technology (e.g., "Medium-altitude\n'
            '       long-endurance UAV" is an aircraft category, not a technology)'
        )
        mode_non_domain = (
            "a non-digital domain (e.g., geology, mining, biology, agriculture, civil engineering, "
            "cryobiology, geotechnical methods) even if it uses the word \"system\" or \"method\""
        )
    elif mode == "green":
        mode_label = "green / environmental"
        mode_examples_1 = (
            '(e.g., "Wind turbine," "Carbon capture," "Battery recycling," "Solar inverter")'
        )
        mode_examples_0 = (
            '     * A mathematical concept, theorem, or problem (e.g., "Granger causality," "Travelling salesman\n'
            '       problem," "Four-valued logic")\n'
            '     * A physics phenomenon, natural process, or model (e.g., "Carnot cycle," "Greenhouse effect,"\n'
            '       "Albedo," "Photosynthesis")\n'
            '     * A theoretical framework, abstract principle, or classification system\n'
            '     * A material or substance (e.g., "Indium tin oxide") rather than a device or system that uses it\n'
            '     * A category or class of things rather than a specific technology'
        )
        mode_non_domain = (
            "a non-green domain (e.g., pure computing, social media, entertainment, finance) "
            "with no clear environmental or energy relevance"
        )
    else:
        mode_label = "digital or green"
        mode_examples_1 = (
            '(e.g., "Transformer architecture," "TCP/IP," "Wind turbine," "Carbon capture")'
        )
        mode_examples_0 = (
            '     * A mathematical concept, theorem, or problem\n'
            '     * A physics phenomenon, property, or model\n'
            '     * A theoretical framework, abstract principle, or classification system\n'
            '     * A material or substance rather than a device or system that uses it\n'
            '     * A category or class of things rather than a specific technology'
        )
        mode_non_domain = ""

    non_domain_line = ""
    if mode_non_domain:
        non_domain_line = f"     * {mode_non_domain.capitalize()}\n"

    return f"""You are helping to clean and validate a taxonomy of {mode_label} technologies.

You are given:
- page_title: the Wikipedia page title
- description: a short textual description of the page

Your tasks:

1) Classify the provided Wikipedia page into one of two categories (1 or 0):
 - Return 1 (Core {mode_label.title()} Technology): If the page describes something that can be built, implemented,
   or deployed and has a clear function — it solves a problem, performs an operation, or enables a
   capability in the {mode_label} technology domain {mode_examples_1}. Do not include classification or rating systems.
 - Return 0 (Non-Technology/Out-of-Scope): If the page describes:
{mode_examples_0}
     * A person, company, brand, commercial product, service, organization, event, or location
     * A specific named facility, infrastructure project, or piece of equipment (e.g., a particular
       dam, power plant, refinery, wind farm, factory, building, locomotive class, vehicle model)
       rather than the generic technology category that facility/instance exemplifies
     * History, social impact, or cultural context of a technology rather than its technical mechanics
{non_domain_line}
   Key test: A technology DOES something. A concept DESCRIBES or EXPLAINS something. If the page
   is about something that is USED within technologies but is not itself buildable or implementable,
   return 0.

Strict Rule (commercial products): If the page is a specific commercial product (e.g., "iPhone") or
a platform (e.g., "Facebook"), you must return 0.

Strict Rule (named instances): If the page describes a SPECIFIC NAMED INSTANCE of a technology —
one particular dam, power plant, wind farm, hydroelectric station, refinery, factory, named
locomotive class, named vehicle model, named building or infrastructure project — rather than
the generic technology category, you must return 0. Examples of the distinction:
  - "Three Gorges Dam" → 0 (specific facility)   |  "Hydroelectric power" → 1 (technology)
  - "Lamu Wind Power Station" → 0 (specific facility)  |  "Wind turbine" → 1 (technology)
  - "Tesla Model 3" → 0 (specific product)  |  "Lithium-ion battery" → 1 (technology)
  - "South African Class 10E2" → 0 (specific locomotive class)  |  "Electric locomotive" → 1 (technology)
  - "Prelude FLNG" → 0 (specific vessel)  |  "Floating LNG" → 1 (technology)
The page describes a generic technology if and only if it would be implementable across many
projects/facilities. A specific facility is one project at one location.

2) Rate your confidence in your is_technology classification as a probability between 0 and 1:
   0.0 = no confidence in your answer, 1.0 = fully confident in your answer.
   Use the full range (e.g., 0.95 for clear-cut cases, 0.5 for borderline).

3) If is_technology=1, extract the core functionality:
   - functionality_keywords: 1-5 comma-separated ACTION PHRASES describing what this technology does.
     Each phrase must describe a specific operation, action, or capability — NOT a category label.

     GOOD (action-oriented, specific to this technology):
       "TCP/IP" → "routes packets between networks, ensures reliable delivery via acknowledgments,
                    fragments and reassembles datagrams"
       "Pulse-Doppler radar" → "measures target range via pulse timing, extracts velocity from
                                Doppler frequency shift, separates moving targets from ground clutter"
       "AES encryption" → "encrypts fixed-size data blocks using substitution-permutation rounds,
                           expands cipher key into round keys, mixes columns for diffusion"

     BAD (DO NOT output category labels or generic terms):
       "TCP/IP" → "network protocol, data transmission, internet communication"  ← describes WHAT IT IS
       "Pulse-Doppler radar" → "target detection, signal processing, radar system"  ← too generic
       "Bitcoin" → "payment protocol, cryptocurrency, blockchain"  ← category labels

     Focus on what makes THIS technology DISTINCT from similar technologies.

   - functionality_explanation: 1-2 sentences (max 50 words) explaining the technology's core mechanism
   - confidence_functionality: confidence in the extracted functionality, between 0.0 and 1.0
   If is_technology=0, return empty strings and 0.0 for these fields.

4) Provide a SHORT reasoning (max 50 words) explaining why you chose the is_technology value.

CRITICAL:
- Return your answer as STRICT JSON ONLY.
- Do NOT include any other text, no markdown, no natural language outside JSON.
- JSON fields:
    "is_technology": 1 or 0 (integer)
    "confidence_technology": number between 0.0 and 1.0
    "functionality_keywords": string (comma-separated, or empty string if is_technology=0)
    "functionality_explanation": string (max 50 words, or empty string if is_technology=0)
    "confidence_functionality": number between 0.0 and 1.0 (0.0 if is_technology=0)
    "reasoning": string (max 50 words)

Now analyze this item:

page_title: {page_title}
description: {description}""".strip()


def _call_llm_validation(
    api_key: str,
    page_title: str,
    description: str,
    model: str,
    mode: str = "digital",
) -> Dict:
    """Call OpenRouter for a single page, returning structured JSON."""
    import json as _json
    import requests

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "Taxonomy Technology Validator",
    }

    prompt = _build_llm_validation_prompt(page_title, description, mode=mode)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _LLM_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "TaxonomyClassification",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "is_technology": {"type": "integer"},
                        "confidence_technology": {"type": "number"},
                        "functionality_keywords": {"type": "string"},
                        "functionality_explanation": {"type": "string"},
                        "confidence_functionality": {"type": "number"},
                        "reasoning": {"type": "string"},
                    },
                    "required": [
                        "is_technology",
                        "confidence_technology",
                        "functionality_keywords",
                        "functionality_explanation",
                        "confidence_functionality",
                        "reasoning",
                    ],
                    "additionalProperties": False,
                },
            },
        },
    }

    resp = requests.post(
        OPENROUTER_API_URL, headers=headers, json=payload, timeout=(10, 60),
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter API error: {resp.status_code} - {resp.text}")

    data = resp.json()
    content = data["choices"][0]["message"]["content"].strip()
    result = _json.loads(content)

    return {
        "is_technology": 1 if int(result.get("is_technology", 0)) == 1 else 0,
        "confidence_technology": round(float(result.get("confidence_technology", 0.0)), 4),
        "functionality_keywords": str(result.get("functionality_keywords", "")).strip(),
        "functionality_explanation": str(result.get("functionality_explanation", "")).strip(),
        "confidence_functionality": round(float(result.get("confidence_functionality", 0.0)), 4),
        "reasoning": str(result.get("reasoning", "")).strip(),
    }


def _build_llm_validation_prompt_batched(
    pages: List[Dict[str, str]],
    mode: str = "digital",
) -> str:
    """Build a batched prompt for evaluating multiple pages in one LLM call.

    pages: list of {"page_title": str, "description": str}
    Returns a single prompt asking the model to classify all pages and return
    a JSON object {"results": [<one item per input page>, ...]}.
    """
    if mode == "digital":
        mode_label = "digital"
        mode_examples_1 = (
            '(e.g., "Transformer architecture," "TCP/IP," "CMOS sensor," '
            '"Blockchain protocol," "AES encryption," "PageRank")'
        )
        mode_examples_0 = (
            '     * A mathematical concept, theorem, or problem (e.g., "Granger causality," "Travelling salesman\n'
            '       problem," "Four-valued logic," "Steiner problem")\n'
            '     * A physics phenomenon, property, or model (e.g., "Six degrees of freedom," "Inverted pendulum,"\n'
            '       "Shot noise," "Gimbal lock," "Mode collapse")\n'
            '     * A theoretical framework, abstract principle, or classification system\n'
            '     * A material or substance (e.g., "Indium tin oxide") rather than a device or system that uses it\n'
            '     * A category or class of things rather than a specific technology'
        )
        mode_non_domain = (
            "a non-digital domain (e.g., geology, mining, biology, agriculture, civil engineering, "
            "cryobiology, geotechnical methods) even if it uses the word \"system\" or \"method\""
        )
    elif mode == "green":
        mode_label = "green / environmental"
        mode_examples_1 = (
            '(e.g., "Wind turbine," "Carbon capture," "Battery recycling," "Solar inverter")'
        )
        mode_examples_0 = (
            '     * A mathematical concept, theorem, or problem\n'
            '     * A physics phenomenon, natural process, or model\n'
            '     * A theoretical framework, abstract principle, or classification system\n'
            '     * A material or substance rather than a device or system that uses it\n'
            '     * A category or class of things rather than a specific technology'
        )
        mode_non_domain = (
            "a non-green domain (e.g., pure computing, social media, entertainment, finance) "
            "with no clear environmental or energy relevance"
        )
    else:
        mode_label = "digital or green"
        mode_examples_1 = (
            '(e.g., "Transformer architecture," "TCP/IP," "Wind turbine," "Carbon capture")'
        )
        mode_examples_0 = (
            '     * A mathematical concept, theorem, or problem\n'
            '     * A physics phenomenon, property, or model\n'
            '     * A theoretical framework, abstract principle, or classification system\n'
            '     * A material or substance rather than a device or system that uses it\n'
            '     * A category or class of things rather than a specific technology'
        )
        mode_non_domain = ""

    non_domain_line = ""
    if mode_non_domain:
        non_domain_line = f"     * {mode_non_domain.capitalize()}\n"

    pages_block_lines = []
    for i, p in enumerate(pages, 1):
        pages_block_lines.append(f"--- Page {i} ---")
        pages_block_lines.append(f"page_title: {p['page_title']}")
        pages_block_lines.append(f"description: {p['description']}")
        pages_block_lines.append("")
    pages_block = "\n".join(pages_block_lines).rstrip()

    return f"""You are helping to clean and validate a taxonomy of {mode_label} technologies.

You will evaluate {len(pages)} Wikipedia pages. For each page, apply the rules below and produce
one classification result. Return ALL {len(pages)} results in a single JSON array, in the same
order as the input pages. Each result MUST echo back the exact page_title for verification.

For each page:

1) Classify the page into one of two categories (1 or 0):
 - Return 1 (Core {mode_label.title()} Technology): If the page describes something that can be built, implemented,
   or deployed and has a clear function — it solves a problem, performs an operation, or enables a
   capability in the {mode_label} technology domain {mode_examples_1}. Do not include classification or rating systems.
 - Return 0 (Non-Technology/Out-of-Scope): If the page describes:
{mode_examples_0}
     * A person, company, brand, commercial product, service, organization, event, or location
     * A specific named facility, infrastructure project, or piece of equipment (e.g., a particular
       dam, power plant, refinery, wind farm, factory, building, locomotive class, vehicle model)
       rather than the generic technology category that facility/instance exemplifies
     * History, social impact, or cultural context of a technology rather than its technical mechanics
{non_domain_line}
   Key test: A technology DOES something. A concept DESCRIBES or EXPLAINS something. If the page
   is about something that is USED within technologies but is not itself buildable or implementable,
   return 0.

Strict Rule (commercial products): If the page is a specific commercial product (e.g., "iPhone") or
a platform (e.g., "Facebook"), you must return 0.

Strict Rule (named instances): If the page describes a SPECIFIC NAMED INSTANCE of a technology —
one particular dam, power plant, wind farm, hydroelectric station, refinery, factory, named
locomotive class, named vehicle model, named building or infrastructure project — rather than
the generic technology category, you must return 0. Examples of the distinction:
  - "Three Gorges Dam" → 0 (specific facility)   |  "Hydroelectric power" → 1 (technology)
  - "Lamu Wind Power Station" → 0 (specific facility)  |  "Wind turbine" → 1 (technology)
  - "Tesla Model 3" → 0 (specific product)  |  "Lithium-ion battery" → 1 (technology)
  - "South African Class 10E2" → 0 (specific locomotive class)  |  "Electric locomotive" → 1 (technology)
  - "Prelude FLNG" → 0 (specific vessel)  |  "Floating LNG" → 1 (technology)
The page describes a generic technology if and only if it would be implementable across many
projects/facilities. A specific facility is one project at one location.

2) Rate confidence_technology (number in [0, 1]).

3) If is_technology=1, extract:
   - functionality_keywords: 1-5 ACTION PHRASES (specific operations, not category labels)
   - functionality_explanation: 1-2 sentences (max 50 words)
   - confidence_functionality: number in [0, 1]
   If is_technology=0, return empty strings and 0.0.

4) reasoning: short (max 50 words) explaining the is_technology decision.

CRITICAL output format:
- Return STRICT JSON ONLY, no markdown, no commentary outside JSON.
- Top-level shape: {{"results": [<{len(pages)} items in input order>]}}
- Each item MUST include all of these fields:
    "page_title": string (echo the input page_title EXACTLY for verification)
    "is_technology": 1 or 0 (integer)
    "confidence_technology": number in [0, 1]
    "functionality_keywords": string (comma-separated, or empty if is_technology=0)
    "functionality_explanation": string (max 50 words, or empty if is_technology=0)
    "confidence_functionality": number in [0, 1] (0.0 if is_technology=0)
    "reasoning": string (max 50 words)

Now evaluate these {len(pages)} pages:

{pages_block}""".strip()


def _call_llm_validation_batched(
    api_key: str,
    pages: List[Dict[str, str]],
    model: str,
    mode: str = "digital",
) -> List[Dict]:
    """Call OpenRouter once for a batch of pages. Returns list of result dicts.

    Each result is a dict with the same fields as _call_llm_validation, plus
    a "page_title" field echoed by the model so the caller can verify ordering.

    On failure (HTTP error, malformed JSON, schema violation), raises an
    exception. The caller is responsible for falling back to single-page calls.
    """
    import json as _json
    import requests

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "Taxonomy Technology Validator (batched)",
    }

    prompt = _build_llm_validation_prompt_batched(pages, mode=mode)

    item_schema = {
        "type": "object",
        "properties": {
            "page_title": {"type": "string"},
            "is_technology": {"type": "integer"},
            "confidence_technology": {"type": "number"},
            "functionality_keywords": {"type": "string"},
            "functionality_explanation": {"type": "string"},
            "confidence_functionality": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": [
            "page_title",
            "is_technology",
            "confidence_technology",
            "functionality_keywords",
            "functionality_explanation",
            "confidence_functionality",
            "reasoning",
        ],
        "additionalProperties": False,
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _LLM_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "TaxonomyClassificationBatch",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": item_schema,
                        },
                    },
                    "required": ["results"],
                    "additionalProperties": False,
                },
            },
        },
    }

    resp = requests.post(
        OPENROUTER_API_URL, headers=headers, json=payload, timeout=(10, 120),
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter API error: {resp.status_code} - {resp.text[:500]}")

    data = resp.json()
    content = data["choices"][0]["message"]["content"].strip()
    parsed = _json.loads(content)

    raw_results = parsed.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("Batched response missing 'results' array")

    out: List[Dict] = []
    for r in raw_results:
        if not isinstance(r, dict):
            raise ValueError("Batched response item is not an object")
        out.append({
            "page_title": str(r.get("page_title", "")).strip(),
            "is_technology": 1 if int(r.get("is_technology", 0)) == 1 else 0,
            "confidence_technology": round(float(r.get("confidence_technology", 0.0)), 4),
            "functionality_keywords": str(r.get("functionality_keywords", "")).strip(),
            "functionality_explanation": str(r.get("functionality_explanation", "")).strip(),
            "confidence_functionality": round(float(r.get("confidence_functionality", 0.0)), 4),
            "reasoning": str(r.get("reasoning", "")).strip(),
        })
    return out


def _safe_checkpoint_write(df: pd.DataFrame, path: Path) -> None:
    """Write a checkpoint CSV robustly.

    Uses ``QUOTE_ALL`` + ``escapechar='\\\\'`` so any embedded quotes /
    delimiters / special chars in LLM responses or descriptions don't break
    the writer with ``_csv.Error: need to escape, but no escapechar set``.
    """
    import csv as _csv
    df.to_csv(path, index=False, quoting=_csv.QUOTE_ALL, escapechar="\\")


def filter_matched_pages(
    wiki_csv: Path,
    match_dir: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    threshold: float = 0.6,
    validate_llm: bool = False,
    openrouter_model: str = "google/gemini-3-flash-preview",
    sleep_s: float = 0.05,
    mode: str = "digital",
    **_kwargs,
) -> Path:
    """Filter wiki pages: threshold + dedup + noise + LLM validation.

    Run **after** ``match-patents`` (step 03).  The threshold filter uses
    the ``wiki_top5_patents.csv`` from match-patents to drop pages with
    low patent similarity before the (expensive) LLM validation.

    Parameters
    ----------
    wiki_csv : Path
        Cleaned wiki taxonomy CSV (from the ``clean`` step 02).
    match_dir : Path or None
        Output directory from ``match-patents`` (contains
        ``wiki_top5_patents.csv``).  Required for threshold filtering.
    output_csv : Path or None
        Output path for the filtered CSV.  Auto-derived from *wiki_csv*
        when ``None``.
    threshold : float
        Minimum ``top1_similarity`` to keep a page (default 0.6).
    validate_llm : bool
        If True, validate surviving pages via LLM (per-row, resumable).
    openrouter_model : str
        OpenRouter model for LLM validation.
    sleep_s : float
        Delay between LLM API calls.
    mode : str
        Technology domain: ``"digital"``, ``"green"``, or ``"both"``.

    Returns
    -------
    Path
        Path to the output CSV.
    """
    import time

    wiki_csv = Path(wiki_csv)
    if match_dir is not None:
        match_dir = Path(match_dir)

    if output_csv is None:
        stem = wiki_csv.stem.replace("_clean", "")
        filter_dir = wiki_csv.parent.parent / "04_filter"
        filter_dir.mkdir(parents=True, exist_ok=True)
        output_csv = filter_dir / f"{stem}_filtered.csv"
    else:
        output_csv = Path(output_csv)

    df = pd.read_csv(wiki_csv, low_memory=False)
    n_before = len(df)
    print(f"[filter] Loaded wiki CSV: {n_before:,} pages")

    # --- Step 0a: Dedup by page_title ---
    before_dedup = len(df)
    df["page_title"] = df["page_title"].astype(str)
    df = df.drop_duplicates(subset=["page_title"], keep="first").copy()
    n_dedup = before_dedup - len(df)
    if n_dedup:
        print(f"[filter] Dedup by page_title: removed {n_dedup:,}, remaining {len(df):,}")

    # --- Step 0b: Drop pages with empty descriptions ---
    if "description" in df.columns:
        before_desc = len(df)
        df["description"] = df["description"].astype(str).fillna("")
        df = df[df["description"].str.strip().str.len() > 0].reset_index(drop=True)
        n_empty = before_desc - len(df)
        if n_empty:
            print(f"[filter] Empty descriptions: removed {n_empty:,}, remaining {len(df):,}")

    # --- Step 0c: Noise page filtering ---
    # For green mode: skip the digital-curated HARDCODED_EXCLUSION_TITLES list
    # (it contains real green-tech titles like Microbial fuel cell, Yaw drive,
    # Photoelectrochemical cell, etc.). Mirror of the same bypass in
    # `load_or_build_wiki_embeddings_filtered`. The downstream LLM validation
    # below handles per-page tech vs non-tech.
    _NOISE_PATTERNS_FILTER = NOISE_TITLE_PATTERNS + [r"aquarium", r"\bsoil\b"]
    _exclusion_titles = set() if mode == "green" else HARDCODED_EXCLUSION_TITLES
    if mode == "green":
        print(f"[filter] mode=green: skipping HARDCODED_EXCLUSION_TITLES (digital-bias bypass)")
    df = flag_noise_pages(
        df,
        title_col="page_title",
        desc_col="description",
        noise_title_patterns=_NOISE_PATTERNS_FILTER,
        bad_path_patterns=MORE_BAD_PATH_PATTERNS,
        noise_title_only_patterns=NOISE_TITLE_ONLY_PATTERNS,
        noise_desc_only_patterns=NOISE_DESC_ONLY_PATTERNS,
        exclusion_titles=_exclusion_titles,
    )
    n_noise = int(df["flag_noise_structural"].sum())
    df = df[~df["flag_noise_structural"]].drop(columns=["flag_noise_structural"]).reset_index(drop=True)
    if n_noise:
        print(f"[filter] Noise filtering: removed {n_noise:,}, remaining {len(df):,}")

    # --- Step 1: Threshold filter (requires match-patents output) ---
    if match_dir is not None:
        match_csv = match_dir / "wiki_top5_patents.csv"
        if match_csv.exists():
            match_df = pd.read_csv(match_csv, low_memory=False)
            print(f"[filter] Loaded match CSV: {len(match_df):,} rows")

            df = df.merge(
                match_df[["wiki_page_title", "top1_similarity"]],
                left_on="page_title",
                right_on="wiki_page_title",
                how="left",
            )

            keep_mask = df["top1_similarity"].notna() & (df["top1_similarity"] >= threshold)
            n_dropped_thresh = int((~keep_mask).sum())
            df = df[keep_mask].copy()
            df.drop(columns=["wiki_page_title", "top1_similarity"], inplace=True, errors="ignore")

            print(f"[filter] Threshold={threshold}: dropped {n_dropped_thresh:,}, remaining {len(df):,}")
        else:
            print(f"[filter] WARNING: No match CSV at {match_csv}. Run match-patents first.")
    else:
        print(f"[filter] No --match-dir provided, skipping threshold filter")

    # --- Step 2: LLM validation (per-row, resumable via checkpoint file) ---
    if validate_llm and len(df) > 0:
        from shared.llm import read_openrouter_key

        api_key = read_openrouter_key()

        # Checkpoint path: separate file that keeps ALL rows (unfiltered)
        checkpoint_csv = output_csv.parent / (output_csv.stem + "_llm_checkpoint.csv")

        # LLM result columns
        _llm_cols = [
            "llm_is_technology", "llm_confidence_technology",
            "llm_functionality_keywords", "llm_functionality_explanation",
            "llm_confidence_functionality", "llm_reasoning",
        ]
        for col in _llm_cols:
            if col not in df.columns:
                df[col] = None

        # Ensure required columns have string values
        df["page_title"] = df["page_title"].fillna("").astype(str)
        if "description" in df.columns:
            df["description"] = df["description"].fillna("").astype(str)
        else:
            df["description"] = ""

        # Resume logic: load checkpoint if it exists
        start_pos = 0
        if checkpoint_csv.exists():
            try:
                df_ckpt = pd.read_csv(checkpoint_csv, low_memory=False)
                # Detect old-schema checkpoint (v1 had llm_is_hierarchy_sensible)
                if "llm_is_hierarchy_sensible" in df_ckpt.columns and "llm_confidence_technology" not in df_ckpt.columns:
                    print("[filter-llm] Old-schema checkpoint detected (v1). Ignoring — starting fresh.")
                    df_ckpt = None
                if df_ckpt is not None and "llm_is_technology" in df_ckpt.columns and len(df_ckpt) > 0:
                    for col in _llm_cols:
                        if col not in df_ckpt.columns:
                            df_ckpt[col] = None

                    # Match by page_title to handle row-order changes
                    ckpt_map = df_ckpt.set_index("page_title")[_llm_cols].to_dict("index")

                    for idx, row in df.iterrows():
                        title = str(row["page_title"])
                        if title in ckpt_map:
                            vals = ckpt_map[title]
                            for col in _llm_cols:
                                df.at[idx, col] = vals[col]

                    completed = (
                        df["llm_is_technology"].notna()
                        & df["llm_reasoning"].notna()
                        & (df["llm_reasoning"].astype(str).str.len() > 0)
                    )
                    n_done = int(completed.sum())
                    if completed.all():
                        print(f"[filter-llm] Checkpoint complete ({n_done:,} pages). Skipping LLM calls.")
                    else:
                        start_pos = int((~completed).values.argmax())
                        print(f"[filter-llm] Resumed checkpoint: {n_done:,} done, "
                              f"resuming from row {start_pos:,} ({len(df) - start_pos:,} remaining)")
            except Exception as e:
                print(f"[filter-llm] Could not load checkpoint ({e}). Starting from scratch.")
                start_pos = 0

        remaining = df.iloc[start_pos:]
        if len(remaining) > 0:
            print(f"[filter-llm] Validating {len(remaining):,} pages "
                  f"(model={openrouter_model}, batch=5 with per-item fallback)")

            BATCH_SIZE = 5

            def _apply_result_to_df(idx: int, result: Dict) -> None:
                df.at[idx, "llm_is_technology"] = result["is_technology"]
                df.at[idx, "llm_confidence_technology"] = result["confidence_technology"]
                df.at[idx, "llm_functionality_keywords"] = result["functionality_keywords"]
                df.at[idx, "llm_functionality_explanation"] = result["functionality_explanation"]
                df.at[idx, "llm_confidence_functionality"] = result["confidence_functionality"]
                df.at[idx, "llm_reasoning"] = result["reasoning"]

            def _validate_item(item: Dict) -> bool:
                """Per-item validation: required fields present, types ok, ranges ok."""
                try:
                    if not isinstance(item.get("page_title", ""), str) or not item["page_title"].strip():
                        return False
                    if int(item.get("is_technology", -1)) not in (0, 1):
                        return False
                    ct = float(item.get("confidence_technology", -1))
                    cf = float(item.get("confidence_functionality", -1))
                    if not (0.0 <= ct <= 1.0):
                        return False
                    if not (0.0 <= cf <= 1.0):
                        return False
                    for key in ("functionality_keywords", "functionality_explanation", "reasoning"):
                        if not isinstance(item.get(key, ""), str):
                            return False
                except Exception:
                    return False
                return True

            cache: Dict[str, Dict] = {}
            checkpoint_counter = 0
            n_batched_ok = 0
            n_batched_fail_items = 0
            n_single_fallback = 0

            # Build the list of (idx, title, desc) tuples that still need work.
            # (Skip ones already in checkpoint — same logic as before.)
            pending: List[tuple] = []
            for row in remaining.itertuples(index=True):
                idx = row.Index
                if (pd.notna(df.at[idx, "llm_is_technology"])
                        and pd.notna(df.at[idx, "llm_reasoning"])
                        and str(df.at[idx, "llm_reasoning"]).strip()):
                    continue
                pending.append((idx, str(row.page_title), str(row.description)))

            pbar = tqdm(total=len(pending), desc="[filter-llm] pages")

            i = 0
            while i < len(pending):
                batch = pending[i:i + BATCH_SIZE]
                i += len(batch)

                # Resolve cache hits up-front (for duplicate titles within run)
                to_call: List[tuple] = []
                for idx, title, desc in batch:
                    if title in cache:
                        _apply_result_to_df(idx, cache[title])
                        pbar.update(1)
                    else:
                        to_call.append((idx, title, desc))

                if not to_call:
                    continue

                pages_for_llm = [{"page_title": t, "description": d} for (_, t, d) in to_call]
                input_titles = [t for (_, t, _) in to_call]
                results_by_title: Dict[str, Dict] = {}

                # Try batched call (only worth it if more than one page)
                batched_results: Optional[List[Dict]] = None
                if len(to_call) > 1:
                    try:
                        batched_results = _call_llm_validation_batched(
                            api_key=api_key,
                            pages=pages_for_llm,
                            model=openrouter_model,
                            mode=mode,
                        )
                        if sleep_s > 0:
                            time.sleep(sleep_s)
                    except Exception as e:
                        print(f"[filter-llm] batched call failed ({type(e).__name__}: {str(e)[:200]}), "
                              f"falling back to single-page for this batch", flush=True)
                        batched_results = None

                # Match batched results back to inputs by exact title
                if batched_results is not None:
                    available = list(batched_results)
                    for t in input_titles:
                        match_i = None
                        for j, item in enumerate(available):
                            if _validate_item(item) and item.get("page_title") == t:
                                match_i = j
                                break
                        if match_i is not None:
                            r = available.pop(match_i)
                            results_by_title[t] = {
                                k: r[k] for k in (
                                    "is_technology", "confidence_technology",
                                    "functionality_keywords", "functionality_explanation",
                                    "confidence_functionality", "reasoning",
                                )
                            }
                            n_batched_ok += 1

                # Fall back to single-page for any unmatched / invalid items
                for (idx, title, desc) in to_call:
                    if title in results_by_title:
                        result = results_by_title[title]
                    else:
                        if batched_results is not None:
                            n_batched_fail_items += 1
                        n_single_fallback += 1
                        try:
                            result = _call_llm_validation(
                                api_key=api_key,
                                page_title=title,
                                description=desc,
                                model=openrouter_model,
                                mode=mode,
                            )
                            if sleep_s > 0:
                                time.sleep(sleep_s)
                        except Exception as e:
                            result = {
                                "is_technology": None,
                                "confidence_technology": None,
                                "functionality_keywords": "",
                                "functionality_explanation": "",
                                "confidence_functionality": None,
                                "reasoning": f"ERROR: {str(e)[:200]}",
                            }

                    cache[title] = result
                    _apply_result_to_df(idx, result)
                    pbar.update(1)
                    checkpoint_counter += 1

                # Checkpoint every ~50 rows (every ~10 batches)
                if checkpoint_counter >= 50:
                    ensure_dir(str(checkpoint_csv.parent))
                    _safe_checkpoint_write(df, checkpoint_csv)
                    checkpoint_counter = 0

            pbar.close()
            print(f"[filter-llm] batched_ok={n_batched_ok}, "
                  f"batched_failed_items={n_batched_fail_items}, "
                  f"single_fallback_calls={n_single_fallback}")

            # Final checkpoint write (all rows, unfiltered)
            ensure_dir(str(checkpoint_csv.parent))
            _safe_checkpoint_write(df, checkpoint_csv)
            print(f"[filter-llm] Checkpoint saved: {checkpoint_csv}")

        # Apply LLM decisions: keep only pages where is_technology == 1
        keep_mask = (
            df["llm_is_technology"].isna()  # errors: keep
            | (df["llm_is_technology"].astype(float) == 1)
        )
        n_dropped_llm = int((~keep_mask).sum())
        df = df[keep_mask].copy()
        print(f"[filter-llm] LLM dropped {n_dropped_llm:,}, remaining {len(df):,}")

    # --- Write output ---
    ensure_dir(str(output_csv.parent))
    df.to_csv(output_csv, index=False)
    n_after = len(df)
    print(f"[filter] DONE: {n_before:,} → {n_after:,} pages "
          f"(removed {n_before - n_after:,}). Output: {output_csv}")
    return output_csv


# ------------------------------------------------------------------
# Page-level patent relevance
# ------------------------------------------------------------------

def compute_page_relevance(
    pages_csv: Path,
    faiss_index_path: Path,
    output_csv: Optional[Path] = None,
    embed_model: str = DEFAULT_EMBED_MODEL,
    embed_cache_dir: Optional[Path] = None,
    sim_threshold: float = 0.5,
    k_search: int = 2000,
    batch_size: int = 64,
    desc_col: str = "description",
    title_col: str = "page_title",
    mode: str = "digital",
) -> Path:
    """
    Compute per-page patent relevance with two dimensions:

      1. **coverage_pct**: percentage of total unique patents covered by this page
      2. **uniqueness**: mean IDF weight across the page's patents,
         where IDF = 1 / (number of pages covering that patent).
         High uniqueness → this page covers patents that few other pages cover.

    Parameters
    ----------
    pages_csv : Path
        Filtered wiki pages CSV (must have *title_col* and *desc_col*).
    faiss_index_path : Path
        Path to the FAISS patent index (from ``match-patents`` step).
    output_csv : Path or None
        Where to save the relevance CSV.  Defaults to
        ``<faiss_index_dir>/page_relevance.csv``.
    sim_threshold : float
        Minimum cosine similarity to count a patent as "covered".
    k_search : int
        Number of nearest neighbours to retrieve per page from FAISS.

    Returns
    -------
    Path to the saved CSV with columns:
        page_title, n_patents, coverage_pct, uniqueness
    """
    import faiss
    from collections import Counter
    from p2_taxonomy.analysis_cluster import (
        load_or_build_embeddings,
        clean_description_text,
    )

    if output_csv is None:
        output_csv = faiss_index_path.parent / "page_relevance.csv"

    if embed_cache_dir is None:
        embed_cache_dir = faiss_index_path.parent.parent.parent / "cache" / "wiki_embedding_cache"

    # ── Load pages + embeddings ──────────────────────────────────────
    df = pd.read_csv(pages_csv)
    print(f"[page-relevance] {len(df)} pages from {pages_csv}")

    desc_texts = [clean_description_text(x) for x in df[desc_col].tolist()]
    device = device_auto()
    wiki_emb = load_or_build_embeddings(
        desc_texts=desc_texts, input_path=pages_csv, out_dir=embed_cache_dir,
        embed_model=embed_model, device=device, desc_col=desc_col,
    )

    # ── Load patent embeddings for exact reranking ──────────────────
    print(f"[page-relevance] Loading patent embeddings for exact reranking...")
    patent_emb, _patent_ids = _load_patent_embeddings(
        mode=mode, view="abstract", embed_model=embed_model,
        cache_root=PATENT_EMBED_CACHE_ROOT, corpus_root=PATENT_CORPUS_ROOT,
    )
    print(f"[page-relevance] Patent embeddings: {patent_emb.shape[0]:,} × {patent_emb.shape[1]}")

    # ── FAISS search + exact reranking: count patents per page ─────
    print(f"[page-relevance] Loading FAISS index: {faiss_index_path}")
    index = faiss.read_index(str(faiss_index_path))
    if hasattr(index, "nprobe"):
        index.nprobe = 64
    print(f"[page-relevance] Index: {index.ntotal:,} patents, dim={index.d}")

    n_pages = wiki_emb.shape[0]
    patent_counts = np.zeros(n_pages, dtype=np.int32)
    patent_sets: List[set] = [set() for _ in range(n_pages)]

    for start in range(0, n_pages, batch_size):
        end = min(start + batch_size, n_pages)
        batch = np.ascontiguousarray(wiki_emb[start:end], dtype=np.float32)
        D, I = index.search(batch, k_search)
        for j in range(end - start):
            w_vec = batch[j]  # (dim,)
            cand_idx = I[j]
            cand_idx = cand_idx[cand_idx >= 0]
            if len(cand_idx) == 0:
                continue
            # Exact reranking: compute true dot products
            cand_vecs = np.asarray(patent_emb[cand_idx], dtype=np.float32)
            exact_sims = cand_vecs @ w_vec
            above = exact_sims >= sim_threshold
            patent_counts[start + j] = int(above.sum())
            patent_sets[start + j] = set(cand_idx[above].tolist())
        if start % (batch_size * 20) == 0:
            print(f"  [{start}/{n_pages}] searched...", flush=True)

    # ── Compute relevance dimensions ─────────────────────────────────
    # 1. Patent-to-page count (for IDF)
    print("[page-relevance] Computing IDF uniqueness...")
    patent_page_count: Counter = Counter()
    for ps in patent_sets:
        for pid in ps:
            patent_page_count[pid] += 1

    n_total_patents = len(patent_page_count)

    # 2. Per-page: coverage_pct and uniqueness (mean IDF)
    coverage_pct = np.zeros(n_pages, dtype=np.float32)
    uniqueness = np.zeros(n_pages, dtype=np.float32)

    for i, ps in enumerate(patent_sets):
        n_page = len(ps)
        coverage_pct[i] = (n_page / max(n_total_patents, 1)) * 100
        if n_page > 0:
            uniqueness[i] = sum(1.0 / patent_page_count[pid] for pid in ps) / n_page

    # ── Save ─────────────────────────────────────────────────────────
    out_df = pd.DataFrame({
        title_col: df[title_col].values,
        "n_patents": patent_counts,
        "coverage_pct": np.round(coverage_pct, 6),
        "uniqueness": np.round(uniqueness, 6),
    })

    ensure_dir(str(output_csv.parent))
    out_df.to_csv(output_csv, index=False)

    print(f"\n[page-relevance] DONE — {n_pages} pages, "
          f"{n_total_patents:,} total unique patents")
    print(f"  n_patents : mean={patent_counts.mean():.0f}, "
          f"median={np.median(patent_counts):.0f}, "
          f"min={patent_counts.min()}, max={patent_counts.max()}")
    print(f"  coverage %: mean={coverage_pct.mean():.4f}%, "
          f"median={np.median(coverage_pct):.4f}%, max={coverage_pct.max():.4f}%")
    print(f"  uniqueness: mean={uniqueness.mean():.4f}, "
          f"median={np.median(uniqueness):.4f}, max={uniqueness.max():.4f}")
    print(f"  Output: {output_csv}")
    return output_csv
