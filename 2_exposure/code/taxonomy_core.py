"""Final digital taxonomy (user decision 2026-10-07): the approved structural corrections ("core" items) applied to the
page list build_final.py defines (raw P2 page_meta + page-handling plan MOVE/MOVE-NEW/DROP, six quantum pages removed,
Cloud computing added: 7,351 pages, 186 base technologies).

A language model (Claude) read every base technology with its page titles and proposed the corrections; the user approved
the core items on 2026-10-07 (the optional granularity merges were not adopted). Every change carries an item ID and is
justified by technology content only (wrong place, not a digital technology, duplicate page, naming):
  P1-P16  pages moved to another base technology
  R1      pages removed as not digital technologies (REMOVE_PAGES, with the reason)
  S1-S17  structure: a tenth root (Transport & Navigation Systems), new and split technologies, L1 regrouping, renames,
          six non-digital technologies removed (S14)
  D1      duplicate pages (identical description embeddings): the article page is kept
Result: 6,928 pages, 184 base technologies, 43 L1, 10 roots. build_core.py builds the exposure on it.

    cd p3_exposure/threshold_experiment
    python -u taxonomy_core.py        # counts + check against the final build's levels.csv
"""
import re
from collections import defaultdict
import numpy as np, pandas as pd
import core as C
import build_final as BF

FINAL = C.TRI / "pb_mpnet_final_reviewed_q_taxcore_yearly"
TAX = C.DROPBOX / "taxonomy" / "wiki" / "pipeline" / "07_tag_patents" / "digital_v3" / "taxonomy_used.csv"

T, RS, RM, RN, RR, RQ, RA, RCH = ("Transport & Navigation Systems", "Software & Data Management", "Robotics & Manufacturing",
                                  "Networking & Telecommunications", "Cryptography & Security", "Quantum Computing",
                                  "Artificial Intelligence", "Computing Hardware & Systems")
AVIONICS = ["Air data boom", "Air data computer", "Air data module", "Airspeed indicator", "ArduPilot", "Attitude and heading reference system",
            "Attitude indicator", "Autoland", "Autonomous aircraft", "Autonomous cargo ship", "Autopilot", "Electronic flight instrument system",
            "Fibre-optic gyroscope", "Fine guidance sensor", "Flight envelope protection", "Glass cockpit", "Guidance system", "Gyrocompass",
            "Gyroscope", "Hemispherical resonator gyroscope", "Inertial navigation system", "Inertial reference unit",
            "LN-3 inertial navigation system", "Laser guidance", "Microwave landing system", "Mode control panel",
            "Multi-spacecraft Autonomous Positioning System", "Orientation sensing", "Pressure reference system", "Quantum compass",
            "Uninterruptible autopilot", "Unmanned aircraft system simulation"]
TO_GNSS = ["FDOA", "Gpsd", "Receiver autonomous integrity monitoring", "Machine guidance"]
BUILDING = ["BACnet", "Building automation", "Destination dispatch", "Digital Addressable Lighting Interface", "DMX512", "RDM (lighting)",
            "Lighting control system", "Intelligent street lighting", "Direct digital control", "EBUS (serial buses)", "HVAC control system",
            "Hot water reset", "Home Assistant", "HomeLink Wireless Control System", "Instabus", "Insteon", "Intelligent Home Control", "KNX",
            "LonWorks", "Occupancy sensor", "Occupant-centric building controls", "PLCBUS", "Smart home hub", "Smart lock", "Smart plug",
            "Smart thermostat", "Virtual Cybernetic Building Testbed", "Z-Wave", "Zigbee"]
GRID = ["Automatic meter reading", "DNP3", "Dynamic line rating for electric utilities", "Encoder receiver transmitter", "GridLAB-D",
        "IEC 60870", "Meter-Bus", "Power-system automation", "Smart grid", "Smart meter", "Substation Configuration Language"]
BLOCKCHAIN = ["Bitcoin protocol", "Blockchain", "Blockchain oracle", "Chainlink (blockchain oracle)", "Cryptocurrency tracing",
              "Cryptocurrency wallet", "Distributed ledger", "Hyperbridge", "Layer-1 blockchain", "Lightning Network", "Lightweight blockchain",
              "Nakamoto consensus", "Non-fungible token", "Proof of authority", "Proof of identity (blockchain consensus)", "Proof of stake",
              "Proof of work", "Smart contract", "Unspent transaction output"]
WEARABLE = ["Cyclocomputer", "Fitness tracker", "GPS watch", "Smart band", "Smartwatch", "Smart wearable system"]
CHIPS = ["Biochip", "Chemical compound microarray", "DNA microarray", "Microarray", "Peptide microarray", "Phenotype microarray", "SNP array",
         "Suspension array technology", "Tissue microarray"]
REMOVE_PAGES = {  # page: reason (not a digital technology / not a technology)
    "Anonymous matching": "a dating matchmaking method", "Cyborg antenna": "an artistic skull implant",
    "Thought recording and reproduction device": "a hypothetical device", "Thermowell": "a pipe fitting for thermometers",
    "Wearable generator": "power-generating clothing, not a biosensor", "Seismic interferometry": "geophysical survey method",
    "Passive daytime radiative cooling": "building and climate cooling", "Spacecraft thermal control": "spacecraft engineering",
    "Liquid droplet radiator": "spacecraft heat rejection", "Quantum dot solar cell": "photovoltaics",
    "Mimeograph": "mechanical stencil duplicator", "Offset printing": "lithographic press printing",
    "Chinese typewriter": "mechanical typewriter", "Slate and stylus": "manual Braille writing tool",
    "Stereoscope": "19th-century optical viewer", "Laser pointer": "consumer laser gadget", "Image-stabilized binoculars": "optical binoculars",
    "Tram-train": "a vehicle type", "Architectural endoscopy": "architectural model photography", "Responsive architecture": "architectural concept",
    "Light-emitting diode": "generic electronic component and lighting", "Dielectric heating": "industrial heating",
    "Microwave welding": "industrial welding", "Pasotron": "directed-energy microwave source"}
REMOVED_NODES = ("Power Electronics", "Electrical Generators & Dynamos", "Electromagnetic Propulsion & Accelerators",
                 "Impedance Measurement", "Optical Signaling", "EM Wave-Material Interaction")      # S14, not digital
NAME_MAP = {  # base technology of the build_final.py page list -> final name (renames; merges; a split node -> the part
    # holding most of its pages). Used to re-key the cleaning lists and the link-review verdicts (build_core.py).
    "Autonomous Navigation, Inertial Systems & Robot Localization": "Avionics & Inertial Navigation",
    "Embedded Control and Telematics": "Vehicle Electronics & Telematics",
    "Indoor Positioning & Smart Connected Devices": "Indoor Positioning",
    "Service & Domestic Robotics": "Service Robotics",
    "Microarrays & Bioinformatics Sequence Analysis": "Bioinformatics & Sequence Analysis",
    "Digital Production Printing": "Printing & Copying Technology",
    "Modular Data Centers": "Data Center Infrastructure",
    "Spin Qubits": "Spin, Ion, Atom & Photonic Qubits",
    "Display Panels & Lighting Technology": "Display Panels",
    "Superheterodyne Reception": "Radio Receivers",
    "Industrial and Building Automation": "Industrial Control Systems & Fieldbuses",
    "Access Control Systems": "Authentication & Access Control"}


def production_levels():
    """Every page row of the build_final.py page universe (raw page_meta rows + the added pages), with its L2/L1/root;
    removed pages carry BF.OUT_NODE. -> (titles, {L2, L1, root} object arrays)."""
    pm = pd.read_parquet(C.PBAN / "page_meta_patentsberta.parquet")
    assert (pm.page_row.to_numpy() == np.arange(len(pm))).all() and pm.page_title.is_unique
    plan = pd.read_csv(BF.PLAN)
    lv = BF.page_list(pm, plan[plan.action.isin(["MOVE", "MOVE-NEW", "DROP"])])
    titles = np.append(pm.page_title.astype(str).to_numpy(), list(BF.ADD)).astype(object)
    lv = {k: np.append(np.asarray(lv[k], dtype=object), [v[j] for v in BF.ADD.values()]).astype(object)
          for j, k in enumerate(("L2", "L1", "root"))}
    return titles, lv


def base_pages():
    """The live pages of build_final.py (7,351): [page_title, L2, L1, root]."""
    titles, lv = production_levels()
    d = pd.DataFrame({"page_title": titles, **lv})
    return d[d.L2 != BF.OUT_NODE].reset_index(drop=True)


def apply(base):
    """The approved corrections on the live page list. -> (pages [page_title, L2, L1, root], {page: [item IDs]},
    duplicate groups [group, page, L2, keep, kept_page])."""
    pm = pd.read_parquet(C.PBAN / "page_meta_patentsberta.parquet")
    emb = np.load(C.PBAN / "wiki_page_desc_patentsberta.npy")
    desc = pd.read_csv(TAX).description.astype(str).to_numpy()
    row_of = dict(zip(pm.page_title.astype(str), pm.page_row.astype(int)))
    d = base.copy()
    parent = {r.L2: (r.L1, r.root) for r in d.drop_duplicates("L2").itertuples()}
    log = defaultdict(list)
    has = lambda p: (d.page_title == p).sum() == 1

    def tag(mask, item):
        for p in d.loc[mask, "page_title"]:
            log[p].append(item)

    def pages(lst, item, to):
        for p in lst:
            assert has(p), f"{item}: page not found once: {p}"
        m = d.page_title.isin(lst); tag(m, item); d.loc[m, "L2"] = to

    def node_to(node, L1, root, item):
        assert node in parent, f"{item}: node missing {node}"
        parent[node] = (L1, root); tag(d.L2 == node, item)

    def new_node(node, L1, root):
        assert node not in parent, node
        parent[node] = (L1, root)

    def rename(old, new, item):
        assert old in parent and new not in parent, (item, old, new)
        parent[new] = parent.pop(old); m = d.L2 == old; tag(m, item); d.loc[m, "L2"] = new

    def merge(src, dst, item):
        assert src in parent and dst in parent, (item, src, dst)
        m = d.L2 == src; tag(m, item); d.loc[m, "L2"] = dst; parent.pop(src)

    def drop_pages(lst, item):
        for p in lst:
            assert has(p), f"{item}: page not found once: {p}"
        m = d.page_title.isin(lst); tag(m, item); d.drop(d.index[m], inplace=True)

    def drop_node(node, item):
        m = d.L2 == node; tag(m, item); d.drop(d.index[m], inplace=True); parent.pop(node)

    # ---------------- page moves (P) and removals (R) first, while the source nodes still have their names
    pages(["Liquid handling robot", "MiroSurge", "Remote surgery", "Sofie (surgical robot)", "MRI Robot"], "P1", "Medical & Laboratory Robotics")
    pages(["Delivery robot", "Socially assistive robot", "Cosmobot", "Search and rescue robot", "Demining robot",
           "Battlefield Extraction-Assist Robot"], "P2", "Service & Domestic Robotics")
    pages(["Cobot", "Off-line programming (robotics)", "Vision-guided robot systems"], "P3", "Robot Manipulators and Arm Systems")
    pages(["Robotic process automation", "Monte Carlo tree search"], "P4", "Symbolic AI, Knowledge & Agent Systems")
    pages(["On-street parking sensing"], "P5", "Road Traffic Management & Passenger Information Systems")
    pages(["Carbon dioxide recorder"], "P5", "Sensors and Detectors")
    pages(["MirrorLink", "SmartDeviceLink"], "P6", "Embedded Control and Telematics")
    pages(["Aura Interactor", "Haptic suit", "Haptic technology", "Telehaptics", "Tixel"], "P7", "AR/VR & Immersive Technologies")
    pages(["Skinput", "Finger tracking", "Wired glove", "Gesture recognition"], "P7", "Input Devices & Interaction Techniques")
    pages(["Backscatter X-ray", "Millimeter wave scanner"], "P8", "Imaging Sensors and Camera Systems")
    pages(["Bioelectrical impedance analysis", "Impedance cardiography", "Focused impedance measurement"], "P9", "Neural Interfaces and Biosignal Processing")
    pages(CHIPS, "P10", "Biosensors")
    pages(["Difference of Gaussians"], "P11", "Computer Vision & Image Analysis")
    pages(["Translate Toolkit"], "P11", "NLP, Generative AI & Language Models")
    pages(["AirSim"], "P11", "Physics Simulation Engines")
    pages(["Failure detector", "Snapshot algorithm", "Shared snapshot objects"], "P12", "Grid & Distributed Computing Infrastructure")
    pages(["Fredkin gate"], "P13", "Quantum computation theory")
    pages(["Quantum digital signature"], "P13", "Quantum Cryptographic Primitives")
    pages(["Topological quantum computer", "Electron-on-helium qubit"], "P13", "Spin Qubits")
    pages(["Integrated circuit", "Hybrid integrated circuit", "Linear integrated circuit", "Mixed-signal integrated circuit"], "P14",
          "IC Design, Packaging & Testing")
    pages(["AI data center"], "P14", "Modular Data Centers")
    pages(["Blu-ray"], "P14", "Computer Hardware, Peripherals & Storage")
    pages(["Stepping switch", "Crossbar switch", "Distribution frame", "Main distribution frame", "Automated main distribution frame"], "P15",
          "Telephony, VoIP & Call Management")
    pages(["Radio"], "P16", "Superheterodyne Reception")
    pages(["Distributed source coding"], "P16", "Data Compression & Codecs")
    drop_pages(list(REMOVE_PAGES), "R1")

    # ---------------- splits and new technologies (S)
    new_node("Avionics & Inertial Navigation", "Navigation & Positioning", T)
    pages(AVIONICS + ["Aircraft engine controls", "FADEC"], "S2", "Avionics & Inertial Navigation")
    pages(TO_GNSS, "S2", "GNSS, Satellite Navigation & Positioning")
    rename("Autonomous Navigation, Inertial Systems & Robot Localization", "Robot Localization & Mapping", "S2")
    for n in ("Industrial Control Systems & Fieldbuses", "Building Automation & Smart Home", "Smart Grid & Utility Automation"):
        new_node(n, "Industrial & Building Automation", RM)
    pages(BUILDING + ["Smart speaker", "Smart doorbell", "Smart refrigerator"], "S4", "Building Automation & Smart Home")
    pages(GRID, "S4", "Smart Grid & Utility Automation")
    merge("Industrial and Building Automation", "Industrial Control Systems & Fieldbuses", "S4")
    new_node("Wearable Devices", "Wireless & Connectivity", RN)
    pages(WEARABLE, "S5", "Wearable Devices")
    rename("Embedded Control and Telematics", "Vehicle Electronics & Telematics", "S5")
    rename("Indoor Positioning & Smart Connected Devices", "Indoor Positioning", "S5")
    new_node("Blockchain & Distributed Ledgers", "Cryptography & Cybersecurity", RR)
    pages(BLOCKCHAIN, "S8", "Blockchain & Distributed Ledgers")

    # ---------------- root and L1 placement (S)
    for n, l1 in (("Driver Assistance Systems", "Road Vehicle Systems"), ("Vehicle Electronics & Telematics", "Road Vehicle Systems"),
                  ("Road Traffic Management & Passenger Information Systems", "Traffic & Rail Management"),
                  ("Railway Signalling & Train Control Systems", "Traffic & Rail Management"),
                  ("GNSS, Satellite Navigation & Positioning", "Navigation & Positioning")):
        node_to(n, l1, T, "S1")
    rename("Service & Domestic Robotics", "Service Robotics", "S3")
    for n in ("Robot Manipulators and Arm Systems", "Swarm and Legged Robotics", "Service Robotics", "Medical & Laboratory Robotics",
              "Powered Exoskeletons", "Autonomous Farm Equipment", "Robot Localization & Mapping", "Actuators and Motion Components",
              "Tactile Sensing", "Autonomous Underwater Vehicles", "Remotely Operated Underwater Vehicles"):
        node_to(n, "Robots", RM, "S3")
    for n in ("Sensors and Detectors", "Position and Proximity Sensing", "Biosensors"):
        node_to(n, "Sensors & Detection", RM, "S3")
    for n in ("Fire and Safety Detection Systems", "Industrial Safety Interlocks", "Sensor Communication Protocols"):
        node_to(n, "Industrial & Building Automation", RM, "S3")
    for n in ("Magnetic NDT Methods", "Ultrasonic Testing"):
        node_to(n, "Non-Destructive Testing", RM, "S3")
    node_to("System Security & Malware Defense", "Cryptography & Cybersecurity", RR, "S6")
    node_to("Web Services Security", "Cryptography & Cybersecurity", RR, "S6")
    for n in ("Authentication & Access Control", "Encryption & Cryptographic Primitives", "PKI, Certificates & Secure Protocols",
              "Random Number Generation"):
        node_to(n, "Cryptography & Cybersecurity", RR, "S6")
    merge("Access Control Systems", "Authentication & Access Control", "S7")
    node_to("Imaging Sensors and Camera Systems", "Imaging & Display", "Imaging & Display Technology", "S9")
    for n in ("Audio Transducers & Equipment", "Digital Signal Processing", "Neural Interfaces and Biosignal Processing"):
        node_to(n, "Audio & Signal Processing", "Signal Processing & Communications", "S10")
    rename("Microarrays & Bioinformatics Sequence Analysis", "Bioinformatics & Sequence Analysis", "S11")
    for n in ("Computational Fluid Dynamics", "Physics Simulation Engines", "Monte Carlo Particle Transport", "Collision Detection",
              "Deformable Body Simulation", "Reactive Flow Modeling", "Numerical Weather Prediction", "Bioinformatics & Sequence Analysis"):
        node_to(n, "Scientific Computing & Simulation", RS, "S11")
    for n in ("Antenna Design & RF Components", "Radar Systems & Signal Processing"):
        node_to(n, "Radar & RF Engineering", "Signal Processing & Communications", "S11")
    rename("Digital Production Printing", "Printing & Copying Technology", "S12")
    rename("Modular Data Centers", "Data Center Infrastructure", "S13")
    for n in ("Printing & Copying Technology", "Stencil & Thermal Duplication", "Thermal Label Printing", "Data Center Infrastructure",
              "Computer Hardware, Peripherals & Storage", "Processor & Memory Architecture", "Thermal Management", "Unconventional Computing"):
        node_to(n, "Computer Hardware & Peripherals", RCH, "S12" if "Print" in n or "Label" in n or "Stencil" in n else "S13")
    for n in ("Android Device Management", "Boot, Firmware & File Systems", "OS Kernel, Process & Memory Management",
              "Storage, Virtualization & File Systems"):
        node_to(n, "System Software & Virtualization", RCH, "S13")
    node_to("Messaging Middleware", "Web Standards & Middleware", RS, "S13")
    node_to("VSAM Data Access", "Data Management & Processing", RS, "S13")
    node_to("OMG Metamodeling Standards", "Programming Languages & Software Engineering", RS, "S13")
    for n in ("Software Calculators", "Computer Reservation Systems", "Clinical Health IT Systems", "Medical Imaging Informatics"):
        node_to(n, "Application & Information Systems", RS, "S13")
    for n in ("Graphical User Interfaces & UX Design", "Input Devices & Interaction Techniques", "Data Sonification"):
        node_to(n, "Human-Computer Interaction", RS, "S13")
    for n in ("Data Capture and Validation", "Data Integration", "Privacy-Preserving Data Techniques"):
        node_to(n, "Data Governance & Integration", RS, "S13")
    for n in ("Additive Manufacturing & 3D Printing", "CAD, Solid Modeling & BIM"):
        node_to(n, "Digital Fabrication", RM, "S3")
    # non-digital technologies removed (S14), quantum memory placement and renames (S16, S17)
    for n in REMOVED_NODES:
        drop_node(n, "S14")
    node_to("Quantum State Storage", "Quantum Computing Hardware", RQ, "S16")
    rename("Spin Qubits", "Spin, Ion, Atom & Photonic Qubits", "S16")
    rename("Display Panels & Lighting Technology", "Display Panels", "S17")
    rename("Superheterodyne Reception", "Radio Receivers", "S17")

    # ---------------- duplicates (D1): identical description embeddings, keep the article page
    rows = d.page_title.map(row_of)
    live = d[rows.notna()].copy(); live["row"] = rows[rows.notna()].astype(int)
    live["key"] = [hash(np.round(emb[r] / np.linalg.norm(emb[r]), 5).tobytes()) for r in live.row]
    norm = lambda s: re.sub(r"\s*\(.*?\)", "", s).strip().lower()
    dd = []
    for k, g in live.groupby("key"):
        if len(g) < 2:
            continue
        txt = desc[g.row.iloc[0]].lower()[:300]
        sc = [(txt.find(norm(t)) if norm(t) in txt else 10 ** 6, len(t), t) for t in g.page_title]
        keep = min(sc)[2]
        for t, l2 in zip(g.page_title, g.L2):
            dd.append({"group": k, "page": t, "L2": l2, "keep": t == keep, "kept_page": keep})
    DD = pd.DataFrame(dd)
    gone = DD[~DD.keep].page.tolist()
    m = d.page_title.isin(gone); tag(m, "D1"); d.drop(d.index[m], inplace=True)

    d["L1"] = d.L2.map(lambda n: parent[n][0]); d["root"] = d.L2.map(lambda n: parent[n][1])
    empty = [n for n in parent if n not in set(d.L2)]
    assert not empty, f"technologies left without pages: {empty}"
    return d.reset_index(drop=True), log, DD


def final_levels():
    """Every page row of the build_final.py universe relabelled to the final taxonomy: pages removed by R1/S14 and the
    dropped duplicates of D1 carry BF.OUT_NODE. -> (titles, levels {L2, L1, root}, duplicate pages taken out, log)."""
    titles, lv = production_levels()
    d, log, DD = apply(base_pages())
    place = d.set_index("page_title")[["L2", "L1", "root"]]
    gone_np = {p for p, it in log.items() if ("R1" in it or "S14" in it)}
    dedup = set(DD[~DD.keep].page)
    lv = {k: v.copy() for k, v in lv.items()}
    for i, t in enumerate(titles):
        if t in place.index:
            lv["L2"][i], lv["L1"][i], lv["root"][i] = place.loc[t].tolist()
        elif lv["L2"][i] != BF.OUT_NODE:
            assert t in gone_np or t in dedup, f"live page not placed: {t}"
            lv["L2"][i] = lv["L1"][i] = lv["root"][i] = BF.OUT_NODE
    return titles, lv, dedup, log


if __name__ == "__main__":
    titles, lv, dedup, log = final_levels()
    live = lv["L2"] != BF.OUT_NODE
    print(f"final taxonomy: pages {int(live.sum()):,} | roots {len(set(lv['root'][live]))} | L1 {len(set(lv['L1'][live]))} | "
          f"L2 {len(set(lv['L2'][live]))} | duplicate pages out {len(dedup)}")
    ref = pd.read_csv(FINAL / "levels.csv")
    mine = pd.DataFrame({"page_row": np.arange(len(titles)), "page_title": titles, **lv})
    same = len(ref) == len(mine) and all((ref[c].astype(str).to_numpy() == mine[c].astype(str).to_numpy()).all()
                                         for c in ("page_row", "page_title", "L2", "L1", "root"))
    print(f"check against {FINAL.name}/levels.csv ({len(ref):,} rows): {'IDENTICAL' if same else 'DIFFERENT'}")
    assert same
