"""Off-theme (entity,node) cleaning blacklists + apply -- ported VERBATIM from the recovered builders
(build_changes.py / build_green_changes.py). Zeros off-theme cells in an exposure dataframe; matches on
(entity-code, tech_node), so on a YEAR-VARYING frame it zeros the same cells in every year. NEVER drops a
node (zero exposure is exposure). Digital also needs page-handling (see page_handling_plan.csv) applied at
build time; this module is only the surgical off-theme zeroing stage.
"""
import pandas as pd
import core as C

# ---- digital off-theme blacklist (exact, from build_changes.py) ----
OCC_OFF = {
 "Sorting Algorithms": ["refuse","filing","copying","mail","motor vehicle mech","coding","sewing"],
 "Demodulation & Detection": ["coding","translat","fibre","spinning","sales demonstr","wood treat","metal","well driller","borer","mineral","stone","pre-press"],
 "Optical Signaling": ["photographic products","railway","brake","motorcycle","transport conduct"],
 "General Error Correction": ["coding","proof-read","data entry","typist"],
 "Processor & Memory Architecture": ["chemical","photographic","metal","petroleum","cement"],
 "AR/VR & Immersive Technologies": ["optometr","optician","medical","dental","physiother"],
 "Symbolic AI, Knowledge & Agent Systems": ["advertis","metal","business serv","buyer"],
 "Unconventional Computing": ["chemist","biolog","pathology","chemical","life scien"],
 "Collision Detection": ["dancer","choreograph","air traffic"],
 "Remote Procedure Calls & Distributed Objects": ["farm","forestry","lifting truck"],
 "Indoor Positioning & Smart Connected Devices": ["medical","veterinar","dentist","pathology"],
 "Software Calculators": ["medical","physiother","paramedic","pharmac","health","nursing","dental","traditional"],
 "Physics Simulation Engines": ["agricultur","farming","forestry","fisher","field crop"],
 "Neural Networks & Deep Learning": ["librarian","teaching","education","proof-read"],
 "Evolutionary & Swarm Optimization": ["buyer","sales","stall","market","proof-read"],
 "Pattern Matching & String Algorithms": ["garment","pattern-mak","sewing","forestry","wood","mobile farm"],
}
IND_OFF = {
 "Deformable Body Simulation": ["textile","weaving","garment","felt","fashion","luggage","knitted","footwear","babies","synthetic","artificial","doll","toy","fur","jewel","sofa","fibre"],
 "Powered Exoskeletons": ["footwear","garment","sporting","babies","sofa","weighing","freight","laborator","communicat","watch","clock"],
 "EM Wave-Material Interaction": ["seamless","tube","machine tool","boring","turning","milling","drawn steel","forging","pressing","stamping","casting","metal"],
 "Evolutionary & Swarm Optimization": ["retail","post-harvest","vegetat","fish","crustacean","pet","preservation","planting","agricultur"],
 "Privacy-Preserving Data Techniques": ["publishing","health","catering","binding","laborator",
                                        "collection of solid and non-solid hazardous waste"],
 "Service & Domestic Robotics": ["astrolog","spiritual","tractor","weighing","protective safety","storage","verbatim"],
 "System Logging": ["roundwood","forestry"],
 "Unconventional Computing": ["medicinal","synthetic rubber","photographic","optical instrum","rubber"],
 "Physics Simulation Engines": ["crop","animal","tractor","vegeta","growing","agricultur","forestr","wholesale"],
 "Software Calculators": ["dental","medical","watch"],
 "Sorting Algorithms": ["binding","textile","finishing"],
 # word collisions found in the nine-root audit of the cloud build, 2009-2023 top-3 codes (user decision 2026-10-01);
 # each substring is matched against the code's description and hits exactly the one NACE group named in the comment
 "Random Number Generation": ["vegetative planting materials"],                                   # 01.3 ("seed")
 "PKI, Certificates & Secure Protocols": ["pool securities"],                                     # 64.3 ("trust")
 "Email Systems": ["postal services operating under a universal", "letter-post"],                 # 53.1, 53.2 ("mail")
 "Storage, Virtualization & File Systems": ["storage and warehouse facilities"],                  # 52.1 ("storage")
 "Database Architecture & Query Systems": ["storage and warehouse facilities"],                   # 52.1 ("warehouse")
 "Tree Data Structures": ["storage and warehouse facilities"],                                    # 52.1 ("storage")
 "Data Capture and Validation": ["collection of solid and non-solid hazardous waste"],            # 38.1 ("collection")
 "Semantic Web & Linked Data": ["water collection, treatment and distribution"],                  # 36.0 ("collection")
 "Software Design Patterns": ["via pipelines"],                                                   # 49.5 ("pipeline")
 "Compiler Optimization & Code Generation": ["complete television channel programme"],            # 60.2 ("programming")
 "Driver Assistance Systems": ["aimed at the elderly and disabled in their homes"],               # 88.1 ("assistance")
 "Monte Carlo Particle Transport": ["other passenger road transport"],                            # 49.3 ("transport")
 "Quantum Error Correction": ["repair and maintenance of communications equipment"],              # 95.1 ("correction")
 "Burst Error Correction": ["repair and maintenance of communications equipment"],                # 95.1 ("correction")
 "Erasure Coding": ["repair and maintenance of communications equipment",
                    "repair and maintenance of equipment not covered"],                           # 95.1, 33.1 ("erasure")
 "Soft-Decision Decoding": ["translation and interpretation activities"],                         # 74.3 ("decoding")
 "General Error Correction": ["catalogues, photos, engravings"],                                  # 58.1 ("correction")
}
# ---- quantum collision cells (user decision 2026-10-07): exact (node, entity code) cells of the Quantum Computing nodes
# zeroed as off-theme. (a) cells whose reviewed links the LLM link review judged mostly off-theme (zero > keep in
# notes/validation/finetune/link_review_top50.csv: the review zeroed those links, the unreviewed tail kept the cell);
# (b) unreviewed cells that are plain word collisions (light, clock, image, photo, optic, pattern, installation).
# Tied or keep-majority cells and ambiguous unreviewed cells are kept (notes/validation/quantum_relative_exposure_2026_10.md).
OCC_OFF_CODES = {
 "Quantum computation theory": ["3133", "2152", "2153", "8111",                     # (a) 12/12, 10/11, 5/7, 4/4 zero
                                "3254", "7532", "2267", "3431", "8132"],            # (b) opticians, garment, optometrists, photo
 "Quantum optimization": ["2152", "2631", "2120"],                                  # (a) 5/9, 7/10, 1/1
 "Superconducting Qubits": ["2153", "3133", "7532"],                                # (a) 2/3, 1/1, 1/1
 "Quantum Error Correction": ["7311", "3511", "8111"],                              # (a) 24/28, 2/2, 2/2
 "Quantum Network Protocols": ["3133", "8111"],                                     # (a) 5/5, 3/3
 "Quantum simulation": ["3133"],                                                    # (a) 1/1
 "Spin Qubits": ["3133"],                                                           # (a) 5/5
 "Quantum image processing": ["3431"],                                              # (a) 6/7
}
IND_OFF_CODES = {
 "Superconducting Qubits": ["27.3",                                                 # (a) 15/18 ("circuit")
                            "43.2"],                                                # (b) installation
 "Quantum computation theory": ["27.3", "60.2",                                     # (a) 5/5, 2/2
                                "27.4"],                                            # (b) light bulbs
 "Quantum Cryptographic Primitives": ["27.3",                                       # (a) 1/1
                                      "58.1", "26.5"],                              # (b) catalogues/photos, clocks
 "Quantum image processing": ["18.1"],                                              # (b) printing services
}
# green off-theme: non-motorized vehicles at the 4.x transport nodes (from build_green_changes.py)
GREEN_SUBS = {"occ": ["animal-drawn","hand and pedal"], "ind": ["hand-propelled","animal-drawn"]}


def _names(side):
    """entity-code -> lowercased title (part before '. '), exactly as the recovered builders."""
    if side == "occ":
        df = pd.read_csv(C.PB/"embeddings"/"isco_titletask.csv", dtype={"occ_code": str})
        return {str(c): str(s).split(". ")[0].lower() for c, s in zip(df.occ_code, df.sentence_text)}, "occ_code"
    df = pd.read_csv(C.PB/"embeddings"/"nace_titlesent_rev2.csv", dtype={"nace_code": str})
    return {str(c): str(s).split(". ")[0].lower() for c, s in zip(df.nace_code, df.sentence_text)}, "nace_code"


def apply_digital(df, side):
    """Zero digital off-theme (entity,node) cells in `df` (pooled or year-varying). Returns (df, n_cells_zeroed)."""
    nm, code = _names(side); OFF = OCC_OFF if side == "occ" else IND_OFF
    df = df.copy(); nmcol = df[code].astype(str).map(lambda c: nm.get(c, c))
    z = 0
    for node, subs in OFF.items():
        m = (df.tech_node == node) & nmcol.apply(lambda n: any(s in n for s in subs))
        df.loc[m, "value"] = 0.0; z += int(m.sum())
    for node, codes in (OCC_OFF_CODES if side == "occ" else IND_OFF_CODES).items():     # exact cells
        m = (df.tech_node == node) & df[code].astype(str).isin(codes)
        df.loc[m, "value"] = 0.0; z += int(m.sum())
    return df, z


def reclean_build(src, out):
    """Re-apply the digital off-theme blacklist to an existing yearly build `src` (folder with exposure/*.parquet) and
    write it to `out`. Only matched cells change (to 0, every measure and year); every other row is asserted identical.
        python -u cleaning.py <src build folder name> <out build folder name>     (both under core.TRI)"""
    import json
    src, out = C.TRI / src, C.TRI / out
    (out / "exposure").mkdir(parents=True, exist_ok=True)
    rep = {}
    for side, fn in (("occ", "occupation"), ("ind", "industry")):
        df = pd.read_parquet(src / "exposure" / f"{fn}.parquet")
        new, _ = apply_digital(df, side)
        changed = (new.value != df.value)
        assert (new.value[~changed] == df.value[~changed]).all() and (new.drop(columns="value") == df.drop(columns="value")).all().all()
        rep[fn] = {"cells_newly_zeroed": int(changed.sum()),
                   "by_node_level": new[changed].groupby(["tech_node", "level"]).size().to_dict().__repr__()}
        new.to_parquet(out / "exposure" / f"{fn}.parquet", index=False)
        print(f"  {fn}: {int(changed.sum()):,} rows newly zeroed", flush=True)
        if changed.any():
            print(new[changed].groupby(["level", "tech_node"]).size().to_string(), flush=True)
    (out / "README.json").write_text(json.dumps({"source": str(src), "step": "cleaning.reclean_build (off-theme blacklist re-applied)",
                                                 "report": rep, "script": "p3_exposure/threshold_experiment/cleaning.py"}, indent=1))


def apply_green(df, side, transport_nodes):
    """Zero green off-theme transport cells in `df`. transport_nodes = node codes whose label starts '4.'."""
    nm, code = _names(side); subs = GREEN_SUBS[side]
    df = df.copy(); nmcol = df[code].astype(str).map(lambda c: nm.get(c, c))
    m = df.tech_node.isin(transport_nodes) & nmcol.apply(lambda n: any(s in n for s in subs))
    z = int(m.sum()); df.loc[m, "value"] = 0.0
    return df, z


if __name__ == "__main__":
    import sys
    reclean_build(sys.argv[1], sys.argv[2])
