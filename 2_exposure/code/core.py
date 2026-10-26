"""Shared engine for the threshold experiment: paths, loaders, relevance-keep, aggregation, and the
CONFIG registry. Every build/coverage/audit script used to re-implement these; now they import from here.

Conventions (both families):
  Leg A (task<->patent) carries the value. Leg B (page<->patent) = membership (tag floor tb).
  Leg C (task<->page, PatentSBERTa) = gate at tc. Pages + C are ALWAYS PatentSBERTa.
  triagree: A = MPNET, value = max(0, A - per-task baseline), no fixed ta, tb=0.50 (ge0p5 tags).
  pb:       A = PatentSBERTa, value = A (plain) or max(0,A-baseline) (dem), fixed ta, tb in {.40,.45} (ge0p4_top20 cap20).
"""
import os, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root -> `shared`
from shared.patent_cache import load_embeddings
warnings.filterwarnings("ignore", category=RuntimeWarning)

DROPBOX = Path(r"C:/Users/sidha/Dropbox/Work/UNU-MERIT/ST4TE/Task2_1/data")
PB  = DROPBOX/"exposure_scores"/"digital_patentsberta"
MP  = DROPBOX/"exposure_scores"/"digital_mpnet"
TRI = DROPBOX/"exposure_scores"/"triagree"
PBAN = PB/"analysis"
COV = TRI/"coverage"
CH = 20000
PB_HF = "AI-Growth-Lab/PatentSBERTa"
unit = lambda x:(lambda a,n:a/np.where(n==0,1,n))(np.asarray(x,np.float32),
        np.linalg.norm(np.asarray(x,np.float32),axis=1,keepdims=True))

# ---- per-family A (task<->patent) matrices, relevance components, membership tag files ----
A_FILES   = {"occ":"occ_titletask_x_patent.npy", "ind":"nace_titlesent_x_patent.npy"}
PTASK     = {"occ":"isco_titletask.npy",          "ind":"nace_titlesent.npy"}
ENT_CSV   = {"occ":("isco_titletask.csv","occ_code"), "ind":("nace_titlesent.csv","nace_code")}
DESC_FILE = {"occ":"occ_description_x_patent.npy","ind":"industry_description_x_patent.npy"}
TASK_FILE = {"occ":"occ_taskonly_x_patent.npy",  "ind":"nace_sentonly_x_patent.npy"}
DESC_CSV  = {"occ":"isco_description.csv","ind":"nace_description.csv"}
TASK_CSV  = {"occ":"isco_taskonly.csv",  "ind":"nace_sentonly.csv"}
TAGFILE   = {"triagree":"tags_patentsberta_ge0p5.parquet", "pb":"tags_patentsberta_ge0p4_top20.parquet"}
PT_CIT = DROPBOX/"taxonomy"/"wiki"/"pipeline"/"07_tag_patents"/"digital_v3"/"patent_technologies.parquet"

NACE_REV2 = True    # default: industry side uses NACE Rev 2 *_rev2 nace files (no 2.1 intermediation magnets)
def _indf(side, name):
    return (name.replace(".npy", "_rev2.npy").replace(".csv", "_rev2.csv")) if (NACE_REV2 and side == "ind") else name

def a_dir(a_embed):
    if a_embed=="mpnet_demean": return MP/"similarity_demean"   # demeaned-embedding A-leg (anisotropy removed)
    return (MP if a_embed=="mpnet" else PB)/"similarity"

# ---------------- shared loaders ----------------
def load_shared(tag_file, with_cit=False):
    """ppage (PB), level arrays, membership tags, patent ids.
    (Callers build.py/coverage.py compute their own tb-filtered argmax from `tags`.)"""
    ppage = unit(np.load(PBAN/"wiki_page_desc_patentsberta.npy"))
    page_meta = pd.read_parquet(PBAN/"page_meta_patentsberta.parquet")
    levels = {lv: page_meta[lv].to_numpy() for lv in ("L2","L1","root")}
    tags = pd.read_parquet(PBAN/tag_file, columns=["patent_col","page_row","sim"])
    _, pids = load_embeddings(mode="digital", view="abstract", embed_model=PB_HF)
    nPat = len(np.asarray(pids))
    out = dict(ppage=ppage, levels=levels, tags=tags, pids=pids, nPat=nPat)
    if with_cit:
        cdf = pd.read_parquet(PT_CIT, columns=["appln_id","forward_citation_global","appln_filing_year"]).drop_duplicates("appln_id")
        cl = dict(zip(cdf.appln_id.astype(int), cdf.forward_citation_global.astype(float).fillna(0.0)))
        yl = dict(zip(cdf.appln_id.astype(int), cdf.appln_filing_year.astype(float)))
        pid_arr = np.asarray(pids)
        out["cit"]  = np.array([cl.get(int(a),0.0) for a in pid_arr], np.float32)   # global cumulative (pooled builds)
        out["year"] = np.array([yl.get(int(a),np.nan) for a in pid_arr], np.float32)  # appln_filing_year per patent col
        # within-year citation weight (Hall-Jaffe-Trajtenberg cohort norm): cit / mean(cit | same filing year)
        cit = out["cit"].astype(np.float64); yr = out["year"]; rel = np.zeros_like(cit)
        for y in np.unique(yr[np.isfinite(yr)]):
            m = yr == y; mu = cit[m].mean()
            if mu > 0: rel[m] = cit[m] / mu
        out["cit_rel"] = rel.astype(np.float32)
    return out

def entity_meta(family, side):
    csv, codecol = ENT_CSV[side]
    meta = pd.read_csv(PB/"embeddings"/_indf(side, csv), dtype={codecol:str})
    codes = meta[codecol].astype(str).to_numpy(); ents = sorted(set(codes)); ec = {c:i for i,c in enumerate(ents)}
    s2e = np.array([ec[c] for c in codes]); return codes, ents, ec, s2e, codecol

def load_A(a_embed, side):
    M = np.load(a_dir(a_embed)/_indf(side, A_FILES[side])); return M, M.mean(1).astype(np.float32)

def load_ptask(side):  return unit(np.load(PB/"embeddings"/_indf(side, PTASK[side])))

def retag_pb_membership(floor=0.40, topn=20, out=None):
    """(Re)build the pb page<->patent membership tags: each patent's top-N strongest pages >= floor
    (PatentSBERTa). Writes analysis/tags_patentsberta_ge0p<floor>_top<topn>.parquet. Was retag_pb_ge0p4.py."""
    page = unit(np.load(PBAN/"wiki_page_desc_patentsberta.npy"))
    pemb, _ = load_embeddings(mode="digital", view="abstract", embed_model=PB_HF); pat = unit(pemb); nPat = pat.shape[0]
    rows=[]; cols=[]; sims=[]
    for i in range(0, nPat, CH):
        S = page @ pat[i:i+CH].T
        top = np.argpartition(-S, topn-1, axis=0)[:topn]; sv = np.take_along_axis(S, top, axis=0); m = sv >= floor
        nn, cc = np.nonzero(m)
        if len(nn): rows.append(top[nn,cc].astype(np.int32)); cols.append((cc+i).astype(np.int32)); sims.append(sv[nn,cc].astype(np.float32))
    df = pd.DataFrame({"patent_col":np.concatenate(cols),"page_row":np.concatenate(rows),"sim":np.concatenate(sims)})
    out = out or PBAN/f"tags_patentsberta_ge0p{int(round(floor*100))}_top{topn}.parquet"
    df.to_parquet(out, index=False); return len(df)

# ---------------- relevance keeps ----------------
def keep_single(M, baseline, s2e, nE, K=10):
    """top-K entities per patent by demeaned A. -> bool (nE, nPat)."""
    keep = np.zeros((nE, M.shape[1]), bool)
    for i in range(0, M.shape[1], CH):
        dem = M[:,i:i+CH]-baseline[:,None]
        om = np.full((nE,dem.shape[1]),-1e9,np.float32); np.maximum.at(om, s2e, dem)
        idx = np.argpartition(-om, K-1, axis=0)[:K]; np.put_along_axis(keep[:,i:i+CH], idx, True, axis=0)
        del dem, om, idx
    return keep

def keep_twocomp(side, ents, ec, nE, Ks=(10,5)):
    """two-component: entity in patent top-K by BOTH description-sim AND task-sim (demeaned, MPNET). -> {K: bool(nE,nPat)}."""
    descA = np.load(MP/"similarity"/DESC_FILE[side]); dm = pd.read_csv(MP/"embeddings"/DESC_CSV[side], dtype=str)
    dcol = "occ_code" if side=="occ" else "nace_code"
    db = np.zeros((nE, descA.shape[1]), np.float32); db[np.array([ec[str(c)] for c in dm[dcol]])] = descA; del descA
    descbase = db.mean(1).astype(np.float32)
    taskA = np.load(MP/"similarity"/TASK_FILE[side]); tm = pd.read_csv(MP/"embeddings"/TASK_CSV[side], dtype=str)
    s2e_t = np.array([ec[str(c)] for c in tm[dcol]]); taskbase = taskA.mean(1).astype(np.float32)
    keeps = {K: np.zeros((nE, taskA.shape[1]), bool) for K in Ks}
    for i in range(0, taskA.shape[1], CH):
        dd = db[:,i:i+CH]-descbase[:,None]; td = taskA[:,i:i+CH]-taskbase[:,None]
        tmx = np.full((nE,td.shape[1]),-1e9,np.float32); np.maximum.at(tmx, s2e_t, td)
        for K in Ks:
            di = np.argpartition(-dd, K-1, axis=0)[:K]; ti = np.argpartition(-tmx, K-1, axis=0)[:K]
            a = np.zeros((nE,dd.shape[1]),bool); np.put_along_axis(a, di, True, 0)
            b = np.zeros((nE,dd.shape[1]),bool); np.put_along_axis(b, ti, True, 0)
            keeps[K][:,i:i+CH] = a & b
        del dd, td, tmx
    del db, taskA
    return keeps

def relevance_keep(config, side, M, baseline, ents, ec, s2e, nE):
    """Return the (nE,nPat) keep mask for a config's relevance scheme."""
    rel = config["relevance"]
    if rel == "single":          return keep_single(M, baseline, s2e, nE, K=10)
    if rel == "2comp-K10":       return keep_twocomp(side, ents, ec, nE, Ks=(10,))[10]
    if rel == "2comp-K5":        return keep_twocomp(side, ents, ec, nE, Ks=(5,))[5]
    raise ValueError(rel)

# ---------------- per-entity aggregation (mean over the entity's tasks) ----------------
def pad_index(s2e, nE):
    counts = np.bincount(s2e, minlength=nE); maxT = int(counts.max())
    pad = np.zeros((nE,maxT),np.int64); pm = np.zeros((nE,maxT),bool); pos = np.zeros(nE,np.int64)
    for si in range(len(s2e)):
        e = s2e[si]; pad[e,pos[e]] = si; pm[e,pos[e]] = True; pos[e] += 1
    return pad, pm, counts
def agg_mean(r, pad, pm, counts):
    return np.where(pm, r[pad], 0.0).sum(1)/np.where(counts>0, counts, 1)

# ---------------- node membership subset (best page per patent within a node) ----------------
def node_members(tags_sub, G, argmax):
    sub = tags_sub[tags_sub.page_row.isin(G)].sort_values("sim").drop_duplicates("patent_col", keep="last")
    if len(sub)==0: return None
    Pcols = sub.patent_col.to_numpy(); pr = sub.page_row.to_numpy()
    gmap = {int(g):i for i,g in enumerate(G)}; gloc = np.array([gmap[int(g)] for g in pr], np.int64)
    is1 = (argmax[Pcols]==pr)
    return Pcols, gloc, is1

# ======================= CONFIG REGISTRY (single source of truth) =======================
def _c(key,family,dir_,measure,mask_file,mask_key,ta,tb,tc,assign,relevance,label,a_embed=None):
    if a_embed is None: a_embed = "mpnet" if family=="triagree" else "pb"
    return dict(key=key,family=family,dir=dir_,measure=measure,mask_file=mask_file,mask_key=mask_key,
                ta=ta,tb=tb,tc=tc,assign=assign,relevance=relevance,label=label,a_embed=a_embed)
_TRI="triagree_mpnet_mean"; _PBM="pb_plain_mean"
_UT=str(COV/"used_masks"/"used_triagree_occ.npz"); _UP=str(COV/"used_pbthr_occ.npz")
CONFIGS=[
 _c("tc25_m2m","triagree","mpnet_winning",_TRI,_UT,"tc25_m2m","adaptive",0.50,0.25,"m2m","single","tc.25 / m2m / single"),
 _c("tc25_1to1","triagree","mpnet_tc25_1to1",_TRI,_UT,"tc25_1to1","adaptive",0.50,0.25,"1to1","single","tc.25 / 1to1 / single"),
 _c("tc30_m2m","triagree","mpnet_tc30_m2m",_TRI,_UT,"tc30_m2m","adaptive",0.50,0.30,"m2m","single","tc.30 / m2m / single"),
 _c("tc30_1to1","triagree","mpnet_tc30_1to1",_TRI,_UT,"tc30_1to1","adaptive",0.50,0.30,"1to1","single","tc.30 / 1to1 / single"),
 _c("tc35_m2m","triagree","mpnet_tc35_m2m",_TRI,_UT,"tc35_m2m","adaptive",0.50,0.35,"m2m","single","tc.35 / m2m / single"),
 _c("tc35_1to1","triagree","mpnet_tc35_1to1",_TRI,_UT,"tc35_1to1","adaptive",0.50,0.35,"1to1","single","tc.35 / 1to1 / single"),
 _c("tc30_m2m_K10","triagree","mpnet_tc30_m2m_hardK10",_TRI,_UT,"tc30_m2m_K10","adaptive",0.50,0.30,"m2m","2comp-K10","tc.30 / m2m / 2comp-K10"),
 _c("tc30_1to1_K10","triagree","mpnet_tc30_1to1_hardK10",_TRI,_UT,"tc30_1to1_K10","adaptive",0.50,0.30,"1to1","2comp-K10","tc.30 / 1to1 / 2comp-K10"),
 _c("tc35_m2m_K10","triagree","mpnet_tc35_m2m_hardK10",_TRI,_UT,"tc35_m2m_K10","adaptive",0.50,0.35,"m2m","2comp-K10","tc.35 / m2m / 2comp-K10"),
 _c("tc35_1to1_K10","triagree","mpnet_tc35_1to1_hardK10",_TRI,_UT,"tc35_1to1_K10","adaptive",0.50,0.35,"1to1","2comp-K10","tc.35 / 1to1 / 2comp-K10"),
 _c("tc30_m2m_K5","triagree","mpnet_tc30_m2m_hardK5",_TRI,_UT,"tc30_m2m_K5","adaptive",0.50,0.30,"m2m","2comp-K5","tc.30 / m2m / 2comp-K5"),
 _c("tc30_1to1_K5","triagree","mpnet_tc30_1to1_hardK5",_TRI,_UT,"tc30_1to1_K5","adaptive",0.50,0.30,"1to1","2comp-K5","tc.30 / 1to1 / 2comp-K5"),
 _c("tc35_m2m_K5","triagree","mpnet_tc35_m2m_hardK5",_TRI,_UT,"tc35_m2m_K5","adaptive",0.50,0.35,"m2m","2comp-K5","tc.35 / m2m / 2comp-K5"),
 _c("tc35_1to1_K5","triagree","mpnet_tc35_1to1_hardK5",_TRI,_UT,"tc35_1to1_K5","adaptive",0.50,0.35,"1to1","2comp-K5","tc.35 / 1to1 / 2comp-K5"),
 _c("pb_t40_m2m","pb","pb_t40_m2m",_PBM,_UP,"t40_m2m",0.40,0.40,0.40,"m2m","single","ta.40 / tb.40 / tc.40 / m2m"),
 _c("pb_t40_1to1","pb","pb_t40_1to1",_PBM,_UP,"t40_1to1",0.40,0.40,0.40,"1to1","single","ta.40 / tb.40 / tc.40 / 1to1"),
 _c("pb_t45_m2m","pb","pb_t45_m2m",_PBM,_UP,"t45_m2m",0.45,0.45,0.35,"m2m","single","ta.45 / tb.45 / tc.35 / m2m"),
 _c("pb_t45_1to1","pb","pb_t45_1to1",_PBM,_UP,"t45_1to1",0.45,0.45,0.35,"1to1","single","ta.45 / tb.45 / tc.35 / 1to1"),
 _c("pb_t50_m2m","pb","pb_t50_m2m",_PBM,_UP,"t50_m2m",0.50,0.40,0.35,"m2m","single","ta.50 / tb.40 / tc.35 / m2m"),
 _c("pb_t50_1to1","pb","pb_t50_1to1",_PBM,_UP,"t50_1to1",0.50,0.40,0.35,"1to1","single","ta.50 / tb.40 / tc.35 / 1to1"),
 _c("pb_t454545_1to1","pb","pb_t454545_1to1",_PBM,_UP,"t454545_1to1",0.45,0.45,0.45,"1to1","single","ta.45 / tb.45 / tc.45 / 1to1"),
 _c("pb_mpnet_plain_1to1","pb","pb_mpnet_plain_1to1",_PBM,_UP,"mpnet_plain_1to1",0.32,0.40,0.40,"1to1","single","mpnet plain / ta.32 tb.40 tc.40 / 1to1",a_embed="mpnet"),
 _c("pb_mpnet_plain_1to1_demean","pb","pb_mpnet_plain_1to1_demean",_PBM,_UP,"mpnet_plain_1to1_demean",0.2056,0.40,0.40,"1to1","single","mpnet DEMEAN / ta.206(p99) tb.40 tc.40 / 1to1",a_embed="mpnet_demean"),
 # finetuned = winner pb_mpnet_plain_1to1 + page-handling (move/drop off-theme pages, new Service&Domestic Robotics node) + abstract-node drop.
 # Built via the levels-patch rebuild (NOT build(); reads existing dir). mask reuses the winner's (coverage panel approximate).
 _c("pb_mpnet_finetuned","pb","pb_mpnet_finetuned",_PBM,_UP,"mpnet_finetuned",0.32,0.40,0.40,"1to1","single","mpnet FINETUNED (page-handling + abstract-drop) / ta.32 tb.40 tc.40 / 1to1",a_embed="mpnet"),
 # same finetuned parquet, demean measure -- so the finetuned can be viewed plain AND demean as separate selectable configs.
 _c("pb_mpnet_finetuned_demean","pb","pb_mpnet_finetuned","pb_dem_mean",_UP,"mpnet_finetuned",0.32,0.40,0.40,"1to1","single","mpnet FINETUNED DEMEAN / ta.32 tb.40 tc.40 / 1to1",a_embed="mpnet"),
 # finetuned + lever (a): finetuned levels + (task,node) gate-suppression on the 6 stable nodes. Same used-mask as finetuned -> TVD identical.
 _c("pb_mpnet_finetuned_leverA","pb","pb_mpnet_finetuned_leverA",_PBM,_UP,"mpnet_finetuned",0.32,0.40,0.40,"1to1","single","mpnet FINETUNED+leverA (task-node suppress) / ta.32 tb.40 tc.40 / 1to1",a_embed="mpnet"),
 _c("pb_mpnet_finetuned_leverA_demean","pb","pb_mpnet_finetuned_leverA","pb_dem_mean",_UP,"mpnet_finetuned",0.32,0.40,0.40,"1to1","single","mpnet FINETUNED+leverA DEMEAN / ta.32 tb.40 tc.40 / 1to1",a_embed="mpnet"),
 # finetuned + CHANGES: off-theme (entity,node) cells zeroed + 6 near-empty nodes dropped. Same used-mask -> TVD = finetuned. Both measures inherit.
 _c("pb_mpnet_finetuned_changes","pb","pb_mpnet_finetuned_changes",_PBM,_UP,"mpnet_finetuned",0.32,0.40,0.40,"1to1","single","mpnet FINETUNED+CHANGES (zero off-theme + drop near-empty) / ta.32 tb.40 tc.40 / 1to1",a_embed="mpnet"),
 _c("pb_mpnet_finetuned_changes_demean","pb","pb_mpnet_finetuned_changes","pb_dem_mean",_UP,"mpnet_finetuned",0.32,0.40,0.40,"1to1","single","mpnet FINETUNED+CHANGES DEMEAN / ta.32 tb.40 tc.40 / 1to1",a_embed="mpnet"),
 # 2012-2019 windowed (within-year citations) measures = the LFS/OJV analysis exposure. SAME tagging/mask as the
 # pooled PLAIN/CLEANED (representativeness = the full-sample one); exposure = arcsinh(sum of within-year-weighted
 # linear over 2012-2019), dir built by collapsing the *_yearly builds. analyze_config works on these unchanged.
 _c("pb_mpnet_plain_2012_2019","pb","pb_mpnet_plain_2012_2019",_PBM,_UP,"mpnet_plain_1to1",0.32,0.40,0.40,"1to1","single","mpnet PLAIN 2012-2019 (LFS/OJV analysis measure) / 1to1",a_embed="mpnet"),
 _c("pb_mpnet_changes_2012_2019","pb","pb_mpnet_changes_2012_2019",_PBM,_UP,"mpnet_finetuned",0.32,0.40,0.40,"1to1","single","mpnet CLEANED 2012-2019 (LFS/OJV analysis measure) / 1to1",a_embed="mpnet"),
]
CFG = {c["key"]: c for c in CONFIGS}
def configs(family=None): return [c for c in CONFIGS if family is None or c["family"]==family]
