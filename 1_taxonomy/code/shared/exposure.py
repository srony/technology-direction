import pandas as pd
import torch 
from sentence_transformers import SentenceTransformer, util
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import platform
from pathlib import Path
import os
from tqdm import tqdm
tqdm.pandas()
import pickle

# extract_tasks and batch_encode are the canonical shared-library helpers; re-export them here
# so `from shared.exposure import *` (used by 01_replicate_techxposure.ipynb) keeps resolving them.
from shared.utils import extract_tasks
from shared.patent_cache import batch_encode



def prepare_patent_embeddings(patent_df, text_columns, model, batch_encode, save_path=None):
    patent_df = patent_df.dropna(subset=text_columns)

    patent_embedding_dict = {}
    
    for col in text_columns:
        print(f"Encoding: {col}")
        texts = patent_df[col].astype(str).tolist()
        patent_embedding_dict[f"{col}_embedding"] = batch_encode(texts, model)

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(patent_embedding_dict, f)

    return patent_embedding_dict



def prepare_occupation_embeddings(occupation_df, model, batch_encode, save_path=None):
    """
    Prepare embeddings for occupations (titles + tasks).

    Parameters
    ----------
    occupation_df : pd.DataFrame
        Must contain columns ['occupation', 'tasks'].
        'tasks' can be a list of strings or a semicolon-separated string.
    model : any
        Embedding model handle (used inside batch_encode).
    batch_encode : callable
        Function (texts, model) -> torch.Tensor [num_texts x dim].
    save_path : str, optional
        If provided, saves the embedding dictionary (pickle).

    Returns
    -------
    dict
        occupation_embeddings = {
            title: {
                'title_embedding': torch.Tensor,
                'task_embeddings': torch.Tensor
            }
        }
    """
    # Drop missing occupations/tasks
    occupation_df = occupation_df.dropna(subset=['tasks', 'occupation'])

    titles = []
    tasks_all = []
    task_to_occ_idx = []

    # Clean titles and tasks
    for idx, row in occupation_df.iterrows():
        title = str(row['occupation']).strip()
        raw_tasks = row['tasks']

        # Ensure tasks are a list
        if isinstance(raw_tasks, str):
            clean_tasks = [t.strip() for t in raw_tasks.split(";") if t.strip()]
        elif isinstance(raw_tasks, list):
            clean_tasks = [str(t).strip() for t in raw_tasks if t and str(t).strip()]
        else:
            clean_tasks = []

        if not clean_tasks:
            continue

        titles.append(title)
        tasks_all.extend(clean_tasks)
        task_to_occ_idx.extend([idx] * len(clean_tasks))

    # Encode titles and tasks
    title_embeddings = batch_encode(titles, model)
    task_embeddings = batch_encode(tasks_all, model)

    # Group task embeddings per occupation index
    occ_task_emb_dict = defaultdict(list)
    for emb, occ_idx in zip(task_embeddings, task_to_occ_idx):
        occ_task_emb_dict[occ_idx].append(emb)
    for k in occ_task_emb_dict:
        occ_task_emb_dict[k] = torch.stack(occ_task_emb_dict[k])

    # Build occupation embeddings dict
    occupation_embeddings = {}
    for idx, title in enumerate(titles):
        original_idx = occupation_df.index[idx]
        task_embs = occ_task_emb_dict.get(original_idx, torch.empty(0))
        occupation_embeddings[title] = {
            'title_embedding': title_embeddings[idx],
            'task_embeddings': task_embs
        }

    # Save if path provided
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(occupation_embeddings, f)

    return occupation_embeddings



def compute_patent_occupation_similarities(
    occupation_embeddings, occupation_df, patent_df, patent_embedding_dict
):

    results = []
    publn_nrs = patent_df["publn_nr"].to_numpy()   # positional lookup, hoisted out of the inner loop

    # Pre-extract occupation titles & embeddings for quick tensor stacking
    occ_titles = list(occupation_embeddings.keys())
    occ_title_embs = torch.stack([occupation_embeddings[t]['title_embedding'] for t in occ_titles])
    occ_task_embs = [occupation_embeddings[t]['task_embeddings'] for t in occ_titles]

    for emb_name, patent_embs in patent_embedding_dict.items():
        print(f"Processing embedding type: {emb_name}")

        # Normalize embeddings (cosine similarity)
        patent_embs_norm = torch.nn.functional.normalize(patent_embs, dim=1)
        occ_title_embs_norm = torch.nn.functional.normalize(occ_title_embs, dim=1)

        # Title similarity: [num_occupations x num_patents]
        title_sim_matrix = torch.mm(occ_title_embs_norm, patent_embs_norm.T)

        # Task similarities per occupation
        for occ_idx, task_embs in enumerate(tqdm(occ_task_embs, desc=f"Task similarity for {emb_name}")):
            if task_embs.nelement() == 0:
                max_task_sim = torch.zeros(patent_embs.shape[0])
                max_sim_task = [None] * patent_embs.shape[0]
            else:
                task_embs_norm = torch.nn.functional.normalize(task_embs, dim=1)  # T x D
                sims = torch.mm(task_embs_norm, patent_embs_norm.T)  # (T x P)
                max_task_sim, max_task_idx = torch.max(sims, dim=0)

                # Get tasks for this occupation
                original_occ_idx = occupation_df.index[occ_idx]
                # inside compute_patent_occupation_similarities loop
                raw_tasks = occupation_df.loc[original_occ_idx, 'tasks']

                if raw_tasks is None:
                    task_list = []
                elif isinstance(raw_tasks, str):
                    task_list = [t.strip() for t in raw_tasks.split(";") if t.strip()]
                elif isinstance(raw_tasks, list):
                    task_list = [str(t).strip() for t in raw_tasks if t and str(t).strip()]
                else:
                    task_list = []

                max_sim_task = [task_list[i] if i < len(task_list) else None for i in max_task_idx.tolist()]

            # Collect results (one .tolist() per row instead of per-element .item())
            occ_title = occ_titles[occ_idx]
            title_row = title_sim_matrix[occ_idx].tolist()
            task_row = max_task_sim.tolist()
            for patent_idx in range(len(publn_nrs)):
                results.append({
                    'publn_nr':  publn_nrs[patent_idx],
                    'embedding_type': emb_name,
                    'occupation': occ_title,
                    'title_similarity': title_row[patent_idx],
                    'max_task_similarity': task_row[patent_idx],
                    'task_with_max_similarity': max_sim_task[patent_idx]
                })

    return pd.DataFrame(results)




def plot_similarity_heatmaps_occupation(results_df, save_dir, derwent_embed="original_derwent_title_embedding"):

    outputs = {}

    # === 1. Task match vs Derwent ===
    tasks_max_simil = results_df.pivot_table(
        index=["publn_nr", "occupation"],
        columns="embedding_type",
        values="task_with_max_similarity",
        aggfunc="first"
    ).reset_index()
    tasks_max_simil.columns.name = None
    embedding_columns = [c for c in tasks_max_simil.columns if c.endswith("_embedding")]

    match_fractions = {}
    for col in embedding_columns:
        if col != derwent_embed:
            match_rate = (tasks_max_simil[derwent_embed] == tasks_max_simil[col]).mean()
            match_fractions[f"{derwent_embed}_vs_{col}"] = match_rate
    match_df = pd.DataFrame.from_dict(match_fractions, orient="index", columns=["match_rate"])

    plt.figure(figsize=(8, len(match_df) * 0.6))
    sns.heatmap(match_df, annot=True, fmt=".2%", cmap="Blues", cbar=False)
    plt.title(f"Task Match Fractions vs {derwent_embed}", fontsize=14)
    plt.tight_layout()

    path1 = save_dir / f"occupation_task_match_derwent_vs_all_{results_df['publn_nr'].nunique()}.png"
    plt.savefig(path1)
    plt.show()
    outputs["derwent_vs_all"] = match_df

    # === 2. Title similarity correlations ===
    title_sim_pivot = results_df.pivot_table(
        index=["publn_nr", "occupation"],
        columns="embedding_type",
        values="title_similarity"
    )
    title_sim_pivot.columns = [f"title_sim_{c}" for c in title_sim_pivot.columns]
    corr_matrix = title_sim_pivot.reset_index().drop(columns=["publn_nr", "occupation"]).corr()

    derwent_col = f"title_sim_{derwent_embed}"
    filtered_corr = corr_matrix[[derwent_col]].drop(index=derwent_col).sort_values(by=derwent_col, ascending=False)

    plt.figure(figsize=(6, len(filtered_corr) * 0.5 + 1))
    sns.heatmap(filtered_corr, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title("Correlation with Derwent Title Similarity")
    plt.tight_layout()
    path2 = save_dir / f"occupation_title_similarity_corr_{results_df['publn_nr'].nunique()}.png"
    plt.savefig(path2)
    plt.show()
    outputs["title_similarity_corr"] = filtered_corr

    # === 3. Task similarity correlations ===
    task_sim_pivot = results_df.pivot_table(
        index=["publn_nr", "occupation"],
        columns="embedding_type",
        values="max_task_similarity"
    )
    task_sim_pivot.columns = [f"task_sim_{c}" for c in task_sim_pivot.columns]
    corr_matrix = task_sim_pivot.reset_index().drop(columns=["publn_nr", "occupation"]).corr()

    derwent_col = f"task_sim_{derwent_embed}"
    filtered_corr = corr_matrix[[derwent_col]].drop(index=derwent_col).sort_values(by=derwent_col, ascending=False)

    plt.figure(figsize=(6, len(filtered_corr) * 0.5 + 1))
    sns.heatmap(filtered_corr, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title("Correlation with Derwent Task Similarity")
    plt.tight_layout()
    path3 = save_dir / f"occupation_task_similarity_corr_{results_df['publn_nr'].nunique()}.png"
    plt.savefig(path3)
    plt.show()
    outputs["task_similarity_corr"] = filtered_corr

    return outputs



def compute_industry_patent_similarities(
    industry_embeddings, industry_df, patent_df, patent_embedding_dict, industry_name_col="Code"
):

    results = []
    publn_nrs = patent_df["publn_nr"].to_numpy()   # positional lookup, hoisted out of the inner loop

    industry_embs_norm = torch.nn.functional.normalize(industry_embeddings, dim=1)

    for emb_name, patent_embs in patent_embedding_dict.items():
        print(f"Processing embedding type: {emb_name}")

        # Normalize patent embeddings
        patent_embs_norm = torch.nn.functional.normalize(patent_embs, dim=1)

        # Similarity matrix: [num_industries x num_patents]
        name_sim_matrix = torch.mm(industry_embs_norm, patent_embs_norm.T)

        # Collect results
        for ind_idx, ind_name in enumerate(industry_df[industry_name_col]):
            name_row = name_sim_matrix[ind_idx].tolist()
            for patent_idx in range(len(publn_nrs)):
                results.append({
                    'publn_nr':  publn_nrs[patent_idx],
                    'embedding_type': emb_name,
                    'industry': ind_name,
                    'description_similarity': name_row[patent_idx]
                })

    return pd.DataFrame(results)


def plot_industry_similarity_heatmap(results_df, save_dir, derwent_embed="original_derwent_title_embedding"):

    save_dir = Path(save_dir)

    # Pivot into industry vs embedding_type similarity values
    name_sim_pivot = results_df.pivot_table(
        index=['publn_nr', 'industry'],
        columns='embedding_type',
        values='description_similarity'
    )

    comparison_df = name_sim_pivot.reset_index()

    # Correlation across embeddings
    correlation_matrix = comparison_df.drop(columns=['publn_nr', 'industry']).corr()

    # Filter correlations vs Derwent
    derwent_col = derwent_embed
    filtered_corr = correlation_matrix[[derwent_col]].drop(index=derwent_col)
    filtered_corr = filtered_corr.sort_values(by=derwent_col, ascending=False)

    # Plot heatmap
    plt.figure(figsize=(6, len(filtered_corr) * 0.5 + 1))
    sns.heatmap(filtered_corr, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
    plt.title("Correlation with Derwent Title Similarity of Industry Description")
    plt.tight_layout()

    save_path = save_dir / f"industry_similarity_corr_{results_df['publn_nr'].nunique()}.png"
    plt.savefig(save_path)
    plt.show()

    return filtered_corr

from typing import Dict, Optional, List, Union 
import requests
import json

def extract_patent_functionality_technology(
    patstat_patent: str,
    OPENROUTER_API_KEY: str,
    model: str = "openai/gpt-4o-mini",
    temperature: float = 0.3,
    top_p: float = 0.9
) -> Dict[str, str]:
    """
    Extracts Functionality and Technology from a patent abstract using an OpenRouter LLM.

    Parameters
    ----------
    patstat_patent : str
        Patent abstract or description text.
    model : str
        Model identifier available on https://openrouter.ai/models
    temperature : float
        Sampling temperature for LLM (lower = more deterministic).
    top_p : float
        Nucleus sampling probability.

    Returns
    -------
    dict
        {
          "functionality": str,
          "technology": str
        }
    """
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    if not OPENROUTER_API_KEY:
        raise EnvironmentError(
            "Missing OPENROUTER_API_KEY environment variable. "
            "Get one from https://openrouter.ai/settings/keys"
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    system_prompt = (
        "You are an expert in patent analysis. "
        "Given a patent abstract, identify two concise elements:\n"
        "1. Functionality — what the invention *does* or *achieves*(3-5 words). If it achieves more than one state, separate them with commas.\n"
        "2. Technology — the specific *technology* mentioned in the abstract to achieve the functionality (e.g., AI, blockchain).\n"
        "Return JSON with keys 'functionality' and 'technology'."
    )

    user_prompt = f"Patent abstract:\n{patstat_patent.strip()}\n\nExtract the functionality and technology."

    payload = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "patent_extraction",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "functionality": {"type": "string"},
                        "technology": {"type": "string"}
                    },
                    "required": ["functionality", "technology"],
                    "additionalProperties": False
                },
            },
        },
    }

    try:
        
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        #print("LLM Response:", content)
        return json.loads(content)
    except (requests.exceptions.RequestException, KeyError, json.JSONDecodeError) as e:
        return {
            "functionality": "",
            "technology": "",
            "error": str(e)
        }


def load_all_results(pattern, save_dir):
    dfs = []
    for p in save_dir.glob(pattern):
        df = pd.read_pickle(p)
        df["embedding_type"] = df["model"] + "_" + df["embedding_type"]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)



def per_model_corr(df, value_col, index_cols):
    rows = []
    for model, g in df.groupby("model"):
        pivot = g.pivot_table(index=index_cols, columns="embedding_type", values=value_col, aggfunc="mean")
        derwent_col = f"{model}_derwent_title_embedding"
        if derwent_col not in pivot.columns:
            continue
        corr = pivot.corr()[[derwent_col]].drop(index=derwent_col).rename(columns={derwent_col: "corr"})
        corr["model"] = model
        corr["embedding"] = corr.index
        rows.append(corr)
    if rows:
        return pd.concat(rows, ignore_index=True)
    else:
        return pd.DataFrame(columns=["embedding", "corr", "model"])

def per_model_task_match(df, index_cols):
    rows = []
    for model, g in df.groupby("model"):
        pivot = g.pivot_table(index=index_cols, columns="embedding_type", values="task_with_max_similarity", aggfunc="first")
        derwent_col = f"{model}_derwent_title_embedding"
        if derwent_col not in pivot.columns:
            continue
        for col in pivot.columns:
            if col == derwent_col: continue
            match_rate = (pivot[col] == pivot[derwent_col]).mean()
            rows.append({"model": model, "embedding": col, "match_rate": match_rate})
    return pd.DataFrame(rows)


def plot_stacked_heatmap(df, value_col, title, out_name, outputs_path, fmt=".2f", cmap="coolwarm", vlim=None):
    df["label"] = df["model"] + " — " + df["embedding"]
    df = df.set_index("label")[[value_col]].sort_values(by=value_col, ascending=False)
    plt.figure(figsize=(8, 0.5 * len(df) + 1))
    sns.heatmap(df, annot=True, cmap=cmap, fmt=fmt, cbar=True, vmin=vlim[0] if vlim else None, vmax=vlim[1] if vlim else None)
    plt.title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(outputs_path / out_name, dpi=200)
    plt.show()


def run_bertopic_model(
    texts,
    model_name: str,
    save_dir: Path,
    n_neighbors: int = 15,
    n_components: int = 5,
    min_dist: float = 0.0,
    umap_metric: str = "cosine",
    min_cluster_size: int = 30,
    hdbscan_metric: str = "euclidean",
    ngram_range: tuple = (1, 3),
    min_df: int = 1,
    max_df: float = 1.0,
    stop_words: str = "english",
    batch_size: int = 64,
    normalize_embeddings: bool = True,
    verbose: bool = True,
):
    """
    Run BERTopic on patent abstracts using a SentenceTransformer model.

    Parameters
    ----------
    texts : list[str]
        List of text documents (e.g., patent abstracts).
    model_name : str
        Name of SentenceTransformer model.
    save_dir : Path
        Directory to save BERTopic model and topic info.
    n_neighbors : int
        UMAP neighbors (smaller = more granular topics).
    n_components : int
        UMAP dimensions.
    min_dist : float
        UMAP distance parameter (0 = tighter clusters).
    umap_metric : str
        Distance metric for UMAP.
    min_cluster_size : int
        Minimum HDBSCAN cluster size.
    hdbscan_metric : str
        Distance metric for HDBSCAN.
    ngram_range : tuple
        n-gram range for CountVectorizer.
    min_df, max_df : int, float
        Document frequency thresholds for CountVectorizer.
    stop_words : str or list
        Stopword list or "english".
    batch_size : int
        Batch size for encoding.
    normalize_embeddings : bool
        Whether to L2-normalize sentence embeddings.
    verbose : bool
        Print progress and info.

    Returns
    -------
    topic_model : BERTopic
        The fitted BERTopic model.
    topic_info : pd.DataFrame
        Topic information DataFrame.
    """
    # Heavy deps imported lazily so importing shared.exposure (e.g. for its plotting
    # helpers) does not pull in bertopic/umap/numba/hdbscan up front.
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    import hdbscan

    save_dir = Path(save_dir)


    # === Load embedding model ===
    if verbose:
        print(f"\n Encoding texts with model: {model_name}")
    model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")

    # === Encode texts ===
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=verbose,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )

    # === Dimensionality reduction ===
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=umap_metric,
        random_state=42,
    )

    # === Clustering ===
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric=hdbscan_metric,
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # === Vectorizer ===
    vectorizer = CountVectorizer(
        stop_words=stop_words,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
    )

    # === BERTopic ===
    topic_model = BERTopic(
        embedding_model=None,  
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        calculate_probabilities=True,
        verbose=verbose,
    )

    # === Fit model ===
    if verbose:
        print(" Fitting BERTopic model...")
    topics, probs = topic_model.fit_transform(texts, embeddings)

    # === Save model & topic info ===
    model_path = save_dir / "bertopic_topics" / f"bertopic_{model_name.replace('/', '_')}"
    topic_model.save(str(model_path))

    topic_info = topic_model.get_topic_info()
    csv_path = save_dir / "bertopic_topics" / f"topics_{model_name.replace('/', '_')}.csv"
    topic_info.to_csv(csv_path, index=False)

    if verbose:
        print(f"✅ Saved BERTopic model to {model_path}")
        print(f"✅ Saved topic info to {csv_path}")

    return topic_model, topic_info


