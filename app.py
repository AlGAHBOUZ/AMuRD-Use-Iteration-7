import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from teradataml import DataFrame
from src.modules.db import TeradataDatabase
from src.modules.models import SentenceEmbeddingModel, SentenceEmbeddingConfig

# ---------- Config ----------
DEVICE = "cuda"
SCHEMA = "demo_user"
PRODUCTS_TBL = "products"       # id, translated_name
CLASSES_TBL  = "classes"        # id, class_name
P_EMB_TBL    = "p_embeddings"   # id, embed_0..embed_N
C_EMB_TBL    = "c_embeddings"   # id, embed_0..embed_N
ACTUALS_TBL  = "actual_classes" # schema may vary; handled flexibly

# ---------- TF-IDF Config ----------
@dataclass
class TfidfClassifierConfig:
    analyzer: str = "char_wb"
    ngram_range: tuple = (3, 5)
    min_df: int = 2
    max_df: float = 0.9
    lowercase: bool = True
    sublinear_tf: bool = True
    smooth_idf: bool = True
    norm: str = "l2"
    strip_accents: Optional[str] = None
    stop_words: Optional[set] = None

class Tfidf:
    def __init__(self, config: Optional[TfidfClassifierConfig]):
        self.vectorizer = TfidfVectorizer(
            analyzer=config.analyzer,
            ngram_range=config.ngram_range,
            min_df=config.min_df,
            max_df=config.max_df,
            lowercase=config.lowercase,
            sublinear_tf=config.sublinear_tf,
            smooth_idf=config.smooth_idf,
            norm=config.norm,
            strip_accents=config.strip_accents,
            stop_words=config.stop_words,
            token_pattern=None if config.analyzer in ("char", "char_wb") else r'(?u)\b\w+\b',
        )

        self.product_vectors = None
        self.class_vectors = None
        self.class_names = None

    def fit(self, product_names, classes):
        corpus = product_names + classes

        tfidf_matrix = self.vectorizer.fit_transform(corpus)

        self.class_names = classes
        self.product_vectors = tfidf_matrix[:len(product_names)]
        self.class_vectors = tfidf_matrix[len(product_names):]

    def predict(self, product_name):
        vector = self.vectorizer.transform([product_name])  # Fixed: wrap in list
        scores = cosine_similarity(vector, self.class_vectors)
        class_idx = np.argmax(scores, axis=1)
        class_name = [self.class_names[idx] for idx in class_idx]

        return class_name[0], scores[0]  # Return single prediction and all scores

    def predict_topk(self, product_name, k=3):
        vector = self.vectorizer.transform([product_name])
        scores = cosine_similarity(vector, self.class_vectors)[0]  # Get first row
        top_k_indices = np.argsort(-scores)[:k]
        top_k_scores = scores[top_k_indices]
        return top_k_scores, top_k_indices

# ---------- DB helpers ----------
@st.cache_resource
def get_db():
    db = TeradataDatabase()
    db.connect()
    return db

@st.cache_data
def load_data():
    _ = get_db()  # ensure teradataml context

    products_df = DataFrame.from_table(PRODUCTS_TBL, schema_name=SCHEMA)[["id", "translated_name"]].to_pandas()
    classes_df  = DataFrame.from_table(CLASSES_TBL,  schema_name=SCHEMA)[["id", "class_name"]].to_pandas()
    p_emb_df    = DataFrame.from_table(P_EMB_TBL,    schema_name=SCHEMA).to_pandas()
    c_emb_df    = DataFrame.from_table(C_EMB_TBL,    schema_name=SCHEMA).to_pandas()

    # Merge embeddings → names
    product_full = p_emb_df.merge(products_df, on="id", how="left")
    class_full   = c_emb_df.merge(classes_df,  on="id", how="left")

    # --- Ground truth (robust join)
    try:
        actuals_df = DataFrame.from_table(ACTUALS_TBL, schema_name=SCHEMA).to_pandas()
        actuals_df.columns = [c.strip().lower() for c in actuals_df.columns]
        classes_df_norm = classes_df.rename(columns={"id": "class_id"}).copy()
        classes_df_norm.columns = [c.strip().lower() for c in classes_df_norm.columns]

        prod_key = "product_id" if "product_id" in actuals_df.columns else ("id" if "id" in actuals_df.columns else None)
        if prod_key is None:
            product_full["true_class_id"] = np.nan
            product_full["true_class_name"] = np.nan
        else:
            if "class_id" in actuals_df.columns:
                gt = actuals_df[[prod_key, "class_id"]].rename(columns={prod_key: "id"})
                gt = gt.merge(
                    classes_df_norm[["class_id", "class_name"]].rename(columns={"class_name": "true_class_name"}),
                    on="class_id", how="left"
                ).rename(columns={"class_id": "true_class_id"})
                product_full = product_full.merge(gt, on="id", how="left")
            elif "class_name" in actuals_df.columns:
                gt = actuals_df[[prod_key, "class_name"]].rename(columns={prod_key: "id"})
                gt = gt.merge(classes_df_norm, on="class_name", how="left")
                gt = gt.rename(columns={"class_id": "true_class_id", "class_name": "true_class_name"})
                product_full = product_full.merge(gt[["id", "true_class_id", "true_class_name"]], on="id", how="left")
            else:
                product_full["true_class_id"] = np.nan
                product_full["true_class_name"] = np.nan
    except Exception:
        product_full["true_class_id"] = np.nan
        product_full["true_class_name"] = np.nan

    return product_full, class_full, classes_df

# ---------- E5 (DB) ----------
@st.cache_data
def load_embeddings_from_db(product_full: pd.DataFrame, class_full: pd.DataFrame):
    prod_cols = sorted([c for c in product_full.columns if c.startswith("embed_")],
                       key=lambda x: int(x.split("_")[1]))
    cls_cols  = sorted([c for c in class_full.columns if c.startswith("embed_")],
                       key=lambda x: int(x.split("_")[1]))
    if not prod_cols or not cls_cols:
        raise ValueError("Missing embed_* columns in DB tables.")

    prod = torch.tensor(product_full[prod_cols].to_numpy(np.float32, copy=False), dtype=torch.float16, device=DEVICE)
    cls  = torch.tensor(class_full[cls_cols].to_numpy(np.float32, copy=False), dtype=torch.float16, device=DEVICE)
    return F.normalize(prod.float(), p=2, dim=1).half(), F.normalize(cls.float(), p=2, dim=1).half()

def predict_topk_e5(prod_vec: torch.Tensor, cls_mat: torch.Tensor, k: int = 3):
    if prod_vec.dim() == 1:
        prod_vec = prod_vec.unsqueeze(0)
    scores = torch.mm(prod_vec, cls_mat.T)  # cosine (normalized)
    vals, idx = torch.topk(scores, k=min(k, cls_mat.size(0)), dim=1)
    return vals[0].cpu().numpy(), idx[0].cpu().numpy()

# ---------- TF-IDF (Fixed Implementation) ----------
@st.cache_resource
def build_tfidf_fixed(product_full: pd.DataFrame, classes_df: pd.DataFrame):
    """Build TF-IDF using your Tfidf class with proper training corpus"""
    config = TfidfClassifierConfig()
    tfidf_model = Tfidf(config)
    
    # Prepare training data
    product_names = product_full["translated_name"].fillna("").astype(str).tolist()
    class_names = classes_df["class_name"].fillna("").astype(str).tolist()
    
    # Fit the model
    tfidf_model.fit(product_names, class_names)
    
    return tfidf_model

def predict_topk_tfidf_fixed(text: str, tfidf_model: Tfidf, k: int = 3):
    """Predict using the fixed TF-IDF implementation"""
    scores, idxs = tfidf_model.predict_topk(text, k=k)
    return scores, idxs

# ---------- Optional E5 encoder (only for free-text + E5) ----------
@st.cache_resource
def load_encoder() -> SentenceEmbeddingModel:
    cfg = SentenceEmbeddingConfig(
        device=DEVICE, dtype="float16", model_id="intfloat/e5-large-v2",
        truncate_dim=None, convert_to_numpy=False, convert_to_tensor=True,
        use_prompt=True, prompt_config={"classification": "passage: {text}"},
        model_kwargs={"torch_dtype": "float16"},
    )
    return SentenceEmbeddingModel(cfg)

def e5_embed_texts(texts):
    enc = load_encoder()
    emb = enc.get_embeddings([t if t is not None else "" for t in texts], prompt_name="classification")
    if not torch.is_tensor(emb):
        emb = torch.tensor(emb)
    emb = emb.to(DEVICE).to(torch.float32)
    emb = F.normalize(emb, p=2, dim=1)
    return emb

# ---------- UI ----------
st.title("🧠 Product Classification — E5 or TF-IDF")

with st.spinner("Loading data from Teradata…"):
    product_full, class_full, classes_df = load_data()

# Model choice
model_choice = st.radio("Choose model", ["E5 (DB embeddings)", "TF-IDF"], horizontal=True)

# Prepare model-specific assets
if model_choice == "E5 (DB embeddings)":
    with st.spinner("Preparing E5 embeddings…"):
        prod_emb, cls_emb = load_embeddings_from_db(product_full, class_full)
else:
    with st.spinner("Building TF-IDF with improved implementation…"):
        tfidf_model = build_tfidf_fixed(product_full, classes_df)

# ---------- Browse & classify DB row ----------
st.subheader("Classify a product from the database")

items_per_page = st.selectbox("Items per page", [5, 10, 25, 50], 1)
total_items = len(product_full)
total_pages = (total_items - 1)//items_per_page + 1
page = st.number_input("Page", 1, total_pages, 1)
s, e = (page-1)*items_per_page, min(page*items_per_page, total_items)
st.dataframe(product_full.iloc[s:e][["id", "translated_name", "true_class_name"]], hide_index=True, use_container_width=True)

chosen_id = st.number_input(
    "Enter product id",
    int(product_full["id"].min()),
    int(product_full["id"].max()),
    int(product_full.iloc[0]["id"])
)
if st.button("🔎 Classify selected product"):
    idx_match = product_full.index[product_full["id"] == chosen_id]
    if len(idx_match) == 0:
        st.error("ID not found.")
    else:
        pidx = int(idx_match[0])
        product_name = str(product_full.loc[pidx, "translated_name"])

        if model_choice == "E5 (DB embeddings)":
            scores, idxs = predict_topk_e5(prod_emb[pidx], cls_emb, k=3)
        else:
            scores, idxs = predict_topk_tfidf_fixed(product_name, tfidf_model, k=3)

        top1_idx = int(idxs[0])
        # classes come from classes_df (id, class_name)
        pred_name = classes_df["class_name"].iloc[top1_idx]
        pred_id   = int(classes_df["id"].iloc[top1_idx])
        st.success(f"Prediction: {pred_name} (id={pred_id}, score={float(scores[0]):.3f})")

        # ground truth
        true_id   = product_full.loc[pidx, "true_class_id"] if "true_class_id" in product_full.columns else np.nan
        true_name = product_full.loc[pidx, "true_class_name"] if "true_class_name" in product_full.columns else np.nan
        if pd.notna(true_id):
            st.write("✅ Correct" if int(true_id) == pred_id else "❌ Incorrect")
            st.write(f"Ground truth: {true_name} (id={int(true_id)})")
        else:
            st.info("No ground truth for this product.")

        st.caption("Top-3:")
        for r, (sc, ci) in enumerate(zip(scores, idxs), start=1):
            cname = classes_df["class_name"].iloc[int(ci)]
            cid   = int(classes_df["id"].iloc[int(ci)])
            st.write(f"{r}. {cname} (id={cid}) — {float(sc):.3f}")

# ---------- Free-text classify ----------
st.subheader("Classify a custom description")
user_text = st.text_area("Type a product description…", "", height=100)
free_model = st.radio("Model for free-text", ["TF-IDF", "E5 (encode now)"], horizontal=True, key="free_model")

if st.button("✨ Classify text"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        if free_model == "TF-IDF":
            # ensure tf-idf is built with the fixed implementation
            if model_choice != "TF-IDF":
                tfidf_model = build_tfidf_fixed(product_full, classes_df)
            scores, idxs = predict_topk_tfidf_fixed(user_text, tfidf_model, k=3)
        else:
            # E5 on-the-fly just for this text
            cls_cols  = sorted([c for c in class_full.columns if c.startswith("embed_")],
                               key=lambda x: int(x.split("_")[1]))
            cls_mat = torch.tensor(class_full[cls_cols].to_numpy(np.float32, copy=False), device=DEVICE)
            cls_mat = F.normalize(cls_mat, p=2, dim=1)
            q = e5_embed_texts([user_text])         # [1, dim], normalized
            scores = torch.mm(q, cls_mat.T).cpu().numpy()[0]
            idxs = np.argsort(-scores)[:3]
            scores = scores[idxs]

        top1_idx = int(idxs[0])
        pred_name = classes_df["class_name"].iloc[top1_idx]
        pred_id   = int(classes_df["id"].iloc[top1_idx])
        st.success(f"Prediction: {pred_name} (id={pred_id}, score={float(scores[0]):.3f})")

        st.caption("Top-3:")
        for r, (sc, ci) in enumerate(zip(scores, idxs), start=1):
            cname = classes_df["class_name"].iloc[int(ci)]
            cid   = int(classes_df["id"].iloc[int(ci)])
            st.write(f"{r}. {cname} (id={cid}) — {float(sc):.3f}")