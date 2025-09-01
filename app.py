import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.constants import E5_LARGE_INSTRUCT_CONFIG_PATH, JOINED_PRODUCTS_DATASET, JOINED_CLASSES_DATASET
from teradataml import DataFrame
from src.modules.db import TeradataDatabase
from src.utils import load_embedding_model, load_tfidf

# ---------- Config ----------
DEVICE = "cuda"
SCHEMA = "demo_user"
PRODUCTS_TBL = "products"       # id, translated_name
CLASSES_TBL  = "classes"        # id, class_name
ACTUALS_TBL  = "actual_classes" # schema may vary; handled flexibly


# ---------- DB helpers ----------
@st.cache_resource
def get_db():
    db = TeradataDatabase()
    db.connect()
    return db

@st.cache_data
def load_basic_data():
    """Load only products and classes tables from database"""
    start_time = time.time()
    db = get_db()
    
    products_df = pd.DataFrame(db.execute_query(f"SELECT * FROM {SCHEMA}.{PRODUCTS_TBL}"))
    classes_df = pd.DataFrame(db.execute_query(f"SELECT * FROM {SCHEMA}.{CLASSES_TBL}"))
    
    # Try to load actuals for ground truth
    try:
        actuals_df = DataFrame.from_table(ACTUALS_TBL, schema_name=SCHEMA).to_pandas()
        actuals_df.columns = [c.strip().lower() for c in actuals_df.columns]
        classes_df_norm = classes_df.rename(columns={"id": "class_id"}).copy()
        classes_df_norm.columns = [c.strip().lower() for c in classes_df_norm.columns]

        prod_key = "product_id" if "product_id" in actuals_df.columns else ("id" if "id" in actuals_df.columns else None)
        if prod_key is None:
            products_df["true_class_id"] = np.nan
            products_df["true_class_name"] = np.nan
        else:
            if "class_id" in actuals_df.columns:
                gt = actuals_df[[prod_key, "class_id"]].rename(columns={prod_key: "id"})
                gt = gt.merge(
                    classes_df_norm[["class_id", "class_name"]].rename(columns={"class_name": "true_class_name"}),
                    on="class_id", how="left"
                ).rename(columns={"class_id": "true_class_id"})
                products_df = products_df.merge(gt, on="id", how="left")
            elif "class_name" in actuals_df.columns:
                gt = actuals_df[[prod_key, "class_name"]].rename(columns={prod_key: "id"})
                gt = gt.merge(classes_df_norm, on="class_name", how="left")
                gt = gt.rename(columns={"class_id": "true_class_id", "class_name": "true_class_name"})
                products_df = products_df.merge(gt[["id", "true_class_id", "true_class_name"]], on="id", how="left")
            else:
                products_df["true_class_id"] = np.nan
                products_df["true_class_name"] = np.nan
    except Exception:
        products_df["true_class_id"] = np.nan
        products_df["true_class_name"] = np.nan

    load_time = time.time() - start_time

    
    return products_df, classes_df

# ---------- Model Loading ----------
@st.cache_resource
def load_e5_encoder():
    """Load E5 embedding model"""
    model = load_embedding_model(E5_LARGE_INSTRUCT_CONFIG_PATH)
    return model

@st.cache_resource
def load_tfidf_model(product_names, class_names):
    """Build TF-IDF model from product and class names"""
    tfidf_model = load_tfidf()
    tfidf_model.fit(product_names, class_names)
    return tfidf_model

# ---------- Embedding Functions ----------
def get_e5_embeddings(texts, model):
    """Get E5 embeddings for given texts"""
    emb = model.get_embeddings([t if t is not None else "" for t in texts], prompt_name="classification")
    if not torch.is_tensor(emb):
        emb = torch.tensor(emb)
    emb = emb.to(DEVICE).to(torch.float32)
    emb = F.normalize(emb, p=2, dim=1)
    return emb

def predict_with_e5(product_text, class_texts, e5_model, k=3):
    """Predict top-k classes using E5 embeddings"""
    # Get embeddings
    prod_emb = get_e5_embeddings([product_text], e5_model)
    cls_emb = get_e5_embeddings(class_texts, e5_model)
    
    # Calculate cosine similarities
    scores = torch.mm(prod_emb, cls_emb.T).cpu().numpy()[0]
    
    # Get top-k
    top_indices = np.argsort(-scores)[:k]
    top_scores = scores[top_indices]
    
    return top_scores, top_indices

def predict_with_tfidf(product_text, tfidf_model, k=3):
    """Predict top-k classes using TF-IDF"""
    scores, indices = tfidf_model.predict_topk(product_text, k=k)
    return scores, indices

# ---------- UI ----------
st.title("Classi-Fy, A Modern Product Classification App")

with st.spinner("Loading data from Teradata…"):
    products_df, classes_df = load_basic_data()


# Model choice
model_choice = st.radio("Choose model", ["E5 (On-Demand)", "TF-IDF"], horizontal=True)

# Prepare model-specific assets
if model_choice == "E5 (On-Demand)":
    with st.spinner("Loading E5 model…"):
        e5_model = load_e5_encoder()
else:
    with st.spinner("Building TF-IDF model…"):
        product_names = products_df["translated_name"].fillna("").astype(str).tolist()
        class_names = classes_df["class_name"].fillna("").astype(str).tolist()
        tfidf_model = load_tfidf_model(product_names, class_names)


# ---------- Browse & classify DB row ----------
st.subheader("Classify a product from the database")

items_per_page = st.selectbox("Items per page", [5, 10, 25, 50], 1)
total_items = len(products_df)
total_pages = (total_items - 1)//items_per_page + 1
page = st.number_input("Page", 1, total_pages, 1)
s, e = (page-1)*items_per_page, min(page*items_per_page, total_items)

# Display current page with ground truth if available
display_cols = ["id", "translated_name"]
if "true_class_name" in products_df.columns:
    display_cols.append("true_class_name")
    
st.dataframe(products_df.iloc[s:e][display_cols], hide_index=True, use_container_width=True)

chosen_id = st.number_input(
    "Enter product id",
    int(products_df["id"].min()),
    int(products_df["id"].max()),
    int(products_df.iloc[0]["id"])
)

if st.button("🔎 Classify selected product"):
    idx_match = products_df.index[products_df["id"] == chosen_id]
    if len(idx_match) == 0:
        st.error("ID not found.")
    else:
        pidx = int(idx_match[0])
        product_name = str(products_df.loc[pidx, "translated_name"])
        
        st.info(f"Classifying: {product_name}")
        
        with st.spinner("Computing embeddings and predictions..."):
            if model_choice == "E5 (On-Demand)":
                class_texts = classes_df["class_name"].fillna("").astype(str).tolist()
                scores, idxs = predict_with_e5(product_name, class_texts, e5_model, k=3)
            else:
                scores, idxs = predict_with_tfidf(product_name, tfidf_model, k=3)

        # Display prediction
        top1_idx = int(idxs[0])
        pred_name = classes_df["class_name"].iloc[top1_idx]
        pred_id = int(classes_df["id"].iloc[top1_idx])
        st.success(f"**Top Prediction:** {pred_name} (id={pred_id}, score={float(scores[0]):.3f})")

        # Display ground truth if available
        if "true_class_name" in products_df.columns and pd.notna(products_df.loc[pidx, "true_class_name"]):
            true_class = str(products_df.loc[pidx, "true_class_name"])
            st.info(f"**Ground Truth:** {true_class}")
            
        
        # Display top-3 results
        st.caption("**Top-3 predictions:**")
        for r, (sc, ci) in enumerate(zip(scores, idxs), start=1):
            cname = classes_df["class_name"].iloc[int(ci)]
            cid = int(classes_df["id"].iloc[int(ci)])
            st.write(f"{r}. {cname} (id={cid}) — {float(sc):.3f}")

# ---------- Free-text classify ----------
st.subheader("Classify a custom description")
user_text = st.text_area("Type a product description…", "", height=100)

if st.button("✨ Classify text"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Computing embeddings and predictions..."):
            if model_choice == "E5 (On-Demand)":
                class_texts = classes_df["class_name"].fillna("").astype(str).tolist()
                scores, idxs = predict_with_e5(user_text, class_texts, e5_model, k=3)
            else:
                scores, idxs = predict_with_tfidf(user_text, tfidf_model, k=3)

        # Display prediction
        top1_idx = int(idxs[0])
        pred_name = classes_df["class_name"].iloc[top1_idx]
        pred_id = int(classes_df["id"].iloc[top1_idx])
        st.success(f"**Top Prediction:** {pred_name} (id={pred_id}, score={float(scores[0]):.3f})")

        # Display top-3 results
        st.caption("**Top-3 predictions:**")
        for r, (sc, ci) in enumerate(zip(scores, idxs), start=1):
            cname = classes_df["class_name"].iloc[int(ci)]
            cid = int(classes_df["id"].iloc[int(ci)])
            st.write(f"{r}. {cname} (id={cid}) — {float(sc):.3f}")
