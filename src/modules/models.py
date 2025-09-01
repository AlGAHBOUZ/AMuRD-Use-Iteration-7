import torch
from torch import Tensor
import torch.nn.functional as F 
from sklearn.dummy import DummyClassifier
from sentence_transformers import SentenceTransformer
from transformers import MarianMTModel, MarianTokenizer
from sklearn.linear_model import LogisticRegression
from transformers import BitsAndBytesConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from scipy.sparse import diags
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
import requests
import time
import joblib

@dataclass
class OpusTranslationModelConfig:
    padding: bool
    model_name: str
    device: str
    dtype: str
    truncation: bool
    skip_special_tokens: bool


class OpusTranslationModel:

    def __init__(self, config: OpusTranslationModelConfig):
        self.config = config
        self.model = MarianMTModel.from_pretrained(
            self.config.model_name, 
            device_map=self.config.device, 
            torch_dtype=self.config.dtype
        )
        self.tokenizer = MarianTokenizer.from_pretrained(self.config.model_name)
        
    def translate(self, text: str) -> str:
        tokens = self.tokenizer(
            text, 
            padding=self.config.padding, 
            truncation=self.config.truncation, 
            return_tensors="pt"
        ).to(self.config.device)
        translated_tokens = self.model.generate(**tokens)
        translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=self.config.skip_special_tokens)

        return translated_text


@dataclass
class SentenceEmbeddingConfig:
    device: str
    dtype: str
    model_id: str
    truncate_dim: Optional[int]
    convert_to_numpy: bool
    convert_to_tensor: bool
    use_prompt: bool = False
    prompt_config: Optional[Dict[str, str]] = None
    model_kwargs: Optional[Dict[str, Any]] = None


class SentenceEmbeddingModel:
    def __init__(self, config: SentenceEmbeddingConfig):
        super().__init__()
        self.config = config
        self.model_id = config.model_id
        self.device = config.device
        self.dtype = config.dtype
        self.truncate_dim = config.truncate_dim

        model_kwargs = config.model_kwargs or {}

        if "quantization_config" in model_kwargs:
            quant_config = model_kwargs["quantization_config"]
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quant_config.get("load_in_4bit", True),
                bnb_4bit_compute_dtype=getattr(torch, quant_config.get("bnb_4bit_compute_dtype", "float16")),
                bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True)
            )

        if "torch_dtype" in model_kwargs:
            model_kwargs["torch_dtype"] = getattr(torch, model_kwargs["torch_dtype"])

        self.model = SentenceTransformer(
            self.model_id,
            device=self.device,
            truncate_dim=self.truncate_dim,
            model_kwargs=model_kwargs
        )

    def get_embeddings(self, texts: List[str], prompt_name: Optional[str] = None) -> Tensor:
        if self.config.use_prompt and prompt_name and self.config.prompt_config:
            if prompt_name in self.config.prompt_config:
                prompt_template = self.config.prompt_config[prompt_name]
                texts = [prompt_template.format(text=t) for t in texts]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=self.config.convert_to_numpy,
            convert_to_tensor=self.config.convert_to_tensor,
            show_progress_bar=True
        )
        return embeddings

    def calculate_scores(self, query_embeddings: Tensor, document_embeddings: Tensor) -> Tensor:
        return self.model.similarity(query_embeddings, document_embeddings)

    def get_scores(self, queries: List[str], documents: List[str]) -> Tensor:
        query_embeddings = self.get_embeddings(queries, "classification")
        document_embeddings = self.get_embeddings(documents, "classification")
        return self.calculate_scores(query_embeddings, document_embeddings)

@dataclass
class LLMModelConfig:
    api_key: str
    model_name: str = "openai/gpt-4o-mini"
    api_url: str = "https://openrouter.ai/api/v1/chat/completions"
    temperature: float = 0.1
    max_tokens: int = 100
    top_p: float = 0.9
    site_url: str = ""  # Optional
    site_name: str = ""  # Optional



class LLMModel:
    
    def __init__(self, config: LLMModelConfig, prompt_path: str):
        self.config = config
        self.prompt_path = prompt_path
        self.prompt_template = self._load_prompt_template()
        
    def _load_prompt_template(self) -> str:
        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def _prepare_prompt(self, product_name: str, allowed_labels: List[str]) -> str:
        labels_text = "\n".join(allowed_labels)
        return self.prompt_template.replace("{{Product_Name}}", product_name).replace("{{LABEL_1}}\n{{LABEL_2}}\n{{LABEL_3}}\n{{LABEL_4}}\n{{LABEL_5}}\n....", labels_text)
    
    def _make_api_request(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
        }
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": self.config.temperature,
                "topP": self.config.top_p,
                "maxOutputTokens": self.config.max_tokens,
            }
        }
        
        url = f"{self.config.api_url}?key={self.config.api_key}"
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            if "candidates" in result and len(result["candidates"]) > 0:
                return result["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                return ""
        except Exception as e:
            print(f"API request failed: {e}")
            return ""
    
    def _parse_response(self, response: str, allowed_labels: List[str]) -> str:
        if not response:
            return allowed_labels[0] if allowed_labels else ""
        
        response = response.strip()
        
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        for line in lines:
            line_clean = line.replace('```', '').strip()
            if line_clean in allowed_labels:
                return line_clean
        
        for label in allowed_labels:
            if label.lower() in response.lower():
                return label
        
        if lines:
            return lines[0].replace('```', '').strip()
        
        return allowed_labels[0] if allowed_labels else ""
    
    def predict(self, products_df: pd.DataFrame, allowed_labels: List[str], 
                product_text_col: str = "translated_name") -> pd.DataFrame:
        results = []
        
        for idx, row in products_df.iterrows():
            product_name = row[product_text_col]
            
            prompt = self._prepare_prompt(product_name, allowed_labels)
            response = self._make_api_request(prompt)
            predicted_label = self._parse_response(response, allowed_labels)
            
            result = {
                "product_id": row.get("id", idx),
                "product_text": product_name,
                "predicted_label_llm": predicted_label,
                "raw_response": response
            }
            results.append(result)

        time.sleep(0.5)
        
        return pd.DataFrame(results)


class HierarchicalGPCClassifier:
    
    def __init__(self, config: LLMModelConfig, prompt_path: str, gpc_data_df: pd.DataFrame):
        self.config = config
        self.prompt_path = prompt_path
        self.prompt_template = self._load_prompt_template()
        self.gpc_df = gpc_data_df
        
        # Create hierarchical mappings
        self.hierarchy = self._build_hierarchy_mapping()
        
    def _load_prompt_template(self) -> str:
        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def _build_hierarchy_mapping(self) -> Dict:
        """Fast vectorized hierarchy building with caching"""
        import pickle
        import os
        import hashlib
        
        # Create cache filename based on GPC data hash
        gpc_hash = hashlib.md5(str(self.gpc_df.shape).encode() + str(self.gpc_df.columns.tolist()).encode()).hexdigest()[:8]
        cache_file = f"gpc_hierarchy_cache_{gpc_hash}.pkl"
        
        # Check if cached hierarchy exists
        if os.path.exists(cache_file):
            print(f"Loading cached GPC hierarchy from {cache_file}...")
            try:
                with open(cache_file, 'rb') as f:
                    hierarchy = pickle.load(f)
                print(f"✅ Cached hierarchy loaded successfully!")
                print(f"   - {len(hierarchy['segments'])} segments")
                print(f"   - {len(hierarchy['families'])} families") 
                print(f"   - {len(hierarchy['classes'])} classes")
                return hierarchy
            except Exception as e:
                print(f"Failed to load cache: {e}. Building fresh hierarchy...")
        
        print("Building GPC hierarchy mapping (fast version)...")

        # Remove rows with missing critical data
        clean_df = self.gpc_df.dropna(subset=['SegmentTitle']).copy()
        print(f"Working with {len(clean_df)} clean GPC records...")

        hierarchy = {'segments': {}, 'families': {}, 'classes': {}}

        # Group by segment to avoid repeated filtering
        for segment, segment_group in clean_df.groupby('SegmentTitle'):
            families = segment_group['FamilyTitle'].dropna().unique().tolist()
            hierarchy['segments'][segment] = {'families': families}

            # Group by family within this segment
            for family, family_group in segment_group.groupby('FamilyTitle'):
                if pd.isna(family):
                    continue
                family_key = f"{segment}::{family}"
                classes = family_group['ClassTitle'].dropna().unique().tolist()
                hierarchy['families'][family_key] = {'classes': classes}

                # Group by class within this family
                for class_name, class_group in family_group.groupby('ClassTitle'):
                    if pd.isna(class_name):
                        continue
                    class_key = f"{segment}::{family}::{class_name}"
                    bricks = class_group['BrickTitle'].dropna().unique().tolist()
                    hierarchy['classes'][class_key] = {'bricks': bricks}

        print(f"✅ Fast hierarchy mapping complete!")
        print(f"   - {len(hierarchy['segments'])} segments")
        print(f"   - {len(hierarchy['families'])} families") 
        print(f"   - {len(hierarchy['classes'])} classes")
        
        # Save to cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(hierarchy, f)
            print(f"💾 Hierarchy cached to {cache_file}")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
        
        return hierarchy

    def _prepare_prompt(self, product_name: str, allowed_labels: List[str]) -> str:
        labels_text = "\n".join(allowed_labels)
        return self.prompt_template.replace("{{Product_Name}}", product_name).replace("{{LABEL_1}}\n{{LABEL_2}}\n{{LABEL_3}}\n{{LABEL_4}}\n{{LABEL_5}}\n....", labels_text)
    
    def _make_api_request(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        
        # Add optional headers if provided
        if self.config.site_url:
            headers["HTTP-Referer"] = self.config.site_url
        if self.config.site_name:
            headers["X-Title"] = self.config.site_name
        
        payload = {
            "model": self.config.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }
        
        try:
            response = requests.post(self.config.api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
            else:
                return ""
        except Exception as e:
            print(f"API request failed: {e}")
            return ""

    
    def _parse_response(self, response: str, allowed_labels: List[str]) -> str:
        if not response:
            return allowed_labels[0] if allowed_labels else ""
        
        response = response.strip()
        
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        for line in lines:
            line_clean = line.replace('```', '').strip()
            if line_clean in allowed_labels:
                return line_clean
        
        for label in allowed_labels:
            if label.lower() in response.lower():
                return label
        
        if lines:
            return lines[0].replace('```', '').strip()
        
        return allowed_labels[0] if allowed_labels else ""
    
    def _classify_single_level(self, product_name: str, allowed_labels: List[str], level_name: str) -> Dict:
        """Classify product at a single level"""
        if not allowed_labels:
            return {
                "predicted_label": "",
                "raw_response": f"No labels available for {level_name}"
            }
            
        prompt = self._prepare_prompt(product_name, allowed_labels)
        response = self._make_api_request(prompt)
        predicted_label = self._parse_response(response, allowed_labels)
        
        print(f"  {level_name}: {predicted_label}")
        
        return {
            "predicted_label": predicted_label,
            "raw_response": response
        }
    
    def classify_product_hierarchical(self, product_name: str) -> Dict:
        """Classify a single product through all 4 levels"""
        
        # Level 1: Segment
        segments = list(self.hierarchy['segments'].keys())
        segment_result = self._classify_single_level(product_name, segments, "Segment")
        predicted_segment = segment_result["predicted_label"]
        
        # Level 2: Family (within the predicted segment)
        if predicted_segment and predicted_segment in self.hierarchy['segments']:
            families = self.hierarchy['segments'][predicted_segment]['families']
            family_result = self._classify_single_level(product_name, families, "Family")
            predicted_family = family_result["predicted_label"]
        else:
            family_result = {"predicted_label": "", "raw_response": "No segment found"}
            predicted_family = ""
        
        # Level 3: Class (within the predicted family)
        if predicted_family:
            family_key = f"{predicted_segment}::{predicted_family}"
            if family_key in self.hierarchy['families']:
                classes = self.hierarchy['families'][family_key]['classes']
                class_result = self._classify_single_level(product_name, classes, "Class")
                predicted_class = class_result["predicted_label"]
            else:
                class_result = {"predicted_label": "", "raw_response": "Family key not found"}
                predicted_class = ""
        else:
            class_result = {"predicted_label": "", "raw_response": "No family found"}
            predicted_class = ""
        
        # Level 4: Brick (within the predicted class)
        if predicted_class:
            class_key = f"{predicted_segment}::{predicted_family}::{predicted_class}"
            if class_key in self.hierarchy['classes']:
                bricks = self.hierarchy['classes'][class_key]['bricks']
                brick_result = self._classify_single_level(product_name, bricks, "Brick")
                predicted_brick = brick_result["predicted_label"]
            else:
                brick_result = {"predicted_label": "", "raw_response": "Class key not found"}
                predicted_brick = ""
        else:
            brick_result = {"predicted_label": "", "raw_response": "No class found"}
            predicted_brick = ""
        
        # Show final classification path
        print(f"  ✅ Final: {predicted_segment} → {predicted_family} → {predicted_class} → {predicted_brick}")
        
        time.sleep(0.5)  # Rate limiting
        
        return {
            "product_name": product_name,
            "predicted_segment": predicted_segment,
            "predicted_family": predicted_family,
            "predicted_class": predicted_class,
            "predicted_brick": predicted_brick,
            "segment_response": segment_result["raw_response"],
            "family_response": family_result["raw_response"],
            "class_response": class_result["raw_response"],
            "brick_response": brick_result["raw_response"]
        }
    
    def predict_batch(self, products_df: pd.DataFrame, product_text_col: str = "product_name", save_interval: int = 10) -> pd.DataFrame:
        """Predict hierarchical classification for a batch of products with progress tracking and periodic saving"""
        import datetime
        from tqdm import tqdm
        
        results = []
        total_products = len(products_df)
        
        # Create timestamp for filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"gpc_classification_results_{timestamp}.csv"
        clean_csv_filename = f"gpc_classification_clean_{timestamp}.csv"
        
        print(f"🚀 Starting classification of {total_products} products...")
        print(f"📁 Results will be saved to: {csv_filename}")
        print(f"💾 Saving progress every {save_interval} products")
        
        # Use tqdm for progress bar
        for idx, row in tqdm(products_df.iterrows(), total=total_products, desc="Classifying products"):
            product_name = row[product_text_col]
            
            print(f"\n[{idx+1}/{total_products}] Processing: {product_name[:50]}...")
            
            result = self.classify_product_hierarchical(product_name)
            result["product_id"] = row.get("id", idx)
            results.append(result)
            
            # Periodic saving
            if (idx + 1) % save_interval == 0 or (idx + 1) == total_products:
                try:
                    # Save full results
                    temp_df = pd.DataFrame(results)
                    temp_df.to_csv(csv_filename, index=False)
                    
                    # Save clean results
                    clean_results = temp_df[['product_name', 'predicted_segment', 'predicted_family', 
                                        'predicted_class', 'predicted_brick']].copy()
                    clean_results.to_csv(clean_csv_filename, index=False)
                    
                    print(f"💾 Progress saved: {idx+1}/{total_products} products completed")
                    
                except Exception as e:
                    print(f"⚠️ Warning: Failed to save progress: {e}")
        
        results_df = pd.DataFrame(results)
        
        print(f"\n✅ Classification complete!")
        print(f"📊 Final results saved to: {csv_filename}")
        print(f"📋 Clean results saved to: {clean_csv_filename}")
        
        return results_df

@dataclass
class ICFTDCBModelConfig:
    ngram_range: tuple = (1, 2)
    min_df: int = 1
    k: int = 3
    class_name_col: str = "SegmentTitle"
    class_text_col: str = "SegmentDefinition"
    product_id_col: str = "id"
    product_text_col: str = "translated_name"


class ICFTDCBModel:
    
    def __init__(self, config: ICFTDCBModelConfig):
        self.config = config
        self.vectorizer = None
        self.class_centroids = None
        self.global_weights = None
        self.gpc_df = None
    
    def _prep_text(self, s: pd.Series) -> pd.Series:
        return s.fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    
    def fit(self, products_df: pd.DataFrame):
        products_df = products_df.copy()
        products_df[self.config.product_text_col] = self._prep_text(products_df[self.config.product_text_col])
        products_df[self.config.class_name_col] = self._prep_text(products_df[self.config.class_name_col])

        self.gpc_df = (
            products_df[[self.config.class_name_col]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        self.vectorizer = CountVectorizer(
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df
        )
        X_all = self.vectorizer.fit_transform(products_df[self.config.product_text_col])

        C = X_all.shape[0]
        cf = (X_all > 0).sum(axis=0).A1 + 1e-9
        ICF = np.log(C / cf)

        col_sums = np.asarray(X_all.sum(axis=0)).ravel() + 1e-9
        P = X_all.multiply(1.0 / col_sums)
        TDCB = 1.0 - np.asarray(P.power(2).sum(axis=0)).ravel()

        self.global_weights = np.maximum(ICF * TDCB, 1e-9)

        centroids = []
        class_names = self.gpc_df[self.config.class_name_col].tolist()
        for cname in class_names:
            mask = (products_df[self.config.class_name_col] == cname).to_numpy()
            if not mask.any():
                continue
            X_cls = X_all[mask]
            centroid = X_cls.mean(axis=0)      
            weighted_centroid = centroid @ diags(self.global_weights)
            centroids.append(np.asarray(weighted_centroid).ravel())  

        self.class_centroids = normalize(np.vstack(centroids), norm="l2", axis=1)

    def predict(self, products_df: pd.DataFrame) -> pd.DataFrame:
        # if self.vectorizer is None or self.class_centroids is None:
        #     raise ValueError("Model must be fitted before prediction")
        
        products_df = products_df.copy()
        products_df[self.config.product_text_col] = self._prep_text(products_df[self.config.product_text_col])
        
        X_prod = self.vectorizer.transform(products_df[self.config.product_text_col])
        V_prod = normalize(X_prod @ diags(self.global_weights), norm="l2", axis=1)
        
        S = (V_prod @ self.class_centroids.T)
        topk_idx = (-S).argsort(1)[:, :self.config.k]
        topk_scores = np.take_along_axis(S, topk_idx, axis=1)
        
        rows = []
        for i in range(S.shape[0]):
            base = {}
            if self.config.product_id_col in products_df.columns:
                base["product_id"] = products_df.loc[i, self.config.product_id_col]
            base["product_text"] = products_df.loc[i, self.config.product_text_col]

            best_class_idx = topk_idx[i,0]
            base["predicted_label_icf"] = self.gpc_df.reset_index(drop=True).iloc[best_class_idx][self.config.class_name_col]

            for j in range(self.config.k):
                base[f"score_{j+1}"] = float(topk_scores[i,j])
            rows.append(base)
        
        return pd.DataFrame(rows)


@dataclass
class TFIDFCentroidModelConfig:
    ngram_range: tuple = (1, 2)
    min_df: int = 1
    max_df: float = 0.95
    max_features: Optional[int] = None
    k: int = 3
    class_name_col: str = "SegmentTitle"
    class_text_col: str = "SegmentDefinition"
    product_id_col: str = "id"
    product_text_col: str = "translated_name"

class TFIDFCentroidModel:
    
    def __init__(self, config: TFIDFCentroidModelConfig):
        self.config = config
        self.vectorizer = None
        self.class_centroids = None
        self.gpc_df = None
    
    def _prep_text(self, s: pd.Series) -> pd.Series:
        return s.fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    
    def fit(self, product_df: pd.DataFrame):
        self.gpc_df = product_df.copy()
        self.gpc_df[self.config.class_name_col] = self._prep_text(
            self.gpc_df[self.config.class_name_col]
        )

        product_df = product_df.copy()
        product_df[self.config.product_text_col] = self._prep_text(
            product_df[self.config.product_text_col]
        )

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            max_features=self.config.max_features
        )

        X_products = self.vectorizer.fit_transform(product_df[self.config.product_text_col])


        centroids = []
        class_names = []
        for cname in self.gpc_df[self.config.class_name_col].unique():
            class_mask = (product_df[self.config.class_name_col] == cname).to_numpy()
            # class_mask = product_df[self.config.class_name_col] == cname
            if not class_mask.any():
                continue
            class_vecs = X_products[class_mask]
            centroid = class_vecs.mean(axis=0)
            centroid = np.asarray(centroid).ravel()
            centroids.append(centroid)
            class_names.append(cname)

        self.class_centroids = normalize(np.vstack(centroids), norm="l2", axis=1)
        self.gpc_df = pd.DataFrame({self.config.class_name_col: class_names})

    
    def predict(self, products_df: pd.DataFrame) -> pd.DataFrame:
        if self.vectorizer is None or self.class_centroids is None:
            raise ValueError("Model must be fitted before prediction")
        
        products_df = products_df.copy()
        products_df[self.config.product_text_col] = self._prep_text(products_df[self.config.product_text_col])
        
        X_prod = self.vectorizer.transform(products_df[self.config.product_text_col])
        V_prod = normalize(X_prod, norm="l2", axis=1)
        
        similarities = cosine_similarity(V_prod, self.class_centroids)
        topk_idx = (-similarities).argsort(1)[:, :self.config.k]
        topk_scores = np.take_along_axis(similarities, topk_idx, axis=1)
        
        rows = []
        for i in range(similarities.shape[0]):
            base = {}
            if self.config.product_id_col in products_df.columns:
                base["product_id"] = products_df.loc[i, self.config.product_id_col]
            base["product_text"] = products_df.loc[i, self.config.product_text_col]

            best_class_idx = topk_idx[i,0]
            base["predicted_label_idf"] = self.gpc_df.reset_index(drop=True).iloc[best_class_idx][self.config.class_name_col]

            for j in range(self.config.k):
                base[f"class_{j+1}_name"] = self.gpc_df.iloc[topk_idx[i,j]][self.config.class_name_col]
                base[f"score_{j+1}"] = float(topk_scores[i,j])
            rows.append(base)
        
        return pd.DataFrame(rows)

class EnsembleClassifier:
    
    def __init__(self, icf_config: ICFTDCBModelConfig,
                 tfidf_config: TFIDFCentroidModelConfig,
                 embedding_model: SentenceEmbeddingModel,
                 llm_model: LLMModel):
        self.icf_model = ICFTDCBModel(icf_config)
        self.tfidf_model = TFIDFCentroidModel(tfidf_config)
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        self.gpc_df = None
        self.segment_embeddings = None
        self.segment_labels = None
    
    def fit(self, gpc_df: pd.DataFrame, segment_col: str = "SegmentTitle", product_col : str = "translated_name"):
        self.gpc_df = gpc_df.copy()
        
        unique_segments = gpc_df[segment_col].unique()
        self.segment_labels = unique_segments.tolist()
        product_names =  gpc_df[product_col].tolist()
        
        segment_gpc = pd.DataFrame({
            'class_id': range(len(unique_segments)),
            'class_name': unique_segments,
            'class_text': unique_segments,
            'translated_name': product_names,
        })
        
        # self.icf_model.fit(segment_gpc)
        self.tfidf_model.fit(segment_gpc)
        
        self.segment_embeddings = self.embedding_model.get_embeddings(
            self.segment_labels, 
            prompt_name="classification"
        )
    
    def _cosine_similarity_predict(self, products_df: pd.DataFrame, 
                                   product_text_col: str = "translated_name") -> pd.DataFrame:
        product_texts = products_df[product_text_col].tolist()
        product_embeddings = self.embedding_model.get_embeddings(
            product_texts, 
            prompt_name="classification"
        )
        
        similarities = cosine_similarity(product_embeddings, self.segment_embeddings)
        
        results = []
        for i, row in products_df.iterrows():
            best_idx = np.argmax(similarities[i])
            predicted_label = self.segment_labels[best_idx]
            
            result = {
                "product_id": row.get("id", i),
                "product_text": row[product_text_col],
                "predicted_label_cosine": predicted_label,
                "confidence": float(similarities[i][best_idx])
            }
            results.append(result)
        
        return pd.DataFrame(results)
    
    def predict_ensemble(self, products_df: pd.DataFrame, 
                        product_text_col: str = "translated_name",
                        voting_strategy: str = "majority") -> pd.DataFrame:
        
        #icf_predictions = self.icf_model.predict(products_df)
        tfidf_predictions = self.tfidf_model.predict(products_df)
        cosine_predictions = self._cosine_similarity_predict(products_df, product_text_col)
        llm_predictions = self.llm_model.predict(products_df, self.segment_labels, product_text_col)
        
        ensemble_results = []
        
        for i in range(len(products_df)):
            product_id = products_df.iloc[i].get("id", i)
            product_text = products_df.iloc[i][product_text_col]
            
            predictions = {
                # "icf": icf_predictions.iloc[i]["predicted_label_icf"],
                "tfidf": tfidf_predictions.iloc[i]["predicted_label_idf"],
                "cosine": cosine_predictions.iloc[i]["predicted_label_cosine"], 
                "llm": llm_predictions.iloc[i]["predicted_label_llm"]
            }
            
            if voting_strategy == "majority":
                votes = list(predictions.values())
                vote_counts = Counter(votes)
                final_prediction = vote_counts.most_common(1)[0][0]
                confidence = vote_counts.most_common(1)[0][1] / len(votes)
                if len(vote_counts) == 3:
                    final_prediction = predictions["llm"]
                    confidence = 1.0
            
            elif voting_strategy == "weighted":
                # icf_weight = 0.3
                tfidf_weight = 0.3
                cosine_weight = 0.4
                llm_weight = 0.3
                
                label_scores = {}
                weights_list = [tfidf_weight, cosine_weight, llm_weight]
                for pred, weight in zip(predictions.values(), weights_list):
                    if pred not in label_scores:
                        label_scores[pred] = 0
                    label_scores[pred] += weight
                
                final_prediction = max(label_scores.keys(), key=lambda x: label_scores[x])
                confidence = label_scores[final_prediction]
            
            else:
                final_prediction = predictions["llm"]
                confidence = 1.0
            
            result = {
                # "product_id": product_id,
                "product_text": product_text,
                "llm_prediction": predictions["llm"],
                # "final_prediction": final_prediction,
                # "confidence": confidence,
                # #"icf_prediction": predictions["icf"],
                # "tfidf_prediction": predictions["tfidf"],
                # "cosine_prediction": predictions["cosine"],
            }
            
            ensemble_results.append(result)
        
        return pd.DataFrame(ensemble_results)



@dataclass
class LogisticRegressionConfig:
    max_iter: int = 1000
    solver: str = "lbfgs"
    random_state: int = 42
    C: float = 1.0  
    n_jobs: int = -1


class WeightedLogisticRegressionClassifier:
    def __init__(self, config: LogisticRegressionConfig, 
                 special_class_weights: Optional[Dict[str, float]] = None,
                 default_weight: float = 1.0,
                 use_balanced: bool = False):
        self.config = config
        self.special_class_weights = special_class_weights or {}
        self.default_weight = default_weight
        self.use_balanced = use_balanced
        self.class_weight_dict = None
        self.model = None
    
    
    def _create_class_weights(self, y):
        if self.use_balanced:
            return "balanced"
        
        unique_classes = np.unique(y)
        weights = {}
        
        for cls in unique_classes:
            cls_lower = str(cls).lower().strip()
            special_weight = None
            
            for special_cls, weight in self.special_class_weights.items():
                if special_cls.lower() in cls_lower or cls_lower in special_cls.lower():
                    special_weight = weight
                    break
            
            weights[cls] = special_weight if special_weight is not None else self.default_weight
        
        return weights
    
    def fit(self, X, y):
        self.class_weight_dict = self._create_class_weights(y)
        
        self.model = LogisticRegression(
            max_iter=self.config.max_iter,
            solver=self.config.solver,
            random_state=self.config.random_state,
            C=self.config.C,
            n_jobs=self.config.n_jobs,
            class_weight=self.class_weight_dict
        )
        
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
    
    def get_model(self):
        return self.model
    

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
    food_weight: float = 5.0


class TfidfClassifier:

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

        self.clf = None

    def fit(self, X_train, y_train):
        self.clf = Pipeline(
            [
                ("vectorizer_tfidf", self.vectorizer),
                ("random_forest", RandomForestClassifier())
            ]
        )
        self.clf.fit(X_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def predict_topk(self, product_name, k=3):

        probabilities = self.clf.predict_proba([product_name])[0]

        classes = self.clf.classes_

        prob_class_pairs = list(zip(probabilities, classes))

        prob_class_pairs.sort(key=lambda x: x[0], reverse=True)

        return [class_label for prob, class_label in prob_class_pairs[:k]]

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
    
@dataclass
class DummyModelConfig:
    strategy: str


class DummyModel:

    def __init__(self, config: DummyModelConfig):
        self.model = DummyClassifier(strategy=config.strategy)

    def fit_model(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
            