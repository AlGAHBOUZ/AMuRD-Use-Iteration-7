import torch
import torch.nn.functional as F
import pandas as pd
import chardet
import unicodedata

import re
import json
from typing import List


from constants import ALL_STOPWORDS, ALL_BRANDS, GPC_PATH, OPUS_TRANSLATION_CONFIG_PATH, PROMPT_PATH
from modules.models import (
    SentenceEmbeddingModel, 
    SentenceEmbeddingConfig,
    OpusTranslationModel,
    OpusTranslationModelConfig,
    LLMModel, 
    LLMModelConfig,
    HierarchicalGPCClassifier
)

def remove_repeated_words(text):
    text = text.split()
    final_text = []
    for word in text:
        if word in final_text:
            continue
        final_text.append(word)

    return " ".join(final_text)

def remove_brand_name(text: str) -> str:
    for brand in ALL_BRANDS:
        if brand in text:
            text = text.replace(brand, "")
            break

    return text

def remove_strings(text: str, strings: List[str]) -> str:
    for s in strings:
        s = str(s)
        if s in text:
            text = text.replace(s, "")

    return text

def remove_numbers(text: str, remove_string: bool = False) -> str:
    text = text.split()
    text = [t for t in text if not re.search(r"\d", t)] if remove_string else [re.sub(r"\d+", "", t) for t in text]

    return " ".join(text)

def remove_stopwords(text: str):
    text = text.split()
    text = [t for t in text if t not in ALL_STOPWORDS or t == "can"]

    return " ".join(text)

def remove_punctuations(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", text)

    return " ".join(text.strip().split())

def remove_special_chars(text: str) -> str:
    text = re.sub(r"[-_/\\|]", " ", text)  

    return " ".join(text.strip().split()).lower()

def remove_extra_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def clean_text(text) -> str:
    # text = row["Item_Name"]
    # brand = row["Brand"]
    # text = remove_strings(text, [brand])
    text = remove_punctuations(text)
    text = remove_numbers(text)
    text = remove_extra_space(text)
    # text = remove_brand_name(text)
    # text = remove_stopwords(text)
    # if unit not in text and text == "":
        # text += unit

    return text.lower()

def load_embedding_model(config_path: str):
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    try:
        config = SentenceEmbeddingConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid configuration keys: {e}.")

    model = SentenceEmbeddingModel(config)

    return model

def load_llm_model(config_path: str):
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    try:
        config = LLMModelConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid configuration keys: {e}.")

    model = LLMModel(config, PROMPT_PATH)

    return model

def load_HierarchicalGPCClassifier(config_path: str, gpc_data_df):
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    try:
        config = LLMModelConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid configuration keys: {e}.")

    model = HierarchicalGPCClassifier(config, PROMPT_PATH, gpc_data_df)

    return model

def load_translation_model(config_path: str):
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    try:
        config = OpusTranslationModelConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid configuration keys: {e}.")
    
    model = OpusTranslationModel(config)

    return model

def join_non_empty(*args):
    return " ".join([str(a).strip() for a in args if pd.notna(a) and str(a).strip()])
    

def load_gpc_to_classes():
    df = pd.read_excel(GPC_PATH)

    df["class_name"] = (
        df["SegmentTitle"].fillna("") + " " +
        df["FamilyTitle"] + " " + 
        df["ClassTitle"] + " " +
        df["BrickTitle"].fillna("")
    )

    df["class_name_cleaned"] = df["class_name"].apply(remove_repeated_words)

    df["description"] = df.apply(lambda row: join_non_empty(
        row["BrickDefinition_Includes"],
        row["BrickDefinition_Excludes"],
    ), axis=1)

    df_new = df[["class_name", "class_name_cleaned", "description"]]

    return df_new

def cluster_topk_classes(cluster_embeddings: List[List[float]], classes_embeddings: List[List[float]], k: int) -> torch.Tensor:
    cluster_embeddings = F.normalize(cluster_embeddings, p=2, dim=1)
    classes_embeddings = F.normalize(classes_embeddings, p=2, dim=1)

    scores = (cluster_embeddings @ classes_embeddings.T)

    topk_classes = torch.topk(scores, k=k)

    return topk_classes[1]
 
def detect_file_encoding(file_path, n_bytes=100000):
    with open(file_path, "rb") as f:
        raw_data = f.read(n_bytes)
        result = chardet.detect(raw_data)
    return result

def unicode_clean(s):
    if not isinstance(s, str):
        return s
    s = unicodedata.normalize('NFKC', s)
    s = ''.join(c for c in s if unicodedata.category(c)[0] != 'C') 
    return s.strip()
