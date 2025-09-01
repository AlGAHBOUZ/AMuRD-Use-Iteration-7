# Classi-Fy
An automated product classification system using multilingual embeddings and semantic similarity for Arabic and English product names.

## Overview
The Multilingual Product Classification System is an AI-powered solution that automatically classifies products based on their names using semantic embeddings. The application leverages advanced multilingual embedding models to handle both Arabic and English product names, translates them when necessary, and performs classification using cosine similarity in vector space.

Key features:
- **Multilingual support** - Handles Arabic and English product names seamlessly
- **Neural translation** - Automatic Arabic to English translation using OPUS-MT models
- **Semantic embeddings** - Uses state-of-the-art embedding model (E5-Large-Instruct)
- **Vector similarity search** - Efficient classification using cosine similarity in Teradata
- **Database integration** - Full pipeline integration with Teradata for scalable processing
- **Performance metrics** - F1-score calculation and evaluation

## Technology Stack
- **Frontend**: Python-based pipeline with streamlit
- **Backend**: Python - Core application logic and model inference
- **AI Models**: 
  - Multilingual E5-Large-Instruct (0.6B) - For text embeddings
  - Helsinki-NLP OPUS-MT (0.2B) - For Arabic to English translation
- **Database**: Teradata - Enterprise data warehouse with vector operations
- **Model Optimization**: PyTorch, transformers, bitsandbytes - For model loading
- **ML Framework**: sentence-transformers, scikit-learn
  
## Important Files:
- **Core Pipeline**: `main.py` - Main processing pipeline with database operations
- **Model Definitions**: `modules/models.py` - AI model classes and configurations
- **Database Operations**: `modules/db.py` - Teradata connection and query handling
- **Utilities**: `utils.py` - Text processing and model loading utilities
- **Constants**: `constants.py` - Path definitions and configuration constants
- **Model Configs**: 
  - `e5_large_instruct_config.json` - E5 embedding model configuration
  - `opus_translation_config.json` - Translation model configuration

## Setup for Development

### Conda Environment
Create a new Conda environment:
```bash
conda create --name Classi-Fy
```

Install pip and project dependencies:
```bash
pip install -r requirements.txt
```

### GPU Configuration
1. **Download and install the Nvidia driver** appropriate for your GPU

2. **Install the CUDA toolkit**:
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Follow the installation instructions

3. **Install CUDA deep learning package (cuDNN)**:
   - Download from: https://developer.nvidia.com/cudnn-downloads
   - Extract and follow installation instructions

4. **Set up PyTorch with CUDA support**:
   ```bash
   # In your Conda environment
   pip uninstall torch torchvision torchaudio -y
   pip install torch --index-url https://download.pytorch.org/whl/cu126
   ```

5. **Verify CUDA installation**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA device count: {torch.cuda.device_count()}")
   print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
   ```

### Mac GPUs (Apple Silicon or Metal-compatible Intel)
Ensure PyTorch 2.0+ is installed:
```bash
pip install --upgrade torch
```

## Database Setup

1. **Create a Teradata account** on the Clearscape Analytics platform: https://clearscape.teradata.com/

2. **Configure database credentials** in `.env`:
   ```env
   TD_HOST=your-teradata-host.com
   TD_NAME=your-database-name
   TD_USER=your-username
   TD_PASSWORD=your-password
   TD_PORT=1025
   TD_AUTH_TOKEN=your-auth-token
   ```

## How to Test

### Test Database Connection
```bash
python tests/test_db.py
```

### Test Model Loading
```python
from utils import load_embedding_model, load_translation_model
from constants import E5_LARGE_INSTRUCT_CONFIG_PATH, OPUS_TRANSLATION_CONFIG_PATH

# Test embedding model
embedding_model = load_embedding_model(E5_LARGE_INSTRUCT_CONFIG_PATH)
print("✅ Embedding model loaded")

# Test translation model  
translation_model = load_translation_model(OPUS_TRANSLATION_CONFIG_PATH)
print("✅ Translation model loaded")
```

## How to Run

### Full Pipeline Mode (Default)
```python
python main.py
```

This runs the complete classification pipeline:
1. **Data Insertion**: Load products and classes into Teradata
2. **Text Cleaning**: Clean and normalize product/class names
3. **Translation**: Translate Arabic products to English
4. **Embedding Generation**: Create vector embeddings for products and classes
5. **Classification**: Perform similarity-based classification using Teradata vector functions
6. **Evaluation**: Calculate F1-score and display results

### Individual Pipeline Steps

You can also run individual components by uncommenting specific functions in `main()`:

```python
# Insert data only
insert_to_db()

# Translation and embeddings only  
translate_products()
create_product_embeddings()

# Classification only
calculate_in_db_similarity()
```

## Pipeline Architecture

### Data Flow:
1. **Input**: Product names in Arabic/English + Class labels
2. **Cleaning**: Text normalization and preprocessing
3. **Translation**: Arabic → English using OPUS-MT
4. **Embedding**: Text → 1024D vectors using E5
5. **Storage**: Vectors stored in Teradata with optimized indexing
6. **Classification**: Cosine similarity search using TD_VectorDistance
7. **Evaluation**: F1-score calculation and performance metrics

### Model Configurations:

**E5-Large-Instruct**:
- Model: `intfloat/multilingual-e5-large-instruct`
- Dimensions: 1024 (configurable)
- Languages: 100+ including Arabic and English
- Memory: ~2.4GB GPU VRAM


## Performance Metrics

The system automatically calculates:
- **F1-Score**: Weighted average across all classes
- **Similarity Scores**: Cosine similarity confidence scores

## Configuration Options

### Embedding Models:
- Switch between E5 by updating config paths in `constants.py`

### Database Settings:
- Batch processing sizes configurable for large datasets
- Vector similarity thresholds adjustable
- Memory optimization settings for large embedding tables

