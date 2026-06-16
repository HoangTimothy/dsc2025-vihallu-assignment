# DSC2025 ViHallu - Collaborative NLP Framework

This repository provides a flexible, modular framework designed to benchmark, evaluate, and deploy hallucination detection models on the **ViHallu** dataset (Vietnamese Hallucination Detection). It supports classical ML baselines, BERT-based sequence classification, parameter-efficient fine-tuning (PEFT/LoRA) on Large Language Models, semantic uncertainty estimation, layer-wise probing, and agentic multi-stage reasoning.

---

## Directory Structure
```text
.
├── src/vihallu/
│   ├── __init__.py
│   ├── config.py              # Constants, label mapping (no, intrinsic, extrinsic)
│   ├── data.py                # Data ingestion, stratification, and split logic
│   ├── features.py            # Rule-based overlap, negation & extrinsic features
│   ├── metrics.py             # Evaluation score computations & reports
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classical.py       # TF-IDF + Logistic Regression baseline
│   │   ├── transformer.py     # PhoBERT/XLM-RoBERTa training & evaluation wrapper
│   │   ├── hybrid.py          # SLSQP ensemble weight blending & rule-based probability
│   │   └── advanced.py        # Semantic Entropy, HaluAgent, MHAD, & LoRA Ensemble
│   └── utils/
│       ├── io_utils.py        # Safe read/write helpers
│       └── text_utils.py      # Unicode normalization & word tokenizers (pyvi/underthesea)
├── scripts/
│   ├── preprocess.py          # Multimodal text preprocessing CLI
│   ├── train_classical.py     # Classical model trainer
│   ├── train_transformer.py   # Transformer model trainer
│   ├── train_hybrid.py        # SLSQP probability ensembling script
│   ├── benchmark_transformers.py # Multi-hyperparameter search CLI
│   └── predict.py             # Public test inference script
├── requirements.txt           # Dependency specifications
├── pyproject.toml             # Python build config & local package settings
├── vihallu-train.csv          # Training dataset (Context, Prompt, Response, Label)
└── vihallu-public-test.csv    # Evaluation dataset (Context, Prompt, Response)
```

---

## 1. Exploratory Data Analysis (EDA)

Before running preprocessing, perform EDA to inspect data distributions, text lengths, and semantic overlaps.

### Key Analysis Metrics & Formulas
- **Class Distribution**: Proportions of `no` (non-hallucinated), `intrinsic` (hallucination based on contradicting context), and `extrinsic` (hallucination introducing outside, unverifiable information) labels.
- **Sequence Lengths**: Character and token lengths of `context`, `prompt`, and `response`.
- **Lexical Overlap**: Jaccard similarity and token overlap rates between text columns:
  $$\text{Overlap}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
- **Negation Rate**: Occurrence rates of negation cue tokens (e.g., *không*, *chưa*, *chẳng*, *khỏi*, *chả*).

### Setup and Parameters
There is no standalone script for EDA; data stats are generated dynamically as a JSON log metadata sidecar during preprocessing. 
- Output location: `outputs/eda/preprocess_stats.json`
- Output parameters saved: `rows`, `columns`, and `sample_rows`.

---

## 2. Preprocessing Strategies

The preprocessing pipeline cleans, tokenizes, and formats input rows into optimized structures matching the downstream architecture's expectations.

### Preprocessing Modes
1. **Traditional & BERT-based (`baseline`, `phobert`):**
   - Normalizes unicode text to NFC form.
   - Cleans HTML tags, URLs, and emails.
   - Word segments the text using `pyvi` or `underthesea` (converting multi-word tokens to `word_segment` with underscores, e.g., `trí_tuệ_nhân_tạo`).
   - Lowercases tokens and serializes via: `[CLS] context [SEP] prompt [SEP] response [SEP]`.
2. **Subword-based Transformers (`xlm-roberta`):**
   - Skips manual word segmentation to let the tokenizer apply BPE/SentencePiece directly.
   - Normalizes string encodings and serializes using the same sequence formatting.
3. **Generative LLM formatting (`llm`, `semantic_entropy`, `mhad`, `halu_agent`):**
   - Extracts raw cleaned columns.
   - Packages context, prompt, and response into a conversational instruction prompt template.

### Preprocessing Configuration (CLI Parameters)
To execute the preprocess pipeline:
```bash
python scripts/preprocess.py [ARGS]
```

| Parameter | Type | Default | Choices | Description |
| :--- | :---: | :---: | :---: | :--- |
| `--input`, `-i` | `str` | `vihallu-train.csv` | - | Path to the raw input dataset CSV. |
| `--output`, `-o` | `str` | `outputs/eda/preprocessed_train.csv` | - | Target output path for the processed CSV. |
| `--sample`, `-s` | `int` | `0` | - | Optional subset limit size. Set to `0` to process the entire file. |
| `--method`, `-m` | `str` | `baseline` | `baseline`, `phobert`, `xlm-roberta`, `llm`, `semantic_entropy`, `mhad`, `halu_agent`, `hybrid` | Preprocessing strategy based on downstream architecture requirements. |

---

## 3. Training & Inference Strategies

This framework benchmarks classical baselines against transformer-based and parameter-efficient LLM-based hallucination detection methods.

### Strategy 3.1: TF-IDF + Baseline ML
A classical baseline pairing text representation with linear classifiers to establish a low-compute reference performance.

#### Vectorizer Configuration (TF-IDF)
- **N-grams**: Unigrams and Bigrams (`ngram_range=(1, 2)`) to capture word order.
- **Frequency Filtering**: Minimum document frequency filter (`min_df=2`) to remove tail noise.
- **Maximum Features**: Capped at `120,000` to prevent memory bottlenecks.
- **Scaling**: Sublinear Term Frequency (`sublinear_tf=True`) utilizing $1 + \log(\text{tf})$ scaling.
- **Case**: `lowercase=False` (retains sentence structures post word segmentation).

#### Classifier Setup (Logistic Regression)
- **Loss Solver**: Limited-memory BFGS (`solver='lbfgs'`).
- **Iterations**: `max_iter=2500` to guarantee convergence.
- **Regularization Strength**: $C = 2.0$ (modest L2 regularization).
- **Class Balancing**: Balanced weights (`class_weight='balanced'`) to adjust for label imbalances.

#### CLI Run Setup:
```bash
python scripts/train_classical.py \
  --train_path vihallu-train.csv \
  --output_dir outputs/classical \
  --valid_size 0.15 \
  --seed 42
```

---

### Strategy 3.2: BERT-based Transformers
Fine-tunes encoder-only models (`vinai/phobert-base-v2` and `xlm-roberta-base`) as three-way sequence classifiers.

#### Tokenization & Hyperparameters
- **Max Input Length**: `256` or `512` tokens. Long inputs are truncated from the tail.
- **Learning Rate**: AdamW with `learning_rate=2e-5` and standard `weight_decay=0.01`.
- **Validation Splitting**: Stratified split preserving class frequencies, using $15\%$ of training data.
- **Best Model Selection**: Saves checkpoint showing the highest **Macro-F1** score on validation.

#### Single-Model CLI Setup:
```bash
python scripts/train_transformer.py \
  --train_path vihallu-train.csv \
  --output_dir outputs/transformer \
  --model_name vinai/phobert-base-v2 \
  --epochs 3 \
  --train_batch_size 8 \
  --eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --max_length 256 \
  --text_mode auto \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --valid_size 0.15 \
  --seed 42
```

#### Multi-Model Benchmark Search CLI Setup:
```bash
python scripts/benchmark_transformers.py \
  --train_path vihallu-train.csv \
  --output_dir outputs/transformer_benchmark \
  --model_names vinai/phobert-base-v2 xlm-roberta-base \
  --max_lengths 256 512 \
  --epochs 3 \
  --train_batch_size 8 \
  --eval_batch_size 16 \
  --learning_rate 2e-5 \
  --skip_done
```

---

### Strategy 3.3: Ensemble Fine-tuning (LoRA on Qwen 2.5 / Qwen 3 Instruct)
Adapts causal Large Language Models for classification by applying Low-Rank Adaptation (LoRA) over multiple initialized instances.

#### PEFT LoRA Config:
- **Base Model Configuration**: Default `"Qwen/Qwen3-4B-Instruct"` (or `"Qwen/Qwen2.5-3B-Instruct"`).
- **Target Projection Layers**: `["q_proj", "k_proj", "v_proj", "o_proj"]` to maximize parameter efficiency.
- **LoRA Hyperparameters**:
  - Rank ($r$): `8` (captures primary low-rank directions).
  - Scaling factor ($\alpha$): `16`.
  - Dropout probability: `0.05`.
  - Task Type: Causal Language Modeling (`TaskType.CAUSAL_LM`).
- **Hardware Profile**: CPU default inference and training mode (`device_map="cpu"`), utilizing `torch.float32`.
- **Ensemble Setup**: Runs inference on $N$ independently trained LoRA heads and aggregates generations.

```python
# Initialization example from advanced.py:
from vihallu.models.advanced import setup_qwen_lora_ensemble

ensemble, tokenizer = setup_qwen_lora_ensemble(
    model_name="Qwen/Qwen3-4B-Instruct",
    num_models=3,
    r=8,
    lora_alpha=16
)
```

---

### Strategy 3.4: Semantic Entropy
Estimates generative uncertainty to detect hallucinations without needing external validation tools, mapping the diversity of multiple output predictions.

#### Parameters & Computation Setup
- **Sample Generation Count ($M$)**: Generates $M$ responses (typically `5` to `10`) using high-temperature decoding ($T \ge 0.7$).
- **Semantic Encoder**: Sentence Transformer model (e.g., `sentence-transformers/LaBSE` or a Vietnamese equivalent).
- **Similarity Threshold ($\tau$)**: Capped at `0.8` cosine similarity.
- **Clustering Rule**: Greedy clustering. If the cosine similarity between an output embed and a cluster center is $\ge \tau$, it joins that cluster. Otherwise, a new cluster is initialized.
- **Entropy Formula**:
  $$H_{\text{semantic}} = -\sum_{c \in \text{clusters}} P(c) \log\left(P(c) + \epsilon\right)$$
  Where $P(c) = \frac{|N_c|}{M}$ represents the cluster probability mass, and $\epsilon = 10^{-9}$ prevents logarithmic division by zero.

---

### Strategy 3.5: MHAD Probing (Model Hallucination Awareness)
Leverages the internal representations of an LLM across layers to identify state variations linked to hallucinated vs. factual claims.

#### Architecture and Neuron Selection Setup
- **Hidden Dimensions**: $d_{\text{model}}$ matches base LLM hidden states (e.g., `4096`).
- **Layers Profile**: Set across all $L$ layers of the target network (e.g., layer `0` to layer `31`).
- **Probing Architecture**: 
  - A sequence of two Linear Probe layers per network layer.
  - Structure: `Linear(hidden_dim, hidden_dim // 2)` $\to$ `ReLU()` $\to$ `Linear(hidden_dim // 2, 1)` $\to$ `Sigmoid()`.
  - Trained to classify hallucinated states at initial generation steps ($t=1$) and final generation steps ($t=T$).
- **Variance Filtering ($\alpha$)**: Neurons are ranked by their absolute mean weight representation $W_{\text{abs}}$.
  We select the subset of indices $i$ that contribute to the top cumulative sum of squared weights matching threshold $\alpha$:
  $$\sum_{j=1}^{K} (w_{\text{sorted}, j})^2 \ge \alpha \cdot \sum_{j=1}^{d_{\text{model}}} (w_{j})^2 \quad \text{where } \alpha = 0.9$$
- **HAV Construction**: Concat output layers of selected neurons from initial and final steps into the Hallucination Awareness Vector (HAV):
  $$\text{HAV} = \text{Concat}(\{\mathbf{h}_{\text{init}, l}[:, N_{\text{init}, l}], \mathbf{h}_{\text{final}, l}[:, N_{\text{final}, l}]\}_{l \in \text{selected\_layers}})$$

---

### Strategy 3.6: HaluAgent with Bottleneck Rate Limiting
An autonomous, multi-stage agentic workflow that breaks down responses into individual claims, cross-checks them using external resources, and applies reflection logic.

#### Processing Steps and Parametric Control
1. **Sentence Segmenter**: Segments target inputs by period (`.`) or standard sentence tokenizers.
2. **Tool-based Verification**: Runs claims against external tools (APIs, Google Search, local databases).
3. **Reflective Prompting**: Formats claims and verification transcripts into a reasoning template:
   ```text
   Claim: {claim}
   Evidence: {evidence}
   Based on the evidence, is the claim a hallucination? Think step-by-step.
   ```
4. **Bottleneck Rate Limiting**: Capping request velocity to stay within API resource bounds (e.g., limits external API requests to `60 RPM` or introduces a sleep delay of `1.0s` between sentence checks).

```python
# Script execution initialization block:
from vihallu.models.advanced import HaluAgent

agent = HaluAgent(llm_pipeline=my_llm_pipeline, external_tools=my_tools)
results = agent.detect("Vịnh Hạ Long nằm ở miền Nam Việt Nam. Nơi đây có hơn 1,600 đảo đá vôi.")
```

---

## 4. Evaluation Metrics & Blending Calibration

The final model selection and combination uses a hybrid, feature-weighted ensembling setup calibrated on validation data.

### Evaluation Protocol
- **Primary Metric**: **Macro-averaged F1-Score** across classes: `no`, `intrinsic`, and `extrinsic`.
- **Secondary Metrics**: Accuracy, Confusion Matrix, and precision-recall metrics per class.

### SLSQP Ensemble Blending
Combines prediction probabilities from the Classical Model ($P_{\text{class}}$), BERT Transformer ($P_{\text{trans}}$), and Rule-based features ($P_{\text{rule}}$):
$$P_{\text{blend}} = w_{\text{class}} \cdot P_{\text{class}} + w_{\text{trans}} \cdot P_{\text{trans}} + w_{\text{rule}} \cdot P_{\text{rule}}$$
Where:
- $w_{\text{class}}, w_{\text{trans}}, w_{\text{rule}} \in [0.0, 1.0]$
- $w_{\text{class}} + w_{\text{trans}} + w_{\text{rule}} = 1.0$

The weights are optimized on the validation set using Sequential Least Squares Programming (**SLSQP**):
$$\min_{w} -\text{Macro\_F1}(y_{\text{true}}, \arg\max(P_{\text{blend}}))$$

### Blending Calibration CLI Run Setup:
```bash
python scripts/train_hybrid.py \
  --train_path vihallu-train.csv \
  --classical_dir outputs/classical \
  --transformer_dir outputs/transformer \
  --output_dir outputs/hybrid \
  --valid_size 0.15 \
  --seed 42
```

---

## Quick Reference CLI Commands

### 1. Preprocess raw data for PhoBERT
```bash
python scripts/preprocess.py -i vihallu-train.csv -o outputs/preprocessed_phobert.csv -m phobert
```

### 2. Preprocess raw data for LLMs / Advanced Methods
```bash
python scripts/preprocess.py -i vihallu-train.csv -o outputs/preprocessed_llm.csv -m llm
```

### 3. Train Classical Baseline
```bash
python scripts/train_classical.py --train_path vihallu-train.csv --output_dir outputs/classical
```

### 4. Train PhoBERT Baseline
```bash
python scripts/train_transformer.py --train_path vihallu-train.csv --output_dir outputs/transformer --model_name vinai/phobert-base-v2 --max_length 256
```

### 5. Train Hybrid Blended Ensemble
```bash
python scripts/train_hybrid.py --train_path vihallu-train.csv --classical_dir outputs/classical --transformer_dir outputs/transformer --output_dir outputs/hybrid
```

### 6. Generate Public Test Predictions
```bash
python scripts/predict.py --test_path vihallu-public-test.csv --model_type hybrid --model_dir outputs/hybrid --output_path outputs/submission_public.csv
```
