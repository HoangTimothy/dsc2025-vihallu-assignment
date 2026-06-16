# DSC2025 ViHallu - Collaborative NLP Framework

## 1) Framework Objectives
- Executable right out-of-the-box on the dataset provided in the folder.
- Facilitate comparison across multiple approaches: Classical, Transformer, and Hybrid.

## 2) Directory Structure
```text
.
├─ src/vihallu/
│  ├─ data.py
│  ├─ features.py
│  ├─ metrics.py
│  ├─ config.py
│  ├─ models/
│  │  ├─ classical.py
│  │  ├─ transformer.py
│  │  └─ hybrid.py
│  └─ utils/
│     ├─ io_utils.py
│     └─ text_utils.py
├─ scripts/
│  ├─ train_classical.py
│  ├─ train_transformer.py
│  ├─ train_hybrid.py
│  └─ predict.py
├─ requirements.txt
├─ vihallu-train.csv
└─ vihallu-public-test.csv
```

## 3) Design: From Baseline to SOTA-ish

### A. Classical baseline
- Input serialize theo paper: `[CLS] context [SEP] prompt [SEP] response [SEP]`
- TF-IDF + Logistic Regression
- Fast and easy to debug; serves as the initial baseline for comparison.

### B. Transformer baseline (PhoBERT or mBERT)
- Fine-tune the encoder model for 3 classes: `no`, `intrinsic`, `extrinsic`
- Loss: CrossEntropy, Primary metric : macro-F1.

### C. Hybrid (điểm "mới nhẹ")
- Combine prediction probabilities from Classical + Transformer models.
- Incorporate rule-based evidence scores (e.g., token overlap, contradiction cues, unsupported cues).
- Optimize ensemble weights on the validation set using SLSQP (inspired by top-performing competitive teams).

## 4) Quick Setup 
```bash
pip install -r requirements.txt
pip install -e .
```

## 5) How to Run

### 5.1 Classical
```bash
python scripts/train_classical.py --train_path vihallu-train.csv --output_dir outputs/classical
```

### 5.2 Transformer
```bash
python scripts/train_transformer.py --train_path vihallu-train.csv --output_dir outputs/transformer --model_name vinai/phobert-base
```

### 5.3 Hybrid
```bash
python scripts/train_hybrid.py \
  --train_path vihallu-train.csv \
  --classical_dir outputs/classical \
  --transformer_dir outputs/transformer \
  --output_dir outputs/hybrid
```

### 5.4 Predict public test
```bash
python scripts/predict.py \
  --test_path vihallu-public-test.csv \
  --model_type hybrid \
  --model_dir outputs/hybrid \
  --output_path outputs/submission_public.csv
```

## 6) Task Division
- Person A: data pipeline + classical + error analysis.
- Person B: transformer fine-tuning + prompt/noise augmentation.
- Cùng làm: Hybrid stacking + ablation study report.

## 7) Roadmap progress
1. Baseline classical 
2. PhoBERT fine-tune
3. Feature engineering + calibration
4. Ensemble/hybrid + ablation 
5. Error analysis across the 3 hallucination types + final report

## 8) Novel Ideas & Enhancements
- Dynamic weighting based on prompt noise/adversarial types (estimated from prompt features).
- Evidence consistency scoring at the token overlap and negation conflict level.
- Confidence-aware fallback: If the transformer's entropy is high, dynamically increase the weight of the classical/rule-based predictions.

Note: Use the `basic_classifier.ipynb` notebook to quickly prototype ideas before standardizing them into the main scripts.
