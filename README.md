# DSC2025 ViHallu - Collaborative NLP Framework

## 1) Mục tiêu framework
- Chạy được ngay trên dataset có sẵn trong folder.
- So sánh nhiều hướng: Classical, Transformer, Hybrid.

## 2) Cấu trúc thư mục
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

## 3) Thiết kế baseline -> SOTA-ish

### A. Classical baseline
- Input serialize theo paper: `[CLS] context [SEP] prompt [SEP] response [SEP]`
- TF-IDF + Logistic Regression
- Nhanh, dễ debug, tạo baseline đầu tiên để so sánh.

### B. Transformer baseline (PhoBERT hoặc mBERT)
- Fine-tune mô hình encoder cho 3 lớp: `no`, `intrinsic`, `extrinsic`
- Loss: CrossEntropy, metric chính: macro-F1.

### C. Hybrid (điểm "mới nhẹ")
- Kết hợp xác suất từ Classical + Transformer.
- Thêm rule-based evidence score (overlap, contradiction cue, unsupported cue).
- Tối ưu trọng số ensemble trên validation bằng SLSQP (ý tưởng gần hướng top đội thi).

## 4) Setup nhanh
```bash
pip install -r requirements.txt
pip install -e .
```

## 5) Cách chạy

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

## 6) Chia việc 
- Person A: data pipeline + classical + error analysis.
- Person B: transformer fine-tuning + prompt/noise augmentation.
- Cùng làm: hybrid stacking + báo cáo so sánh ablation.

## 7) Roadmap progress
1. Baseline classical 
2. PhoBERT fine-tune
3. Feature engineering + calibration
4. Ensemble/hybrid + ablation 
5. Error analysis theo 3 loại hallucination + report

## 8) Ý tưởng mới nhẹ
- Dynamic weight theo prompt type noise/adversarial (ước lượng từ đặc trưng prompt).
- Evidence consistency score ở mức token overlap + negation conflict.
- Confidence-aware fallback: nếu transformer entropy cao thì tăng trọng số classical/rule.

Dùng notebook `basic_classifier.ipynb` để thử nhanh ý tưởng trước khi chuẩn hóa vào scripts.
