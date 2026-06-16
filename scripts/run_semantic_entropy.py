#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run Semantic Entropy Hallucination Detection.
This script loads a generative LLM to sample multiple responses for each prompt,
computes their semantic embeddings, clusters them by similarity, and calculates
the semantic entropy. Higher entropy indicates higher generative uncertainty,
which correlates with a higher likelihood of hallucination.

Usage:
    python scripts/run_semantic_entropy.py \
      --input_path vihallu-train.csv \
      --output_dir outputs/semantic_entropy \
      --llm_name Qwen/Qwen2.5-1.5B-Instruct \
      --embed_name keepitreal/vietnamese-sbert \
      --sample_size 10 \
      --num_samples 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vihallu.models.advanced import compute_semantic_entropy
from vihallu.utils.io_utils import ensure_dir, save_json


# Robust wrapper for sentence encoding that works with or without sentence-transformers package
class SentenceEncoder:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        try:
            from sentence_transformers import SentenceTransformer
            print(f"[Embedding] Loading '{model_name}' using sentence-transformers...")
            self.model = SentenceTransformer(model_name, device=device)
            self.use_st = True
        except ImportError:
            from transformers import AutoModel, AutoTokenizer
            print(f"[Embedding] sentence-transformers not found. Loading '{model_name}' using transformers fallback...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.use_st = False

    def encode(self, texts: list[str], convert_to_tensor: bool = True) -> torch.Tensor | np.ndarray:
        if not texts:
            return torch.empty(0) if convert_to_tensor else np.empty(0)

        if self.use_st:
            return self.model.encode(texts, convert_to_tensor=convert_to_tensor)
        else:
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Mean Pooling
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            if convert_to_tensor:
                return embeddings
            return embeddings.cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Semantic Entropy on ViHallu.")
    parser.add_argument("--input_path", type=str, default="vihallu-train.csv", help="Path to input dataset CSV.")
    parser.add_argument("--output_dir", type=str, default="outputs/semantic_entropy", help="Directory to save output results.")
    parser.add_argument("--llm_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Causal LLM model name/path.")
    parser.add_argument("--embed_name", type=str, default="keepitreal/vietnamese-sbert", help="Sentence embedding model name/path.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of responses M to generate per prompt.")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature for sampling (>= 0.7 is recommended).")
    parser.add_argument("--similarity_threshold", type=float, default=0.8, help="Cosine similarity threshold for semantic clustering.")
    parser.add_argument("--sample_size", type=int, default=0, help="Number of prompts to sample (0 to process all).")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max new tokens to generate per response.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"], help="Device to run inference on.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    
    # 1. Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"[*] Running on device: {device}")

    # Set random seed
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # 2. Read dataset
    print(f"[*] Reading dataset: {args.input_path}")
    if not os.path.exists(args.input_path):
        # Fallback search
        from vihallu.data import resolve_data_path
        try:
            resolved_path = resolve_data_path(args.input_path)
            df = pd.read_csv(resolved_path)
        except Exception as e:
            print(f"[Error] Could not find input file: {args.input_path}. {e}")
            sys.exit(1)
    else:
        df = pd.read_csv(args.input_path)

    # Sample rows if requested
    if args.sample_size > 0:
        print(f"[*] Sampling {args.sample_size} rows for benchmarking...")
        df = df.sample(n=min(args.sample_size, len(df)), random_state=args.seed).reset_index(drop=True)
    else:
        print(f"[*] Processing all {len(df)} rows...")

    # 3. Load Embedding Model
    print(f"[*] Loading Sentence Embedding Model: {args.embed_name}")
    encoder = SentenceEncoder(args.embed_name, device=device)

    # 4. Load Generative LLM
    print(f"[*] Loading Causal LLM: {args.llm_name}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Use float16 or bfloat16 for fast GPU inference if supported
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    start_time = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
        # Add pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        if device in ["cuda", "mps"]:
            model = AutoModelForCausalLM.from_pretrained(
                args.llm_name,
                device_map="auto",
                torch_dtype=torch_dtype,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.llm_name,
                torch_dtype=torch_dtype,
            ).to(device)
        print(f"[*] Generative model loaded successfully in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"[Error] Failed to load LLM model {args.llm_name}: {e}")
        print("[!] Please check if the model name is correct, internet connection is active, or VRAM is sufficient.")
        sys.exit(1)

    # 5. Generate and Compute Semantic Entropy
    results = []
    print(f"[*] Starting Semantic Entropy generation and clustering (M={args.num_samples} samples per query)...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Calculating Semantic Entropy"):
        context = str(row.get("context", ""))
        prompt = str(row.get("prompt", ""))
        orig_response = str(row.get("response", ""))
        true_label = str(row.get("label", "unknown"))

        # Build prompt for LLM generation
        # We construct a clean QA template
        input_text = f"Context: {context}\nQuestion: {prompt}\nAnswer the question based on the context. Keep your answer brief.\nAnswer:"
        
        # Tokenize prompt
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        # Generate M samples in parallel with high-temperature sampling
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_p=0.95,
                num_return_sequences=args.num_samples,
            )
        generated_responses = []
        for out in outputs:
            # Decode only the generated part
            gen_tokens = out[input_len:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            if not gen_text:
                gen_text = " "
            generated_responses.append(gen_text)

        # Compute Semantic Entropy using the utility from vihallu
        entropy = compute_semantic_entropy(
            responses=generated_responses,
            semantic_model=encoder,
            similarity_threshold=args.similarity_threshold
        )

        results.append({
            "id": row.get("id", idx),
            "context": context,
            "prompt": prompt,
            "original_response": orig_response,
            "true_label": true_label,
            "generated_responses": generated_responses,
            "semantic_entropy": float(entropy)
        })

    # 6. Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "semantic_entropy_results.csv", index=False)
    
    # Save a JSON file with full responses
    with open(output_dir / "semantic_entropy_details.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 7. Print analysis and report
    print("\n" + "="*50)
    print("SEMANTIC ENTROPY BENCHMARK REPORT")
    print("="*50)
    print(f"Saved detailed results to:")
    print(f"  - {output_dir / 'semantic_entropy_results.csv'}")
    print(f"  - {output_dir / 'semantic_entropy_details.json'}")
    print("-"*50)
    
    # If the labels exist and are known, show average entropy per label
    unique_labels = results_df["true_label"].unique()
    if len(unique_labels) > 1 and "unknown" not in unique_labels:
        print("Average Semantic Entropy by Label:")
        stats = results_df.groupby("true_label")["semantic_entropy"].agg(["mean", "std", "count"])
        print(stats.to_string())
        
        # Calculate evaluation score: AUC-ROC if binary or macro correlation
        # Treat "no" as 0 (factual) and "intrinsic"/"extrinsic" as 1 (hallucinated)
        results_df["is_hallucinated"] = results_df["true_label"].apply(lambda x: 0 if x == "no" else 1)
        if results_df["is_hallucinated"].nunique() > 1:
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(results_df["is_hallucinated"], results_df["semantic_entropy"])
                print(f"\nHallucination Detection AUC-ROC (Entropy as Score): {auc:.4f}")
                print("Note: Higher AUC-ROC means Semantic Entropy is a better indicator of hallucination.")
            except ImportError:
                pass
    else:
        print(f"Overall Average Semantic Entropy: {results_df['semantic_entropy'].mean():.4f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
