import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==============================================================================
# 1. Semantic Entropy
# ==============================================================================
def compute_semantic_entropy(responses, semantic_model, similarity_threshold=0.8):
    """
    Computes semantic entropy over multiple generated responses to detect hallucination.
    """
    if not responses:
        return 0.0
        
    # Step 1: Extract semantic embeddings for all generated responses
    # Assuming semantic_model is a SentenceTransformer or similar
    embeddings = semantic_model.encode(responses, convert_to_tensor=True)
    
    # Step 2: Semantic clustering based on cosine similarity
    clusters = []
    for i, emb in enumerate(embeddings):
        placed = False
        for cluster in clusters:
            center_emb = cluster['center']
            sim = F.cosine_similarity(emb.unsqueeze(0), center_emb.unsqueeze(0)).item()
            if sim >= similarity_threshold:
                cluster['members'].append(i)
                placed = True
                break
        if not placed:
            clusters.append({'center': emb, 'members': [i]})
            
    # Step 3: Compute entropy (-sum p * log p) across semantic clusters
    entropy = 0.0
    total_responses = len(responses)
    for cluster in clusters:
        p_cluster = len(cluster['members']) / total_responses
        entropy -= p_cluster * np.log(p_cluster + 1e-9)
        
    return entropy

# ==============================================================================
# 2. HaluAgent
# ==============================================================================
class HaluAgent:
    """
    Autonomous hallucination detection agent with a multi-stage detection pipeline:
    sentence segmentation, tool-based verification, and reflective reasoning.
    """
    def __init__(self, llm_pipeline, external_tools):
        self.llm = llm_pipeline
        self.tools = external_tools  # Dictionary of tool functions: search, calculator, etc.
        
    def segment_sentences(self, target_text):
        # Placeholder for robust sentence tokenization (e.g., using nltk/spacy)
        return [s.strip() for s in target_text.split('.') if s.strip()]
        
    def verify_with_tools(self, claim):
        """Tool-based verification using external resources."""
        evidence_collected = []
        for tool_name, tool_func in self.tools.items():
            try:
                result = tool_func(claim)
                if result:
                    evidence_collected.append(f"[{tool_name}] {result}")
            except Exception as e:
                continue
        return " | ".join(evidence_collected) if evidence_collected else "No external evidence found."
        
    def reflective_reasoning(self, claim, evidence):
        """Reflective reasoning using the small open-source LLM."""
        prompt = (
            f"Claim: {claim}\n"
            f"Evidence: {evidence}\n"
            f"Based on the evidence, is the claim a hallucination? Think step-by-step."
        )
        # Calling the generative pipeline
        return self.llm.generate(prompt)

    def detect(self, generated_response):
        sentences = self.segment_sentences(generated_response)
        detection_stages = []
        for sent in sentences:
            evidence = self.verify_with_tools(sent)
            reasoning_result = self.reflective_reasoning(sent, evidence)
            detection_stages.append({
                "claim": sent, 
                "evidence": evidence, 
                "reasoning": reasoning_result
            })
        return detection_stages

# ==============================================================================
# 3. MHAD Probing (Model Hallucination Awareness for Hallucination Detection)
# ==============================================================================
class LinearProbe(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Two-layer FFN for mapping internal hidden rep to a binary hallucination label
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.ffn(x))

class MHAD(nn.Module):
    def __init__(self, hidden_dim, num_layers, alpha=0.9):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.alpha = min(max(alpha, 0.0), 1.0) # 0 < alpha < 1
        
        # Probes for EACH layer at initial (t=1) and final (t=T) generation steps
        self.probes_initial = nn.ModuleList([LinearProbe(hidden_dim) for _ in range(num_layers)])
        self.probes_final   = nn.ModuleList([LinearProbe(hidden_dim) for _ in range(num_layers)])
        
    def neuron_selection(self, probe_weights):
        """
        Selects neurons based on the absolute weight parameters.
        Equation 9, 10, 11 from the specification.
        probe_weights shape: (hidden_dim // 2, hidden_dim) or similar
        """
        # Average weight importance across output dimension of the first linear layer
        w_abs = probe_weights.abs().mean(dim=0)
        
        # Sort in descending order
        sorted_indices = torch.argsort(w_abs, descending=True)
        sorted_weights = w_abs[sorted_indices]
        
        # Calculate cumulative sum of squared weights
        w_sq = sorted_weights ** 2
        total_sq_sum = w_sq.sum()
        cumsum_w_sq = torch.cumsum(w_sq, dim=0)
        
        # Find threshold where cumsum exceeds alpha * total
        mask = cumsum_w_sq >= (self.alpha * total_sq_sum)
        if mask.any():
            i_thresh = mask.nonzero(as_tuple=True)[0][0].item() + 1
        else:
            i_thresh = len(sorted_weights)
            
        selected_neurons = sorted_indices[:i_thresh]
        return selected_neurons

    def construct_hav(self, initial_reps, final_reps, selected_layers):
        """
        Construct the Hallucination Awareness Vector (HAV).
        initial_reps / final_reps: list of tensors of shape (batch, hidden_dim), length = num_layers
        """
        hav_components = []
        
        for l in selected_layers:
            # Step Init: Select neurons from the first linear layer of the initial probe
            w_init = self.probes_initial[l].ffn[0].weight.data
            n_init = self.neuron_selection(w_init)
            hav_components.append(initial_reps[l][:, n_init])
            
            # Step Final: Select neurons from the first linear layer of the final probe
            w_final = self.probes_final[l].ffn[0].weight.data
            n_final = self.neuron_selection(w_final)
            hav_components.append(final_reps[l][:, n_final])
            
        # Concatenate outputs from selected neurons to form HAV
        hav_vector = torch.cat(hav_components, dim=-1)
        return hav_vector

# ==============================================================================
# 4. Ensemble Fine-tune (Qwen3-4B-Instruct with LoRA)
# ==============================================================================
def setup_qwen_lora_ensemble(model_name="Qwen/Qwen3-4B-Instruct", num_models=3, r=8, lora_alpha=16):
    """
    Initializes an ensemble of models using Low-Rank Adaptation (LoRA) for fine-tuning.
    Designed for small/efficient tuning. Since there's no GPU in target env, device_map maps to CPU.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ensemble_instances = []
    
    for i in range(num_models):
        # Initialize base model (CPU only as requested)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="cpu", 
            torch_dtype=torch.float32 
        )
        
        # Configure Low-Rank Adaptation (LoRA)
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
            lora_dropout=0.05,
            bias="none"
        )
        
        # Wrap into PEFT model
        peft_model = get_peft_model(base_model, config)
        
        # Make sure weights are trainable
        peft_model.print_trainable_parameters()
        ensemble_instances.append(peft_model)
        
    return ensemble_instances, tokenizer

def run_ensemble_inference(ensemble_instances, tokenizer, prompt):
    """Run inference utilizing the ensemble of LoRA fine-tuned models."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    
    generated_texts = []
    for peft_model in ensemble_instances:
        # Generate with each ensemble member
        output_tokens = peft_model.generate(**inputs, max_new_tokens=100)
        text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        generated_texts.append(text)
        
    return generated_texts
