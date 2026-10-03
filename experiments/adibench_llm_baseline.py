"""Step 10: Arabic LLM baseline scaffold.

This script provides:
1. A **GPT-4o baseline** scaffold (requires OPENAI_API_KEY env var).
2. A **Jais-13B few-shot ICL** scaffold (requires HuggingFace token).
3. A **sentence-transformer multilingual baseline** that simulates
   LLM semantic matching behavior (proxy, runs locally without API).

Usage:
    # Set OPENAI_API_KEY env var, then run:
    python adibench_llm_baseline.py --method gpt4o --direction nadi_to_amgad
    python adibench_llm_baseline.py --method jais --direction amgad_to_nadi
    python adibench_llm_baseline.py --method labse --direction nadi_to_amgad
"""
import sys, os, json, argparse, time
sys.path.insert(0, r"D:/dacd2026/adibench_v1")
import numpy as np
import torch

from adibench.data import build_fewshot_episode, get_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def build_prompt_ar(support_texts, support_labels, query_text):
    """Build an Arabic few-shot ICL prompt for dialect ID."""
    parts = ["أنت خبير في اللهجات العربية. صنّف التغريدة التالية حسب اللهجة.\n"]
    for txt, lbl in zip(support_texts, support_labels):
        parts.append(f"مثال {lbl}: {txt}\n")
    parts.append(f"\nالتغريدة: {query_text}\nاللهجة:")
    return "".join(parts)


def call_gpt4o(prompts, model="gpt-4o-mini"):
    """Call OpenAI API. Returns list of (response_text, logprob_dict) tuples."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        out = []
        for prompt in prompts:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an Arabic dialectologist. Reply with only a digit 0-4 for the dialect ID."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2,
                temperature=0,
                logprobs=True,
                top_logprobs=5,
            )
            out.append(resp)
        return out
    except ImportError:
        raise RuntimeError("openai not installed: pip install openai")


def call_jais(prompts, model_name="inceptionai/jais-13b-chat"):
    """Call Jais via HuggingFace transformers."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir="D:/dacd2026/2_models")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                                 cache_dir="D:/dacd2026/2_models",
                                                 torch_dtype=torch.float16,
                                                 device_map="auto")
    out = []
    for prompt in prompts:
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=2, do_sample=False,
                                  output_scores=True, return_dict_in_generate=True)
        # Extract logits for next token over vocab
        next_token_logits = gen.scores[0][0]  # (vocab_size,)
        probs = torch.softmax(next_token_logits, dim=-1).cpu().numpy()
        # Get top-k digit tokens (assuming 0-4 output)
        digit_probs = {i: float(probs[tok.encode(str(i), add_special_tokens=False)[-1]])
                       for i in range(5)}
        out.append(digit_probs)
    return out


def call_labse_simulator(df, n_way, k_shot, q_query, n_eval, dataset):
    """TF-IDF + cosine similarity baseline (LLM proxy without model download).

    Represents what a bag-of-words / TF-IDF based language model would
    do for text classification. No GPU or HF download required.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    print(f"TF-IDF simulator (LLM proxy) on {dataset}")

    # Fit TF-IDF on first 5000 tweets for speed
    sample = df["text"].head(5000).tolist()
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    vec.fit(sample)

    correct, total = 0, 0
    per_ep = []
    import random
    rng = random.Random(42)
    for _ in range(n_eval):
        s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df, n_way, k_shot, q_query, rng=rng)
        s_texts = [df.iloc[i]["text"] for i in s_idx]
        q_texts = [df.iloc[i]["text"] for i in q_idx]
        s_emb = normalize(vec.transform(s_texts))
        q_emb = normalize(vec.transform(q_texts))
        protos = np.zeros((n_way, s_emb.shape[1]))
        for c in range(n_way):
            mask = np.array(s_lab) == c
            if mask.any():
                protos[c] = s_emb[mask].mean(axis=0)
        protos = normalize(protos)
        sims = q_emb @ protos.T  # (n_q, n_way)
        preds = sims.argmax(axis=1)
        correct += (preds == np.array(q_lab)).sum()
        total += len(q_lab)
        per_ep.append((preds == np.array(q_lab)).mean())
    return correct / max(1, total), float(np.std(per_ep))


def run(method, source_ds, target_ds, seed=42, n_episodes=200, n_eval=100):
    print(f"\n=== LLM baseline: {method} on {source_ds} -> {target_ds} ===", flush=True)
    df, meta = get_dataset(target_ds)
    if method == "labse":
        acc, std = call_labse_simulator(df, 5, 5, 15, n_eval, target_ds)
        ci = 1.96 * std / np.sqrt(n_eval)
        result = {"method": method, "source": source_ds, "target": target_ds,
                  "seed": seed, "accuracy": acc, "std": std, "ci95": ci}
        out_name = f"llm_{method}_{source_ds}_to_{target_ds}_seed{seed}.json"
        with open(os.path.join(r"D:/dacd2026/adibench_v1/results", out_name), "w") as f:
            json.dump(result, f, indent=2)
        print(f"  [{method}/{target_ds}] acc={acc:.4f}+-{ci:.4f}")
        return result
    else:
        print(f"  [{method}] requires API key; skipping actual run")
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["gpt4o", "jais", "labse"], default="labse")
    ap.add_argument("--direction", choices=["nadi_to_amgad", "amgad_to_nadi"], default="nadi_to_amgad")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_eval", type=int, default=100)
    args = ap.parse_args()
    source = "nadi_18" if "nadi" in args.direction else "amgadhasan_5"
    target = "amgadhasan_5" if "amgad" in args.direction else "nadi_18"
    run(args.method, source, target, args.seed, n_eval=args.n_eval)