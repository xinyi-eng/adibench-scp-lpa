"""Patch CF-ProtoNet: thread original episode class ids through so
LDAM margins and class-prior scaling use the ACTUAL sampled classes
rather than positional indexes.
"""
import io, sys

DATA = r"D:/dacd2026/adibench_v1/adibench/data.py"
BASE = r"D:/dacd2026/adibench_v1/adibench/baselines.py"
RUNNER = r"D:/dacd2026/adibench_v1/experiments/run_adibench_baseline.py"


def read(p):
    with io.open(p, "r", encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


def replace_once(src, old, new, label):
    n = src.count(old)
    if n != 1:
        print(f"FAIL [{label}]: found {n} occurrences")
        sys.exit(1)
    print(f"ok  [{label}]")
    return src.replace(old, new)


# ---------------------------------------------------------------- data.py
src = read(DATA)
src += '''

def build_fewshot_episode_ex(
    df: pd.DataFrame,
    n_way: int,
    k_shot: int,
    q_query: int,
    rng: Optional[random.Random] = None,
):
    """Same as build_fewshot_episode but also returns the ORIGINAL class
    ids chosen for this episode, so models with per-class parameters
    (e.g. CF-ProtoNet margins) can index them correctly."""
    if rng is None:
        rng = random.Random()
    by_class = {}
    for idx, label in zip(df.index, df["label"]):
        by_class.setdefault(int(label), []).append(idx)
    available_classes = list(by_class.keys())
    if len(available_classes) < n_way:
        chosen_classes = rng.choices(available_classes, k=n_way)
    else:
        chosen_classes = rng.sample(available_classes, n_way)
    support_idx, query_idx = [], []
    support_labels, query_labels = [], []
    for episode_label, orig_label in enumerate(chosen_classes):
        candidates = by_class[orig_label]
        if len(candidates) < k_shot + q_query:
            sampled = rng.choices(candidates, k=k_shot + q_query)
        else:
            sampled = rng.sample(candidates, k_shot + q_query)
        support_idx.extend(sampled[:k_shot])
        query_idx.extend(sampled[k_shot:k_shot + q_query])
        support_labels.extend([episode_label] * k_shot)
        query_labels.extend([episode_label] * q_query)
    return support_idx, query_idx, support_labels, query_labels, chosen_classes
'''
write(DATA, src)
print("data.py patched")

# ---------------------------------------------------------------- baselines.py
src = read(BASE)

src = replace_once(
    src,
    "        self.register_buffer(\"class_freq\", torch.zeros(1))\n"
    "        self.register_buffer(\"ldam_margins\", torch.zeros(1))",
    "        self.register_buffer(\"class_freq\", torch.zeros(1))\n"
    "        self.register_buffer(\"ldam_margins\", torch.zeros(1))\n"
    "        self._episode_classes = None  # original class ids of current episode\n"
    "\n"
    "    def set_episode_classes(self, chosen):\n"
    "        \"\"\"Original dataset class ids of the current episode's labels 0..n_way-1.\"\"\"\n"
    "        self._episode_classes = list(chosen)",
    "CFProtoNet set_episode_classes")

old_block = """        d_scaled = d * scale.unsqueeze(0)
        # LDAM margin: logits = -d + margin (push minority further)
        if hasattr(self, "ldam_margins") and self.ldam_margins.numel() >= n_way:
            margin = self.ldam_margins.to(d.device)[:n_way]
        else:
            margin = torch.zeros(n_way, device=d.device)"""
new_block = """        d_scaled = d * scale.unsqueeze(0)
        # LDAM margin: logits = -d + margin (push minority further).
        # Index by ORIGINAL class ids when available (subset episodes).
        if self._episode_classes is not None:
            idx = torch.tensor(self._episode_classes, device=d.device)
            margin = self.ldam_margins.to(d.device)[idx]
        elif hasattr(self, "ldam_margins") and self.ldam_margins.numel() == n_way:
            margin = self.ldam_margins.to(d.device)
        else:
            margin = torch.zeros(n_way, device=d.device)"""
src = replace_once(src, old_block, new_block, "CFProtoNet LDAM indexing")

old_prior = """        if hasattr(self, "prior_weights") and self.prior_weights.numel() == self.class_freq.numel():
            # map prototype idx -> original class idx
            # support_y are episode labels 0..n_way-1; need to know original
            # For benchmark fairness, treat prototypes as 0..n_way-1
            # and apply a soft re-weighting based on class_freq of those
            # episodes (we pass them via set_class_freq at train time)
            pw = self.prior_weights.to(d.device)
            # if prior_weights length matches n_way, use directly
            if pw.numel() == n_way:
                scale = pw.pow(0.5)
            else:
                scale = torch.ones(n_way, device=d.device)
        else:
            scale = torch.ones(n_way, device=d.device)"""
new_prior = """        if self._episode_classes is not None and hasattr(self, "prior_weights"):
            idx = torch.tensor(self._episode_classes, device=d.device)
            scale = self.prior_weights.to(d.device)[idx].pow(0.5)
        elif hasattr(self, "prior_weights") and self.prior_weights.numel() == n_way:
            scale = self.prior_weights.to(d.device).pow(0.5)
        else:
            scale = torch.ones(n_way, device=d.device)"""
src = replace_once(src, old_prior, new_prior, "CFProtoNet prior indexing")
write(BASE, src)
print("baselines.py patched")

# ---------------------------------------------------------------- runner
src = read(RUNNER)
src = replace_once(
    src,
    "from adibench.data import (\n"
    "    load_nadi_18way, load_nadi_5way, load_amgadhasan_5city,\n"
    "    build_fewshot_episode, list_datasets, get_dataset, DATASETS,\n"
    ")",
    "from adibench.data import (\n"
    "    load_nadi_18way, load_nadi_5way, load_amgadhasan_5city,\n"
    "    build_fewshot_episode, build_fewshot_episode_ex, list_datasets, get_dataset, DATASETS,\n"
    ")",
    "runner import")

# evaluate(): cf_protonet needs chosen classes
src = replace_once(
    src,
    """        for _ in range(n_episodes):
            s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df, n_way, k_shot, q_query)""",
    """        for _ in range(n_episodes):
            if method == "cf_protonet":
                s_idx, q_idx, s_lab, q_lab, chosen = build_fewshot_episode_ex(df, n_way, k_shot, q_query)
                model.set_episode_classes(chosen)
            else:
                s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df, n_way, k_shot, q_query)""",
    "runner evaluate episodes")

# train_method(): cf_protonet needs chosen classes
src = replace_once(
    src,
    """    for ep in range(1, n_episodes + 1):
        s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df, n_way, k_shot, q_query)""",
    """    for ep in range(1, n_episodes + 1):
        if method == "cf_protonet":
            s_idx, q_idx, s_lab, q_lab, chosen = build_fewshot_episode_ex(df, n_way, k_shot, q_query)
            model.set_episode_classes(chosen)
        else:
            s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df, n_way, k_shot, q_query)""",
    "runner train episodes")
write(RUNNER, src)
print("runner patched")
