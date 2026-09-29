import os, json
import numpy as np
from scipy import stats

DATA_DIR = "output/runs/lethe"

UMIA_KEY = "UMIA"
LT_KEY   = "LinkTeller unlearn auc with sampler balanced:"
LS_KEY   = "Link Stealing Attack 0 unlearned forget/non_exist"
ACC_KEY  = "sklearn.metrics.accuracy_score.test.unlearned.on_graph:False"

EXCLUDE = {"GNNDelete"}

def load_json(path):
    with open(path) as f:
        content = f.read().strip()
    if not content.startswith("["): content = "[" + content
    if content.endswith(","):       content = content[:-1]
    if not content.endswith("]"):   content = content + "]"
    return json.loads(content)

def canonical_label(r):
    u   = r.get("unlearner", "")
    sub = r.get("parameters", {}).get("sub_unlearner") or []
    cls = [s.get("class", "").split(".")[-1] for s in sub]
    if u == "Cascade":
        if any("UNSIR"    in c for c in cls): return "UNSIR"
        if any("Saliency" in c or "SalUn" in c for c in cls): return "SalUn"
        return "Cascade"
    MAP = {
        "Identity":                   "Identity",
        "GoldModelGraph":             "Gold Model",
        "Finetuning":                 "Finetuning",
        "NegGrad":                    "NegGrad",
        "AdvancedNegGrad":            "Adv. NegGrad",
        "BadTeaching":                "BadTeaching",
        "Scrub":                      "Scrub",
        "FisherForgetting":           "Fisher Forgetting",
        "SelectiveSynapticDampening": "SSD",
        "IDEA":                       "IDEA",
        "CEU":                        "CEU",
        "CGU_edge":                   "CGU",
        "ScaleGUN":                   "ScaleGUN",
        "GNNDelete":                  "GNNDelete",
    }
    return MAP.get(u, u)

def load_records(ds, arch, setting, keys):
    path = os.path.join(DATA_DIR, f"{ds}_{arch}_{setting}.json")
    if not os.path.exists(path):
        return {}
    out = {}
    for r in load_json(path):
        lbl = canonical_label(r)
        if lbl in EXCLUDE:
            continue
        if all(r.get(k) is not None for k in keys):
            out[lbl] = {k: r[k] for k in keys}
    return out

# RQ1: F-tests and t-tests (GCN hard)
print("RQ1 (GCN hard)")

RQ1_DATASETS = ["AmazonPhotos", "AmazonComputers", "DBLP", "Flickr", "ogbn-arxiv"]
RQ1_KEYS = [UMIA_KEY, LT_KEY, LS_KEY, ACC_KEY]

for ds in RQ1_DATASETS:
    recs = load_records(ds, "GCN", "hard", RQ1_KEYS)
    if not recs:
        print(f"\n{ds}: NO DATA")
        continue

    # get Identity accuracy as baseline
    identity_acc = recs.get("Identity", {}).get(ACC_KEY)

    umia_vals = np.array([v[UMIA_KEY] for v in recs.values()])
    lt_vals   = np.array([v[LT_KEY]   for v in recs.values()])
    ls_vals   = np.array([v[LS_KEY]   for v in recs.values()])
    acc_vals  = np.array([v[ACC_KEY]  for v in recs.values()])

    std_umia = np.std(umia_vals, ddof=1)
    std_lt   = np.std(lt_vals,   ddof=1)
    std_ls   = np.std(ls_vals,   ddof=1)
    n        = len(recs)

    # one-sided F-test, H1: var(LS) > var(other)
    def f_test_one_sided(s_num, s_den, n):
        F = (s_num ** 2) / (s_den ** 2)
        p = stats.f.sf(F, n - 1, n - 1)
        return F, p

    F_ls_umia, p_ls_umia = f_test_one_sided(std_ls, std_umia, n)
    F_ls_lt,   p_ls_lt   = f_test_one_sided(std_ls, std_lt,   n)

    ratio_umia = std_ls / std_umia if std_umia > 0 else float("nan")
    ratio_lt   = std_ls / std_lt   if std_lt   > 0 else float("nan")

    # accuracy t-test: delta from Identity
    if identity_acc is not None:
        deltas = acc_vals - identity_acc
        t_stat, p_acc = stats.ttest_1samp(deltas, 0)
        mean_delta_pp = np.mean(deltas) * 100
    else:
        p_acc = float("nan")
        mean_delta_pp = float("nan")

    print(f"\n{ds} (n={n} methods):")
    print(f"  std UMIA: {std_umia:.4f}  LT: {std_lt:.4f}  LS: {std_ls:.4f}")
    print(f"  LS/UMIA std ratio: {ratio_umia:.1f}  F={F_ls_umia:.2f}  p={p_ls_umia:.2e}")
    print(f"  LS/LT std ratio: {ratio_lt:.1f}  F={F_ls_lt:.2f}  p={p_ls_lt:.2e}")
    print(f"  acc t-test vs Identity: p={p_acc:.3f}  mean delta={mean_delta_pp:+.1f} pp")

# RQ2: Wilcoxon signed-rank (easy vs hard)
print("\nRQ2 (easy vs hard)")

RQ2_SETTINGS = [
    ("AmazonPhotos",    "GCN"),
    ("AmazonPhotos",    "GIN"),
    ("AmazonPhotos",    "SGC_CGU"),
    ("AmazonComputers", "GCN"),
    ("AmazonComputers", "GIN"),
    ("AmazonComputers", "SGC_CGU"),
    ("DBLP",            "GCN"),
    ("DBLP",            "GIN"),
    ("Flickr",          "GCN"),
    ("ogbn-arxiv",      "GCN"),
    ("ogbn-arxiv",      "GIN"),
    ("ogbn-arxiv",      "SGC_CGU"),
]

RQ2_KEYS = [LT_KEY, ACC_KEY]

all_easy_lt, all_hard_lt   = [], []
all_easy_acc, all_hard_acc = [], []
total_pairs = 0

print(f"\n{'Setting':<35} {'n_pairs':>8}  easy_LT   hard_LT   diff")
for ds, arch in RQ2_SETTINGS:
    easy = load_records(ds, arch, "easy", RQ2_KEYS)
    hard = load_records(ds, arch, "hard", RQ2_KEYS)
    common = sorted(set(easy) & set(hard))
    if not common:
        print(f"  {ds}_{arch}: no common methods")
        continue
    e_lt  = [easy[m][LT_KEY]  for m in common]
    h_lt  = [hard[m][LT_KEY]  for m in common]
    e_acc = [easy[m][ACC_KEY] for m in common]
    h_acc = [hard[m][ACC_KEY] for m in common]
    all_easy_lt.extend(e_lt);   all_hard_lt.extend(h_lt)
    all_easy_acc.extend(e_acc); all_hard_acc.extend(h_acc)
    total_pairs += len(common)
    label = f"{ds}_{arch}"
    print(f"  {label:<33} {len(common):>8}  {np.mean(e_lt):.3f}     {np.mean(h_lt):.3f}     {np.mean(e_lt)-np.mean(h_lt):+.3f}")

print(f"\nTotal pairs: {total_pairs}")
print(f"Settings with both easy+hard: {len([s for s in RQ2_SETTINGS if load_records(s[0], s[1], 'easy', RQ2_KEYS)])}")

all_easy_lt  = np.array(all_easy_lt)
all_hard_lt  = np.array(all_hard_lt)
all_easy_acc = np.array(all_easy_acc)
all_hard_acc = np.array(all_hard_acc)

n_all = len(all_easy_lt)
w_max = n_all * (n_all + 1) / 2
n_easy_gt_hard = np.sum(all_easy_lt > all_hard_lt)

print(f"\nLT Wilcoxon (easy > hard):")
print(f"  easy_LT > hard_LT: {n_easy_gt_hard} / {n_all}")
w_lt, p_lt = stats.wilcoxon(all_easy_lt, all_hard_lt, alternative="greater")
print(f"  W = {w_lt:.0f}  (W_max = {w_max:.0f})")
print(f"  p = {p_lt:.2e}")
print(f"  median diff = {np.median(all_easy_lt - all_hard_lt):+.3f}")
print(f"  mean diff = {np.mean(all_easy_lt - all_hard_lt):+.3f}")

print(f"\nAccuracy Wilcoxon:")
w_acc, p_acc = stats.wilcoxon(all_easy_acc, all_hard_acc)
print(f"  W = {w_acc:.0f}")
print(f"  p = {p_acc:.3f}")
print(f"  median diff = {np.median(all_easy_acc - all_hard_acc):+.4f}")
print(f"  mean diff = {np.mean(all_easy_acc - all_hard_acc)*100:+.1f} pp")
