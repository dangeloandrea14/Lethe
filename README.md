<p align="center">
  <img src="Lethe.png" alt="Lethe logo" width="180"/>
</p>


<h1 align="center">Lethe</h1>
<h3 align="center">Link Inference Attacks for Evaluation of Edge Unlearning Methods</h3>

<p align="center">
  <img alt="NeurIPS 2026 Evaluations &amp; Datasets" src="https://img.shields.io/badge/NeurIPS%202026-Evaluations%20%26%20Datasets-68448B"/>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue"/>
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-orange"/>
  <img alt="PyG" src="https://img.shields.io/badge/PyG-2.3%2B-green"/>
  <img alt="License" src="https://img.shields.io/badge/license-see%20LICENSE-lightgrey"/>
</p>

<p align="center">
  Re-evaluating <strong>Edge Unlearning</strong> in Graph Neural Networks.
</p>

<p align="center">
  <img src="assets/lethe_banner.svg" alt="Lethe — accepted at NeurIPS 2026, Evaluations &amp; Datasets Track" width="100%"/>
</p>

---

> ## ▶ &nbsp;Start here — [`lethe_quickstart.ipynb`](notebooks/lethe_quickstart.ipynb)
>
> Run a real benchmark configuration end-to-end, then drop in **your own unlearning method**
> and see it scored against all 15 baselines.
>
> **Finishes in minutes once the environment is set up.**

### Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Why WalkCentrality works](#why-walkcentrality-works)
- [Framework at a Glance](#framework-at-a-glance)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [License](#license)

---

## Overview

Graph Unlearning (GU) aims to remove the influence of specific training data from a GNN without retraining from scratch. **Edge Unlearning (EU)** targets edges, a practically important setting motivated by privacy regulations such as the GDPR, where a user may request the removal of a relational tie from a trained model.

Existing evaluations of EU methods rely on two systematic shortcomings: they use **small, feature-rich datasets** where retraining from scratch is nearly free, and they use **accuracy metrics** that are insensitive to edge removal. Lethe addresses both issues by (i) focusing on large graphs where runtime savings are meaningful, and (ii) complementing accuracy with **link inference attacks** that directly probe whether a forgotten edge can still be recovered from the model.

Concretely, Lethe provides:

- **15 unlearning methods** (general-purpose MU and graph-specific EU methods)
- **8 datasets** selected for scale and structural informativeness
- **3 link inference attacks**: E-UMIA, LinkTeller (LT), and Link Stealing (LS)
- **WalkCentrality**, a model-agnostic centrality measure used to build forget sets of varying difficulty (low / high centrality)
- A fully **configuration-driven**, modular design: extending to a new dataset, model, or attack requires no changes to the framework core

---

## Key Findings

### The metric you evaluate with decides what you can see

| Metric | Spread across methods | Discriminative? |
| --- | --- | --- |
| Test accuracy | ≤ 11 pp (Photos) · < 1.5 pp (DBLP) | **No**, within statistical noise |
| E-UMIA | AUC ≈ 0.48–0.53 everywhere | **No**, uniformly blind |
| LinkTeller | 0.94–1.00 low-centrality · 0.38–0.64 high | **Yes** |
| **Link Stealing** | **20–30 pp**, consistent ranking | **Yes** |

> **RQ1 — Are link inference attacks more informative than accuracy?**
> Yes. E-UMIA is uniformly blind across all methods; accuracy is essentially uninformative on
> every benchmark dataset. Link Stealing is the most effective attack, producing a consistent,
> dataset-agnostic ranking with a spread accuracy cannot reproduce. 

> **RQ2 — Do WalkCentrality settings reveal qualitatively different behaviour?**
> Yes. **Low**-centrality edges are the *easier* ones to detect because their removal is localized and not absorbed
> by alternative paths. **High**-centrality edges push LT toward chance (0.38–0.64).

> **RQ3 — Are certified EU methods applicable in practice?**
> Only partially. CEU, CGU, and ScaleGUN require a linear architecture (SGC), which underperforms
> GCN by 3.6 pp (Photos), 4.2 pp (Computers) and 1.2 pp (Flickr). At a 5% forget set their runtime
> already exceeds retraining from scratch. Worse, their guarantees do not translate into reduced
> adversary success: on Photos, CGU and ScaleGUN *worsen* LS relative to Identity by 8.3 pp and
> 6.6 pp; on ogbn-arxiv all three *increase* LinkTeller AUC. 

### Why WalkCentrality works

Not all edges are equally hard to unlearn, so sampling the forget set at random — as every prior EU evaluation does — conflates easy and hard cases. WalkCentrality asks instead: *given an edge, how redundant is it in the message-passing system?*

A *k*-layer GNN propagates over the symmetrically normalised adjacency **Â** = D̃<sup>−1/2</sup>(A + I)D̃<sup>−1/2</sup>. Any *k*-step walk crossing edge (*i* → *j*) at step *t* splits uniquely into a *t*-step walk arriving at *i* and a (*k*−*t*−1)-step walk leaving *j*. Summing over every such split:

```
                       k−1
WalkCentrality(i,j)  =  Σ  (1ᵀÂᵗ)ᵢ · (Â^(k−t−1)1)ⱼ
                       t=0
```

which counts how often the edge carries signal during message passing. Precomputing `{Âᵗ1}` and `{1ᵀÂᵗ}` reduces every edge to a lookup and a sum, so scoring the entire graph costs **O(k · |E|)**.

> WalkCentrality measures **propagation redundancy, a property of the graph alone**, computed
> without reference to any model, task, or attack. Attack detectability is a downstream
> empirical consequence of it, which is why the low/high settings act as difficulty *bounds*.

Two measurements confirm the mechanism holds empirically (GCN, 5% forget set):

| Evidence | Photos | Computers | DBLP |
| --- | --- | --- | --- |
| ρ (WalkCentrality, training-loss increase after removal) | −0.989 | −0.993 | −0.989 |
| Endpoint displacement, low ÷ high centrality | **5.2×** | **6.7×** | **5.3×** |


LinkTeller measures exactly that endpoint influence. 

---

## Framework at a Glance

| Component | Details |
|-----------|---------|
| **Datasets** | AmazonPhotos, AmazonComputers, DBLP, Flickr, ogbn-arxiv, Cora, Citeseer, Pubmed |
| **GNN architectures** | GCN, GIN, GAT, GraphSAGE, SGC, SGC-CGU, MLP |
| **Unlearning methods** | Identity, Gold Model, Fine-tuning, NegGrad, Adv. NegGrad, UNSIR, Bad Teaching, SCRUB, Fisher Forgetting, SSD, SalUn, IDEA, CGU, CEU, ScaleGUN, GNNDelete |
| **Link inference attacks** | E-UMIA, LinkTeller (LT), Link Stealing (LS) |
| **Forget-set settings** | Low centrality (easiest 5% by WalkCentrality), High centrality (hardest 5%) |
| **Results format** | JSON files under `output/runs/` |

---

## Installation

We recommend using [conda](https://docs.conda.io/en/latest/). The `environment.yml` file is a conda environment specification.

```bash
conda env create -f environment.yml
conda activate lethe
```

To use a different CUDA version, edit the `pytorch-cuda` line in `environment.yml` before creating the environment (e.g. `pytorch-cuda=12.1`). For CPU-only usage, remove that line entirely.


---

## Quick Start

### Try it in a notebook

[`notebooks/lethe_quickstart.ipynb`](notebooks/lethe_quickstart.ipynb) runs a real benchmark configuration end-to-end and shows how to plug in your own unlearning method. Launch it with the `lethe` environment active, from anywhere inside the repository — it locates the repository root itself.

### Run a single experiment

`test_launch.sh` runs one JSONC configuration file and prints its output to the terminal.

```bash
# Default: AmazonPhotos × GCN × high-centrality forget set
bash test_launch.sh

# Custom config
bash test_launch.sh configs/benchmark/lethe/DBLP_GIN_easy.jsonc

# Small demo config: 3 unlearners instead of 16, finishes in minutes
bash test_launch.sh configs/demo/DBLP_GCN_demo.jsonc
```

Internally, this calls:

```bash
python main.py <config>.jsonc
```

### Run the full Lethe benchmark

`configs/benchmark/lethe/` holds the 60 benchmark configurations. Run them one after another with:

```bash
for cfg in configs/benchmark/lethe/*.jsonc; do
    python main.py "$cfg"
done
```

Results are saved to `output/runs/lethe/` as JSON files, one per configuration.

---

## Configuration

All experiments are specified through **JSONC files** (JSON with comments). A single config fully determines one experiment: the dataset, the model, which unlearning methods to evaluate, and which metrics to compute.

```bash
python main.py configs/benchmark/lethe/AmazonPhotos_GCN_hard.jsonc
```

### Config structure

A config file has five top-level blocks:

```jsonc
{
  "data":       { ... },   // Dataset loading and partitioning
  "predictor":  { ... },   // GNN architecture and training
  "unlearners": [ ... ],   // List of unlearning methods to evaluate
  "evaluator":  { ... },   // Metrics and attacks to compute
  "globals":    { ... }    // Seed, caching, removal type
}
```

---

#### `data` — Dataset and forget-set construction

Defines the data source (any PyTorch Geometric dataset), the train/val/test split, and the forget-set splitter. The splitter uses **WalkCentrality** to select the lowest-scoring (`easy`) or highest-scoring (`hard`) 5% of edges as the forget set, ensuring reproducibility via a fixed `split_seed`.

```jsonc
"data": {
  "class": "lethe.data.datasets.DatasetManager.DatasetManager",
  "parameters": {
    "DataSource": {
      "class": "lethe.data.data_sources.TorchGeometricDataSource.TorchGeometricDataSource",
      "parameters": { "datasource": {
        "class": "torch_geometric.datasets.CitationFull",
        "parameters": { "root": "resources/data", "name": "DBLP" },
        "preprocess": []
      }}
    },
    "partitions": [
      // train / validation / test splits omitted for brevity
      { "class": "lethe.data.datasets.DataSplitterGraph.DataSplitterEdgeDifficulty",
        "parameters": { "parts_names": ["forget", "retain"],
                        "percentage": 0.05, "ref_data": "all", "mode": "hard" } }
    ],
    "batch_size": 4,
    "split_seed": 16
  }
}
```

`mode` is `"hard"` for the high-centrality forget set and `"easy"` for the low-centrality one.

---

#### `predictor` — GNN architecture and training

Specifies the model class, hidden dimensions, learning rate, loss function, and number of epochs. The `alias` field is used to cache the trained Original model and Gold Model across all unlearning methods in the same config, avoiding redundant training.

```jsonc
"predictor": {
  "class": "lethe.model.TorchGraphModel.TorchGraphModel",
  "parameters": {
    "model": {
      "class": "lethe.model.graphs.GCN.GCN",
      "parameters": { "in_channels": 745, "hidden_channels": [64], "out_channels": 8 }
    },
    "optimizer": { "class": "torch.optim.Adam", "parameters": { "lr": 0.001 } },
    "loss_fn":   { "class": "torch.nn.CrossEntropyLoss",
                   "parameters": { "reduction": "mean" } },
    "epochs": 100,
    "alias": "AmazonPhotos_GCN"
  }
}
```

---

#### `unlearners` — Methods to benchmark

A list of unlearning method entries. Each entry has a `class` (Python dotted path) and `parameters`. Methods are evaluated sequentially, each starting from an independent copy of the trained Original model. The Identity baseline (no unlearning) and Gold Model (retrain from scratch) are always included as reference points.

```jsonc
"unlearners": [
  { "class": "lethe.unlearners.composite.Identity", "parameters": {} },
  { "class": "lethe.unlearners.GoldModel.GoldModelGraph",
    "parameters": { "training_set": "retain" } },
  { "class": "lethe.unlearners.graph_unlearners.Finetuning.Finetuning",
    "parameters": { "epochs": 1, "ref_data": "retain",
                    "optimizer": { "class": "torch.optim.Adam",
                                   "parameters": { "lr": 0.001 } } } },
  { "class": "lethe.unlearners.graph_unlearners.NegGrad.NegGrad",
    "parameters": { "epochs": 1, "ref_data": "forget",
                    "optimizer": { "class": "torch.optim.Adam",
                                   "parameters": { "lr": 0.001 } } } }
  // ... add more methods here
]
```

To add a new method, append one entry to this list — no changes to the framework code are needed.

---

#### `evaluator` — Metrics and attacks

Lists the measures to compute for each unlearner. These include runtime, node-classification accuracy and macro-F1 on the test/forget/retain splits, and all three link inference attacks. Results are serialized to JSON.

```jsonc
"evaluator": {
  "class": "lethe.evaluations.manager.Evaluator",
  "parameters": {
    "measures": [
      { "class": "lethe.evaluations.running.RunTime" },
      { "class": "lethe.evaluations.measures.TorchSKLearnGraph",
        "parameters": { "partition": "test", "target": "unlearned" } },
      // E-UMIA is composed from a snippet: it needs a pre-generated attack set.
      { "compose_umia": "configs/snippets/e_umia_graph.json" },
      { "class": "lethe.evaluations.LinkTeller.LinkTeller.LinkTeller",
        "parameters": { "target": "unlearn" } },
      { "class": "lethe.evaluations.link_stealing_attack.link_stealing_attack_0.LinkStealing0",
        "parameters": { "target": "unlearned" } },
      { "class": "lethe.evaluations.measures.SaveValues",
        "parameters": { "path": "output/runs/lethe/MyRun.json" } }
    ]
  }
}
```

`SaveValues` takes a **file** path, not a directory, and **appends** one JSON record per unlearner. Delete the file before re-running a config, or you will accumulate duplicate records. The result is comma-separated objects rather than a JSON array — see the parsing helper in the quickstart notebook.

---

#### `globals` — Run-level settings

```jsonc
"globals": {
  "cached":        true,    // Cache Original + Gold models to disk
  "seed":          0,       // Model weight initialisation seed
  "removal_type":  "edge"   // Type of unlearning request
}
```

---

### Extending Lethe

Lethe is designed to be modular. Adding any of the following requires **no changes to the framework core**, only a new config entry or a new class file.

| Extension | What to do |
|-----------|-----------|
| New dataset | Point `datasource.class` at any `torch_geometric.datasets.*` class, or wrap a custom dataset in a `TorchGeometricDataSource`-compatible class |
| New GNN architecture | Implement a `torch.nn.Module` with `forward(x, edge_index)` under `lethe/model/graphs/`, then reference it in `predictor.model.class` |
| New unlearning method | Subclass `GraphUnlearner`, implement `__unlearn__`, add one entry to the `unlearners` list |
| New link inference attack | Subclass `GraphMeasure`, implement `process(evaluation)`, add one entry to `evaluator.measures` |

Components are resolved by dotted path at runtime, so your class does not need to live inside `lethe/` — any importable module works. A complete worked example of a custom unlearner is in [`notebooks/lethe_quickstart.ipynb`](notebooks/lethe_quickstart.ipynb).

A `GraphUnlearner` subclass receives:

| Attribute | What it is |
| --- | --- |
| `self.predictor` | the trained model to modify (`.model`, `.optimizer`) |
| `self.dataset.partitions[...]` | the `forget` / `retain` / `train` / `test` splits |
| `self.task_loss(node_subset=...)` | the predictor's training loss on a subset of nodes |
| `self.infected_nodes(edges, hops)` | nodes a GNN of this depth can reach the removed edges through |
| `self.hops` | receptive-field depth of the predictor |

> **Note:** attack modules import cleanly only after `lethe.evaluations.manager` has been
> imported (a circular import between `core.measure` and `evaluations.running`). Normal runs
> via `main.py` are unaffected, since the `Evaluator` is constructed before its measures.

---

### Reusable snippets

The `configs/snippets/` directory contains reusable JSON fragments for common unlearner and evaluator configurations (`u_id.json`, `u_gold.json`, `e_umia_graph.json`, `e_umia_edge_graph.json`, `empty_evaluation.json`). These can be composed into any config to avoid repetition.

---

## License

See [LICENSE](LICENSE).
