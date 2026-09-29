from abc import ABC, abstractmethod
from collections import Counter, defaultdict
import random
from torch.utils.data import Subset
from lethe.core.base import Configurable
from .Dataset import DatasetWrapper
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import hashlib
import numpy as np

class DataSplitter(ABC):
    def __init__(self, ref_data,parts_names):
        self.ref_data = ref_data
        self.parts_names = parts_names
    
    @abstractmethod
    def split_data(self, data):
        pass

    def set_source(self, datasource):
        self.source = datasource

    
class _EdgeSplitterBase(DataSplitter):
    """Shared machinery for edge-partitioning splitters."""

    def __init__(self, percentage, parts_names, ref_data='all', exclude_parts=None):
        super().__init__(ref_data, parts_names)
        self.percentage = percentage
        self.exclude_parts = exclude_parts or []

    def split_data(self, partitions):
        edges = self.canonical_edges(partitions)
        forget = self.select_forget(partitions, edges)

        forget_set = set(forget)
        retain = [e for e in edges if e not in forget_set]

        return self.emit(partitions, forget, retain, edges)

    @abstractmethod
    def select_forget(self, partitions, edges):
        pass

    def canonical_edges(self, partitions):
        """Canonical undirected (min,max) edge list, honouring ref_data/exclude_parts."""
        edge_index = partitions['all'].data.edge_index
        all_edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

        if self.ref_data != 'all':
            ref_nodes = set(partitions[self.ref_data])
            directed_edges = [(u, v) for u, v in all_edges if u in ref_nodes and v in ref_nodes]
        else:
            directed_edges = all_edges

        undirected_edges = sorted(set((min(u, v), max(u, v)) for u, v in directed_edges))

        if self.exclude_parts:
            excluded = set()
            for part in self.exclude_parts:
                for u, v in partitions.get(part, []):
                    excluded.add((min(u, v), max(u, v)))
            undirected_edges = [e for e in undirected_edges if e not in excluded]

        return undirected_edges

    def emit(self, partitions, forget, retain, edges):
        """Assign the two partitions after asserting forget and retain tile `edges`."""
        if len(forget) + len(retain) != len(edges):
            raise ValueError(
                f"{type(self).__name__}: forget ({len(forget)}) + retain ({len(retain)}) "
                f"!= eligible edges ({len(edges)})")
        if set(forget) & set(retain):
            raise ValueError(f"{type(self).__name__}: forget and retain overlap")

        partitions[self.parts_names[0]] = self._expand_to_directed(forget)
        partitions[self.parts_names[1]] = self._expand_to_directed(retain)
        return partitions

    def budget(self, edges):
        return int(len(edges) * self.percentage)

    def node_attr(self, data, name):
        """Fetch a per-node attribute as a 1-D tensor."""
        for get in (lambda: getattr(data, name),
                    lambda: getattr(data, '_data')[name],
                    lambda: data[0][name]):
            try:
                value = get()
            except (AttributeError, KeyError, TypeError, IndexError):
                continue
            if value is not None:
                return torch.as_tensor(value).reshape(-1)

        available = []
        for probe in (data, getattr(data, '_data', None)):
            if probe is not None and hasattr(probe, 'keys'):
                keys = probe.keys
                available = list(keys() if callable(keys) else keys)
                break
        raise AttributeError(
            f"{type(self).__name__}: node attribute '{name}' not found on the graph. "
            f"Available keys: {available}")

    def _expand_to_directed(self, undirected_edges):
        directed = []
        for u, v in undirected_edges:
            directed.append((u, v))
            if u != v:
                directed.append((v, u))
        return directed

    def shuffle_with_seed(self, indices, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        permuted_order = torch.randperm(len(indices), generator=generator).tolist()
        return [indices[i] for i in permuted_order]

    def get_seed_from_name(self, name):
        hashed_value = int(hashlib.sha256(name.encode()).hexdigest(), 16)
        return hashed_value % (2**32)

    def seeded_shuffle(self, items):
        """Shuffle seeded from the forget partition's name, as the other splitters do."""
        return self.shuffle_with_seed(items, self.get_seed_from_name(self.parts_names[0]))


class DataSplitterPercentage(DataSplitter):
    def __init__(self, percentage, parts_names, ref_data = 'all', shuffle=True, edge_removal = False):
        super().__init__(ref_data,parts_names) 
        self.percentage = percentage
        self.shuffle = shuffle
        self.edge_removal = edge_removal

    def split_data(self,partitions):

        if self.edge_removal:
            edge_index = partitions['all'].data.edge_index
            all_edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
            if self.ref_data != 'all':
                ref_nodes = set(partitions[self.ref_data])
                directed_edges = [(u, v) for u, v in all_edges if u in ref_nodes and v in ref_nodes]
            else:
                directed_edges = all_edges

            # Canonicalize to undirected edges so both directions are kept together
            indices = sorted(set((min(u, v), max(u, v)) for u, v in directed_edges))

        else:
            indices = partitions[self.ref_data] if self.ref_data != 'all' else list(range(len(partitions[self.ref_data].data.x)))


        self.total_size = len(indices)
        split_point = int(self.total_size * self.percentage)

        indices = self.get_indices(indices) if self.shuffle else indices

        split_indices_1 = indices[:split_point]
        split_indices_2 = indices[split_point:]

        if self.edge_removal:
            split_indices_1 = self._expand_to_directed(split_indices_1)
            split_indices_2 = self._expand_to_directed(split_indices_2)

        partitions[self.parts_names[0]] = split_indices_1
        partitions[self.parts_names[1]] = split_indices_2

        return partitions
    
    def _expand_to_directed(self, undirected_edges):
        """Expand undirected (min,max) edges back to both directions."""
        directed = []
        for u, v in undirected_edges:
            directed.append((u, v))
            if u != v:
                directed.append((v, u))
        return directed

    def get_indices(self, indices):
        seed = self.get_seed_from_name(self.parts_names[0])
        return self.shuffle_with_seed(indices, seed)

    def shuffle_with_seed(self, indices, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)

        permuted_order = torch.randperm(len(indices), generator=generator).tolist()

        shuffled_indices = [indices[i] for i in permuted_order]

        return shuffled_indices

    def get_seed_from_name(self, name):
        hashed_value = int(hashlib.sha256(name.encode()).hexdigest(), 16)
        return hashed_value % (2**32)


class DataSplitterCyclicEdges(DataSplitter):
    """Selects edges that participate in n-cycles, fewest cycles first."""

    def __init__(self, n, parts_names, ref_data='all', percentage=1.0):
        super().__init__(ref_data, parts_names)
        self.n = n
        self.percentage = percentage

    def split_data(self, partitions):
        edge_index = partitions['all'].data.edge_index
        all_edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

        if self.ref_data != 'all':
            ref_nodes = set(partitions[self.ref_data])
            directed_edges = [(u, v) for u, v in all_edges if u in ref_nodes and v in ref_nodes]
        else:
            directed_edges = all_edges

        undirected_edges = sorted(set((min(u, v), max(u, v)) for u, v in directed_edges))

        cycle_counts = self._compute_cycle_counts(partitions['all'].data, undirected_edges)

        # Separate cycle-edges (sorted ascending by count) from non-cycle edges
        cycle_edges = sorted(
            [(e, c) for e, c in zip(undirected_edges, cycle_counts) if c > 0],
            key=lambda x: x[1]
        )
        non_cycle_edges = [e for e, c in zip(undirected_edges, cycle_counts) if c == 0]

        split_point = int(len(cycle_edges) * self.percentage)
        selected  = [e for e, _ in cycle_edges[:split_point]]
        remaining = [e for e, _ in cycle_edges[split_point:]] + non_cycle_edges

        partitions[self.parts_names[0]] = self._expand_to_directed(selected)
        partitions[self.parts_names[1]] = self._expand_to_directed(remaining)

        return partitions

    def _compute_cycle_counts(self, data, undirected_edges):
        """Return [A^{n-1}]_{u,v} for each (u,v) via sparse MV products."""
        from collections import defaultdict

        N = data.x.size(0)
        edge_index = data.edge_index

        vals = torch.ones(edge_index.size(1), dtype=torch.float32, device=edge_index.device)
        A = torch.sparse_coo_tensor(edge_index, vals, (N, N)).coalesce()

        u_groups = defaultdict(list)
        for idx, (u, v) in enumerate(undirected_edges):
            u_groups[u].append((idx, v))

        counts = [0.0] * len(undirected_edges)
        for u, pairs in u_groups.items():
            e_u = torch.zeros(N, dtype=torch.float32, device=edge_index.device)
            e_u[u] = 1.0

            # Compute A^{n-1} e_u via n-1 sparse MV products
            vec = e_u
            for _ in range(self.n - 1):
                vec = torch.sparse.mm(A, vec.unsqueeze(1)).squeeze(1)

            # [A^{n-1}]_{u,v} = (A^{n-1} e_u)[v]  (valid because A is symmetric)
            for idx, v in pairs:
                counts[idx] = vec[v].item()

        return counts

    def _expand_to_directed(self, undirected_edges):
        directed = []
        for u, v in undirected_edges:
            directed.append((u, v))
            if u != v:
                directed.append((v, u))
        return directed


class DataSplitterEdgeHoldout(DataSplitter):
    """Holds out link-prediction supervision edges: test / validation / train."""

    def __init__(self, parts_names, test_percentage=0.1, val_percentage=0.05,
                 ref_data='all'):
        super().__init__(ref_data, parts_names)
        self.test_percentage = test_percentage
        self.val_percentage = val_percentage

    def split_data(self, partitions):
        edge_index = partitions['all'].data.edge_index
        all_edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

        if self.ref_data != 'all':
            ref_nodes = set(partitions[self.ref_data])
            directed_edges = [(u, v) for u, v in all_edges if u in ref_nodes and v in ref_nodes]
        else:
            directed_edges = all_edges

        undirected_edges = sorted(set((min(u, v), max(u, v)) for u, v in directed_edges))

        seed = self.get_seed_from_name(self.parts_names[0])
        undirected_edges = self.shuffle_with_seed(undirected_edges, seed)

        total = len(undirected_edges)
        n_test = int(total * self.test_percentage)
        n_val = int(total * self.val_percentage)

        chunks = [undirected_edges[:n_test],
                  undirected_edges[n_test:n_test + n_val],
                  undirected_edges[n_test + n_val:]]

        for name, chunk in zip(self.parts_names, chunks):
            partitions[name] = self._expand_to_directed(chunk)

        return partitions

    def _expand_to_directed(self, undirected_edges):
        directed = []
        for u, v in undirected_edges:
            directed.append((u, v))
            if u != v:
                directed.append((v, u))
        return directed

    def shuffle_with_seed(self, indices, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        permuted_order = torch.randperm(len(indices), generator=generator).tolist()
        return [indices[i] for i in permuted_order]

    def get_seed_from_name(self, name):
        hashed_value = int(hashlib.sha256(name.encode()).hexdigest(), 16)
        return hashed_value % (2**32)


class DataSplitterEdgeDifficulty(DataSplitter):

    def __init__(self, percentage, parts_names, ref_data='all', mode='hard', k=2,
                 exclude_parts=None):
        super().__init__(ref_data, parts_names)
        self.percentage = percentage
        self.mode = mode
        self.k = k
        self.exclude_parts = exclude_parts or []

    def split_data(self, partitions):
        edge_index = partitions['all'].data.edge_index
        all_edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

        if self.ref_data != 'all':
            ref_nodes = set(partitions[self.ref_data])
            directed_edges = [(u, v) for u, v in all_edges if u in ref_nodes and v in ref_nodes]
        else:
            directed_edges = all_edges

        # Canonicalize to undirected (min, max) pairs
        undirected_edges = sorted(set((min(u, v), max(u, v)) for u, v in directed_edges))

        if self.exclude_parts:
            excluded = set()
            for part in self.exclude_parts:
                for u, v in partitions.get(part, []):
                    excluded.add((min(u, v), max(u, v)))
            undirected_edges = [e for e in undirected_edges if e not in excluded]

        if self.mode == 'simple':
            seed = self.get_seed_from_name(self.parts_names[0])
            undirected_edges = self.shuffle_with_seed(undirected_edges, seed)
        else:  # 'hard' or 'easy': sort by walk centrality
            centralities = self._compute_walk_centrality(partitions['all'].data, undirected_edges)
            descending = (self.mode != 'easy')
            undirected_edges = [e for _, e in sorted(zip(centralities, undirected_edges), reverse=descending)]

        split_point = int(len(undirected_edges) * self.percentage)
        partitions[self.parts_names[0]] = self._expand_to_directed(undirected_edges[:split_point])
        partitions[self.parts_names[1]] = self._expand_to_directed(undirected_edges[split_point:])

        return partitions

    def _compute_walk_centrality(self, data, undirected_edges):
        N = data.x.size(0)
        edge_index = data.edge_index

        # A_tilde = A + I
        self_loops = torch.arange(N, device=edge_index.device)
        loop_index = torch.stack([self_loops, self_loops])
        edge_index_tilde = torch.cat([edge_index, loop_index], dim=1)

        # Symmetric normalisation: A_hat = D_tilde^{-1/2} A_tilde D_tilde^{-1/2}
        row, col = edge_index_tilde
        deg = torch.zeros(N, device=edge_index.device)
        deg.scatter_add_(0, row, torch.ones(edge_index_tilde.size(1), device=edge_index.device))
        deg_inv_sqrt = deg.pow(-0.5).clamp(max=1e9)

        weights = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        A_hat = torch.sparse_coo_tensor(edge_index_tilde, weights, (N, N)).coalesce()

        # r_p = A_hat^p * 1  for p = 0, ..., k-1
        r = [torch.ones(N, device=edge_index.device)]
        for _ in range(self.k - 1):
            r.append(torch.sparse.mm(A_hat, r[-1].unsqueeze(1)).squeeze(1))

        # WalkCentrality(i,j) = sum_t r_t[i] * r_{k-1-t}[j]
        centralities = [
            sum(r[t][u].item() * r[self.k - 1 - t][v].item() for t in range(self.k))
            for (u, v) in undirected_edges
        ]

        return centralities

    def _expand_to_directed(self, undirected_edges):
        directed = []
        for u, v in undirected_edges:
            directed.append((u, v))
            if u != v:
                directed.append((v, u))
        return directed

    def shuffle_with_seed(self, indices, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        permuted_order = torch.randperm(len(indices), generator=generator).tolist()
        return [indices[i] for i in permuted_order]

    def get_seed_from_name(self, name):
        hashed_value = int(hashlib.sha256(name.encode()).hexdigest(), 16)
        return hashed_value % (2**32)


class DataSplitterEdgeTemporal(_EdgeSplitterBase):
    """Forget set drawn from a contiguous slice of the graph's timeline."""

    def __init__(self, percentage, parts_names, ref_data='all', mode='recent',
                 time_attr='node_year', edge_time='max', window=None,
                 window_pad='nearest', missing='exclude', exclude_parts=None):
        super().__init__(percentage, parts_names, ref_data, exclude_parts)
        if mode not in ('recent', 'oldest', 'window'):
            raise ValueError(f"mode must be recent|oldest|window, got {mode!r}")
        if mode == 'window' and window is None:
            raise ValueError("mode='window' requires the `window` parameter")
        self.mode = mode
        self.time_attr = time_attr
        self.edge_time = edge_time
        self.window = window
        self.window_pad = window_pad
        self.missing = missing

    def select_forget(self, partitions, edges):
        times = self.node_attr(partitions['all'].data, self.time_attr)
        reduce = torch.maximum if self.edge_time == 'max' else torch.minimum

        src = torch.tensor([u for u, _ in edges], dtype=torch.long)
        dst = torch.tensor([v for _, v in edges], dtype=torch.long)
        t_edge = reduce(times[src], times[dst]).tolist()

        valid = [t > 0 and t == t for t in t_edge]           # t == t rejects NaN
        if not all(valid) and self.missing == 'median':
            median = float(np.median([t for t, ok in zip(t_edge, valid) if ok]))
            t_edge = [t if ok else median for t, ok in zip(t_edge, valid)]
            valid = [True] * len(t_edge)
        eligible = [(e, t) for e, t, ok in zip(edges, t_edge, valid) if ok]
        budget = self.budget(edges)

        if self.mode == 'window':
            return self._select_window(eligible, budget)

        ordered = sorted(self.seeded_shuffle(eligible), key=lambda pair: pair[1])
        chosen = ordered[-budget:] if self.mode == 'recent' else ordered[:budget]
        return [e for e, _ in chosen]

    def _select_window(self, eligible, budget):
        in_window = [(e, t) for e, t in eligible if t == self.window]

        if len(in_window) >= budget:
            return [e for e, _ in self.seeded_shuffle(in_window)[:budget]]

        if self.window_pad != 'nearest':
            return [e for e, _ in in_window]

        # Pad outwards from the window by temporal distance, ties shuffled.
        rest = self.seeded_shuffle([(e, t) for e, t in eligible if t != self.window])
        rest.sort(key=lambda pair: abs(pair[1] - self.window))
        padded = in_window + rest[:budget - len(in_window)]
        return [e for e, _ in padded]


class DataSplitterEdgeGroup(_EdgeSplitterBase):
    """Forget set scoped to one cohort of nodes."""

    BUDGET_TOLERANCE = 0.01

    def __init__(self, percentage, parts_names, ref_data='all', mode='incident',
                 group_attr='y', group=None, select='nearest', fill='none',
                 exclude_parts=None):
        super().__init__(percentage, parts_names, ref_data, exclude_parts)
        if mode not in ('intra', 'incident'):
            raise ValueError(f"mode must be intra|incident, got {mode!r}")
        if select not in ('nearest', 'largest', 'smallest', 'median'):
            raise ValueError(f"select must be nearest|largest|smallest|median, got {select!r}")
        self.mode = mode
        self.group_attr = group_attr
        self.group = group
        self.select = select
        self.fill = fill

    def select_forget(self, partitions, edges):
        groups = self.node_attr(partitions['all'].data, self.group_attr).long().tolist()
        budget = self.budget(edges)

        wanted = self._resolve_groups(self._eligible_counts(edges, groups), budget)
        eligible = [e for e in edges if self._matches(e, groups, wanted)]

        if len(eligible) < budget and self.fill == 'none':
            return eligible

        return self.seeded_shuffle(eligible)[:budget]

    def _matches(self, edge, groups, wanted):
        u, v = edge
        if self.mode == 'intra':
            return groups[u] in wanted and groups[v] in wanted
        return groups[u] in wanted or groups[v] in wanted

    def _eligible_counts(self, edges, groups):
        """Per-group eligible edge count under the active mode."""
        counts = Counter()
        for u, v in edges:
            gu, gv = groups[u], groups[v]
            if self.mode == 'intra':
                if gu == gv:
                    counts[gu] += 1
            else:
                counts[gu] += 1
                if gv != gu:
                    counts[gv] += 1
        for g in set(groups):
            counts.setdefault(g, 0)
        return counts

    def _resolve_groups(self, counts, budget):
        if self.group is not None:
            return {self.group} if isinstance(self.group, int) else set(self.group)

        table = sorted(counts.items(), key=lambda kv: -kv[1])

        floor = budget * (1 - self.BUDGET_TOLERANCE)
        viable = [g for g, c in table if c >= floor]

        if not viable:
            chosen = [table[0][0]]
        elif self.select == 'nearest':
            chosen = [viable[-1]]
        elif self.select == 'largest':
            chosen = [viable[0]]
        elif self.select == 'smallest':
            chosen = [viable[-1]]
        else:
            chosen = [viable[len(viable) // 2]]

        # Merge in further groups only if a single one cannot fund the budget.
        if self.fill == 'next_group' and counts[chosen[0]] < budget:
            total = counts[chosen[0]]
            for g, c in table:
                if g in chosen:
                    continue
                chosen.append(g)
                total += c
                if total >= budget:
                    break

        return set(chosen)


class DataSplitterEdgeUserDeletion(_EdgeSplitterBase):
    """Forget set built from whole node neighbourhoods -- account deletion."""

    def __init__(self, percentage, parts_names, ref_data='all', mode='random',
                 group_attr=None, group=None, exclude_parts=None):
        super().__init__(percentage, parts_names, ref_data, exclude_parts)
        if mode not in ('random', 'high_degree', 'low_degree'):
            raise ValueError(f"mode must be random|high_degree|low_degree, got {mode!r}")
        self.mode = mode
        self.group_attr = group_attr
        self.group = group

    def select_forget(self, partitions, edges):
        incident = defaultdict(list)
        for edge in edges:
            u, v = edge
            incident[u].append(edge)
            if v != u:
                incident[v].append(edge)

        candidates = self._candidates(partitions, incident)
        budget = self.budget(edges)

        forget, seen = [], set()
        for node in candidates:
            if len(forget) >= budget:
                break
            for edge in incident[node]:
                if edge not in seen:
                    seen.add(edge)
                    forget.append(edge)
        return forget

    def _candidates(self, partitions, incident):
        nodes = sorted(incident)

        if self.group is not None:
            if self.group_attr is None:
                raise ValueError("`group` requires `group_attr`")
            groups = self.node_attr(partitions['all'].data, self.group_attr).long().tolist()
            wanted = {self.group} if isinstance(self.group, int) else set(self.group)
            in_cohort = [n for n in nodes if groups[n] in wanted]
            nodes = self.seeded_shuffle(in_cohort) + self.seeded_shuffle(
                [n for n in nodes if groups[n] not in wanted])
        else:
            nodes = self.seeded_shuffle(nodes)

        if self.mode == 'random':
            return nodes
        # Stable sort over the shuffled order: ties in degree stay randomised.
        return sorted(nodes, key=lambda n: len(incident[n]),
                      reverse=(self.mode == 'high_degree'))
