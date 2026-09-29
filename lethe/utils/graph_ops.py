"""Vectorised graph neighbourhood ops."""

import torch


def khop_infected(edge_index, seed_nodes, hops, num_nodes):
    """Nodes reachable from `seed_nodes` in at most `hops` undirected steps."""
    device = edge_index.device
    if edge_index.numel() == 0:
        return []

    src = torch.cat([edge_index[0], edge_index[1]])
    dst = torch.cat([edge_index[1], edge_index[0]])

    seed_nodes = torch.as_tensor(seed_nodes, dtype=torch.long, device=device).reshape(-1)

    has_edge = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    has_edge[src] = True

    reach = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    reach[seed_nodes] = True
    reach &= has_edge          # isolated seeds are not in the nx graph

    frontier = reach.clone()
    for _ in range(hops):
        nxt = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        nxt[dst[frontier[src]]] = True
        frontier = nxt & ~reach
        if not bool(frontier.any()):
            break
        reach |= frontier

    return reach.nonzero(as_tuple=False).flatten().tolist()


def edge_endpoints(edges):
    """Distinct node ids appearing in an iterable of (u, v) pairs."""
    if isinstance(edges, torch.Tensor):
        return torch.unique(edges.reshape(-1)).tolist()
    nodes = set()
    for u, v in edges:
        nodes.add(int(u))
        nodes.add(int(v))
    return sorted(nodes)
