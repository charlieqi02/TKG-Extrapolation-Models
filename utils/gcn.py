from collections import defaultdict

import dgl
import numpy as np
import torch


def _r2e(snapshot):
    src, rel, dst = snapshot.numpy().transpose()
    # get all relations
    uniq_r = np.unique(rel)
    # generate r2e
    r_to_e = defaultdict(set)
    for (src, rel, dst) in snapshot:
        r_to_e[rel].add(src)
        r_to_e[rel].add(dst)
    r_len = []
    e_idx = []
    idx = 0
    for r in uniq_r:
        r_len.append((idx,idx+len(r_to_e[r])))
        e_idx.extend(list(r_to_e[r]))
        idx += len(r_to_e[r])
    return uniq_r, r_len, e_idx


def build_sub_graph(num_nodes, snapshot, gpu=-1):
    """Build one graph based on given KG.
    
    Args:
        num_nodes: Int number of nodes
        snapshot: torch.LongTensor containing one KG (ent id, rel id, ent id) inversed
        gpu: -1 if on cpu
    
    Returns:
        g: dgl.DGLGraph() containing one graph information
        g additional attributes:
            'norm': reciprocal of a node's in-degree
            'id': node id
            
            .uniq_r: np.array containing all relations' id (inversed)
            .r_len: List containing number of related entities for each relation
            .r2e_idx: List containing related entities' ids for each relation
    """
    def comp_deg_norm(g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        norm = 1.0 / in_deg
        return norm

    src, rel, dst = snapshot.t()

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
    g.edata['type'] = rel

    uniq_r, r_len, r2e_idx = _r2e(snapshot)
    g.uniq_r = uniq_r
    g.r_len = r_len
    g.r2e_idx = torch.from_numpy(np.array(r2e_idx))

    if gpu == -1:
        return g
    return g.to(gpu)


def inverse_snapshot(num_rels, snap, gpu=-1):
    """Add inverse relationship to the triples.
    
    Args:
        num_rels: Int number of relations (non-inversed)
        snap: Non-inversed np.array triples n x (ent id, rel id, ent id)
        gpu: -1 if on cpu
        
    Returns:
        snapshot: inversed torch.LongTensor triples
    """
    src, rel, dst = snap.transpose()
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))
      
    src = torch.from_numpy(src).unsqueeze(1)
    rel = torch.from_numpy(rel).unsqueeze(1)
    dst = torch.from_numpy(dst).unsqueeze(1)
    snapshot = torch.cat([src, rel, dst], dim=1)
    if gpu == -1:
        return snapshot
    return snapshot.to(gpu)