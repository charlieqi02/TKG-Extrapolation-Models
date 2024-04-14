import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.gcn import build_sub_graph, inverse_snapshot

from ._knowledge_graph import _read_triplets_as_list


def load_auxdata(args):
    args.aux = {}
    aux_data = {}
    data_path = os.environ['DATA_PATH']

    if args.use_static:
        logging.info("Loading static graph ...")
        static_path = os.path.join(data_path, args.dataset, "e-w-graph.txt")
        static_triples = np.array(_read_triplets_as_list(static_path, load_time=False))
        logging.info(f"Static num of facts: {static_triples.shape[0]}")
        
        static_num_rels = len(np.unique(static_triples[:, 1]))
        static_num_words = len(np.unique(static_triples[:, 2]))
        args.aux['static_size'] = (static_num_words, static_num_rels)
        logging.info(f"Static info: (num ents, static rels: {static_num_rels}, num words: {static_num_words})")
        
        static_triples[:, 2] = static_triples[:, 2] + args.sizes[0]
        static_triples = inverse_snapshot(static_num_rels, static_triples)
        static_graph = build_sub_graph(static_num_words + args.sizes[0], static_triples, args.gpu)
        aux_data['static'] = static_graph
    else:
        raise NotImplementedError
        
    return aux_data, args

