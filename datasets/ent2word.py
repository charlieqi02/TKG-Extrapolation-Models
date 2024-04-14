import os


def load_index(input_path):
    """Get the map of entity/relation to number.
    
    Args:
        input_path: Path to (entity/realtion, id) file
        
    Returns:
        index: Dictionary mapping entity strings to unique ids
        rev_index: Dictionary mapping unique ids to entity strings 
    """
    index, rev_index = {}, {}
    with open(input_path) as f:
        for line in f.readlines():    # shuold be sorted
            rel, id = line.strip().split("\t")
            index[rel] = id
            rev_index[id] = rel
    return index, rev_index


def get_words(entity_strs):
    """Split enetity strings to words. e.g. Citizen_(Nigeria) -> Citizen, Nigeria
    
    Args:
        entity_strs: List containing all entity strings
    
    Returns:
        word2id: Dictionary mapping word strings to unique ids (entity itself is a word)
        id2word: Dictionary mapping unique ids to word strings
    """
    cmlent_count = 0    # complex entity: splitable entity string with '(' and ')'
    sptwrd_count = 0    # spectial word: words that are not in entity set
    word_set = set()
    for entity_str in entity_strs:
        if "(" in entity_str and ")" in entity_str:
            cmlent_count += 1
            begin = entity_str.find('(')
            end = entity_str.find(')')
            w1 = entity_str[:begin].strip()     # remove blank space
            w2 = entity_str[begin+1: end]
            if w2 not in entity_strs:
                print(w2)
                sptwrd_count += 1
            word_set.add(w1)
            word_set.add(w2)
        else:
            word_set.add(entity_str)

    num_word = len(word_set)

    word2id = {word: id for id, word in enumerate(word_set)}
    id2word = {id: word for id, word in enumerate(word_set)}
    
    print("words num: {}, enity_num: {}".format(num_word, len(entity2id.keys())))
    print("num of complex entity / all entity", float(cmlent_count)/len(entity2id.keys()))
    print("num of entities splitable but not in main set / all complex entity", float(sptwrd_count)/float(cmlent_count))
    return word2id, id2word


def get_static(ent_id, wrd_id):
    """Get static graph, map entity ids to word ids, '0' is for isA; '1' is for from; '2' is for itself.
    
    Args:
        ent_id: Tuple(ent2id: Dict, id2ent: Dict)
        wrd_id: Tuple(wrd2id: Dict, id2wrd: Dict)
    
    Returns:
        eid2wid: List containing (entity_id, 0|1|2, word_id)
    """
    ent2id, id2ent = ent_id
    wrd2id, id2wrd = wrd_id
    
    eid2wid = []    # entity-word-graph(static)
    for id in range(len(id2ent.keys())):
        entity_str = id2ent[str(id)]
        if "(" in entity_str and ")" in entity_str:
            begin = entity_str.find('(')
            end = entity_str.find(')')
            w1 = entity_str[:begin].strip()
            w2 = entity_str[begin+1: end]
            eid2wid.append([str(ent2id[entity_str]), "0", str(wrd2id[w1])])
            eid2wid.append([str(ent2id[entity_str]), "1", str(wrd2id[w2])])
        else:
            eid2wid.append([str(ent2id[entity_str]), "2", str(wrd2id[entity_str])])
    return eid2wid


if __name__ == "__main__":
    DATA_BASE = os.environ['DATA_PATH']
    E2I = 'entity2id.txt'
    W2I = 'word2id.txt'
    EWG = 'e-w-graph.txt'
    static_datasets = ['ICEWS14s', 'ICEWS05-15', 'ICEWS18']

    for stc in static_datasets:
        print(f'Processing {stc} ...')
        DATA_SET = os.path.join(DATA_BASE, stc)
        LOAD_E2I = os.path.join(DATA_SET, E2I)
        SAVE_W2I = os.path.join(DATA_SET, W2I)
        SAVE_EWG = os.path.join(DATA_SET, EWG)
        
        entity2id, id2entity = load_index(LOAD_E2I)
        word2id, id2word = get_words(entity2id.keys())
        with open(SAVE_W2I, 'w') as f:
            for word in word2id.keys():
                f.write(word + "\t" + str(word2id[word])+'\n')
        eid2wid = get_static((entity2id, id2entity), (word2id, id2word))
        with open(SAVE_EWG, 'w') as f:
            for line in eid2wid:
                f.write("\t".join(line)+'\n')
