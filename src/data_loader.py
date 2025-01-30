import collections
import os
import numpy as np
import networkx as nx
from itertools import cycle, islice
import time


def load_data(args):
    train_data, eval_data, test_data, user_history_dict = load_rating(args)
    n_entity, n_relation, kg = load_kg(args)
    ripple_set = get_ripple_set(args, kg, user_history_dict)
    return train_data, eval_data, test_data, n_entity, n_relation, ripple_set


def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    rating_file = '../data/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_np)

    # n_user = len(set(rating_np[:, 0]))
    # n_item = len(set(rating_np[:, 1]))
    return dataset_split(rating_np)


def dataset_split(rating_np):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    # print(len(train_indices), len(eval_indices), len(test_indices))

    # traverse training data, only keeping the users with positive ratings
    user_history_dict = dict()
    for i in train_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]
        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            user_history_dict[user].append(item)

    train_indices = [i for i in train_indices if rating_np[i][0] in user_history_dict]
    eval_indices = [i for i in eval_indices if rating_np[i][0] in user_history_dict]
    test_indices = [i for i in test_indices if rating_np[i][0] in user_history_dict]
    # print(len(train_indices), len(eval_indices), len(test_indices))

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data, user_history_dict

def load_kg_raw(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = '../data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)
    return kg_np

def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = '../data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    kg = construct_kg(kg_np)

    return n_entity, n_relation, kg


def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))
    return kg

def loop_slice(l, start=0, end=0):
    return list(islice(cycle(l), start, end))

def get_ripple_set(args, kg, user_history_dict):
    print(f'constructing ripple set (sampler {args.sampler}) ...')

    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    ripple_set = collections.defaultdict(list)

    # ---- dodane ----
    kg_graph = nx.DiGraph()
    for h in kg:
        for r, t in kg[h]:
            kg_graph.add_edge(h, t, relation=r)

    if args.sampler == 5:
        kg_np = load_kg_raw(args)
        node_ids = set(np.concatenate([kg_np[:, 0], kg_np[:, 2]]))
        out_degrees = {k: kg_graph.out_degree(k) if k in kg_graph.nodes else 0 for k in node_ids}

    if args.sampler == 6:
        kg_np = load_kg_raw(args)
        node_ids = set(np.concatenate([kg_np[:, 0], kg_np[:, 2]]))
        in_degrees = {k: kg_graph.in_degree(k) if k in kg_graph.nodes else 0 for k in node_ids}
    # ---- koniec ----

    for uidxd, user in enumerate(user_history_dict):
        print(f"{uidxd}/{len(user_history_dict)}      ", end='\r')
        for h in range(args.n_hop):
            memories_h = []
            memories_r = []
            memories_t = []

            if h == 0:
                tails_of_last_hop = user_history_dict[user]
            else:
                tails_of_last_hop = ripple_set[user][-1][2]

            for entity in tails_of_last_hop:
                for tail_and_relation in kg[entity]:
                    memories_h.append(entity)
                    memories_r.append(tail_and_relation[1])
                    memories_t.append(tail_and_relation[0])

            # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
            # this won't happen for h = 0, because only the items that appear in the KG have been selected
            # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
            if len(memories_h) == 0:
                ripple_set[user].append(ripple_set[user][-1])
            else:
                # sample a fixed-size 1-hop memory for each user
                if args.sampler == 0:
                    replace = len(memories_h) < args.n_memory
                    indices = np.random.choice(len(memories_h), size=args.n_memory, replace=replace)

                elif args.sampler == 1: # nawiecej out_degree
                    memories_h_out_degs = {i: kg_graph.out_degree(head) for i, head in enumerate(memories_h)}
                    memories_h_out_degs = {k: v for k, v in sorted(memories_h_out_degs.items(), key=lambda m: m[1])}

                    indices = loop_slice(memories_h_out_degs.keys(), end=args.n_memory)

                elif args.sampler == 2: # nawiecej in_degree
                    memories_h_in_degs = {i: kg_graph.in_degree(head) for i, head in enumerate(memories_h)}
                    memories_h_in_degs = {k: v for k, v in sorted(memories_h_in_degs.items(), key=lambda m: m[1])}

                    indices = loop_slice(memories_h_in_degs.keys(), end=args.n_memory)

                elif args.sampler == 3: # nawiecej out_degree
                    memories_h_out_degs = {i: kg_graph.out_degree(head) for i, head in enumerate(memories_t) if head in kg_graph.nodes}
                    memories_h_out_degs = {k: v for k, v in sorted(memories_h_out_degs.items(), key=lambda m: m[1])}

                    indices = loop_slice(memories_h_out_degs.keys(), end=args.n_memory)

                elif args.sampler == 4: # nawiecej in_degree
                    memories_h_in_degs = {i: kg_graph.in_degree(head) for i, head in enumerate(memories_t) if head in kg_graph.nodes}
                    memories_h_in_degs = {k: v for k, v in sorted(memories_h_in_degs.items(), key=lambda m: m[1])}

                    indices = loop_slice(memories_h_in_degs.keys(), end=args.n_memory)

                elif args.sampler == 5: # różne nody - out-degree
                    heads_tails = collections.defaultdict(list)

                    # odfiltrowanie sink entities
                    for i, (h, t) in enumerate(zip(memories_h, memories_t)):
                        # if t in kg_graph.nodes:
                            # heads_tails[h].append(i)
                        heads_tails[h].append(i)
                    # print('filtering done')

                    # sortowanie wg out degree
                    for h in heads_tails:
                        heads_tails[h] = sorted(heads_tails[h], key=lambda ht: out_degrees[memories_t[ht]]) # kg_graph.out_degree(memories_t[ht]))
                    # print('sort 1 done')

                    # sortowanie wg liczby tails
                    heads_tails = {k: v for k, v in sorted(heads_tails.items(), key=lambda ht: len(ht[1]))}
                    # print('sort 2 done')

                    memories_h_tmp = []
                    indices = []
                    heads_keys = list(heads_tails.keys())
                    num_heads = len(heads_keys)
                    hid = 0
                    tid = 0
                    max_tid = max([len(v) for v in heads_tails.values()])
                    while len(indices) < args.n_memory:
                        tid = hid // num_heads
                        hid_key = heads_keys[hid % num_heads]
                        # print(f"indices: {len(indices)} / {args.n_memory}, {hid=}, {tid=}, {hid_key=}     ", end='\r')
                        # time.sleep(0.1)
                        if tid < len(heads_tails[hid_key]):
                            memories_h_tmp.append(hid_key)
                            indices.append(heads_tails[hid_key][tid])
                        hid += 1
                        if tid > max_tid:
                            hid = 0
                            tid = 0

                elif args.sampler == 6: # różne nody - in degree
                    heads_tails = collections.defaultdict(list)

                    # odfiltrowanie sink entities
                    for i, (h, t) in enumerate(zip(memories_h, memories_t)):
                        # if t in kg_graph.nodes:
                            # heads_tails[h].append(i)
                        heads_tails[h].append(i)
                    # print('filtering done')

                    # sortowanie wg out degree
                    for h in heads_tails:
                        heads_tails[h] = sorted(heads_tails[h], key=lambda ht: in_degrees[memories_t[ht]]) # kg_graph.out_degree(memories_t[ht]))
                    # print('sort 1 done')

                    # sortowanie wg liczby tails
                    heads_tails = {k: v for k, v in sorted(heads_tails.items(), key=lambda ht: len(ht[1]))}
                    # print('sort 2 done')

                    memories_h_tmp = []
                    indices = []
                    heads_keys = list(heads_tails.keys())
                    num_heads = len(heads_keys)
                    hid = 0
                    tid = 0
                    max_tid = max([len(v) for v in heads_tails.values()])
                    while len(indices) < args.n_memory:
                        tid = hid // num_heads
                        hid_key = heads_keys[hid % num_heads]
                        # print(f"indices: {len(indices)} / {args.n_memory}, {hid=}, {tid=}, {hid_key=}     ", end='\r')
                        # time.sleep(0.1)
                        if tid < len(heads_tails[hid_key]):
                            memories_h_tmp.append(hid_key)
                            indices.append(heads_tails[hid_key][tid])
                        hid += 1
                        if tid > max_tid:
                            hid = 0
                            tid = 0

                elif args.sampler == 7: # różne nody - random
                    heads_tails = collections.defaultdict(list)

                    # odfiltrowanie sink entities
                    for i, (h, t) in enumerate(zip(memories_h, memories_t)):
                        # if t in kg_graph.nodes:
                            # heads_tails[h].append(i)
                        heads_tails[h].append(i)
                    # print('filtering done')

                    # sortowanie wg out degree
                    # for h in heads_tails:
                    #     heads_tails[h] = sorted(heads_tails[h], key=lambda ht: in_degrees[memories_t[ht]]) # kg_graph.out_degree(memories_t[ht]))
                    # print('sort 1 done')

                    # sortowanie wg liczby tails
                    heads_tails = {k: v for k, v in sorted(heads_tails.items(), key=lambda ht: len(ht[1]))}
                    # print('sort 2 done')

                    memories_h_tmp = []
                    indices = []
                    heads_keys = list(heads_tails.keys())
                    num_heads = len(heads_keys)
                    hid = 0
                    while len(indices) < args.n_memory:
                        hid_key = heads_keys[hid % num_heads]
                        memories_h_tmp.append(hid_key)
                        indices.append(np.random.choice(heads_tails[hid_key]))
                        hid += 1

                elif args.sampler == 8:
                    memories_h = []
                    memories_r = []
                    memories_t = []

                    if h == 0:
                        tails_of_last_hop = user_history_dict[user]
                    else:
                        tails_of_last_hop = ripple_set[user][-1][2]

                    # odfiltrowanie sink entities
                    for entity in tails_of_last_hop:
                        for tail_and_relation in kg[entity]:
                            if tail_and_relation[0] in kg_graph.nodes:
                                memories_h.append(entity)
                                memories_r.append(tail_and_relation[1])
                                memories_t.append(tail_and_relation[0])
                                
                    replace = len(memories_h) < args.n_memory
                    indices = np.random.choice(len(memories_h), size=args.n_memory, replace=replace)

                memories_h = memories_h_tmp if args.sampler in [5, 6, 7] else [memories_h[i] for i in indices]
                memories_r = [memories_r[i] for i in indices]
                memories_t = [memories_t[i] for i in indices]
                ripple_set[user].append((memories_h, memories_r, memories_t))

    return ripple_set
