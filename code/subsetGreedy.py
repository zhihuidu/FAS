#!/usr/bin/env pypy

import csv
import random
from itertools import permutations
from collections import defaultdict

def create_adjacency_lists(csv_file_path):
    node_list = set()
    edges = []

    with open(csv_file_path, mode='r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)

        for row in csvreader:
            source, target, weight = row
            source = int(source)
            target = int(target)
            weight = int(weight)

            edges.append((source, target, weight))
            node_list.add(source)
            node_list.add(target)

    node_list = sorted(list(node_list))
    node_to_index = {node: idx for idx, node in enumerate(node_list)}
    index_to_node = {idx: node for idx, node in enumerate(node_list)}

    n = len(node_list)
    out_adj = [[] for _ in range(n)]
    in_adj = [[] for _ in range(n)]

    for source, target, weight in edges:
        src_idx = node_to_index[source]
        tgt_idx = node_to_index[target]
        out_adj[src_idx].append((tgt_idx, weight))
        in_adj[tgt_idx].append((src_idx, weight))

    return out_adj, in_adj, list(node_list), edges, node_to_index, index_to_node

def generate_random_permutation(n):
    permutation = list(range(n))
    random.shuffle(permutation)
    pos = [0] * n
    for idx, node in enumerate(permutation):
        pos[node] = idx
    return permutation, pos

def write_ordered_nodes_to_csv(permutation, pos, output_csv_file, index_to_node):
    with open(output_csv_file, mode='w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['node_id', 'order in the permutation'])
        for idx in range(len(permutation)):
            node = index_to_node[idx]
            csvwriter.writerow([node, pos[idx]])
 

csv_file_path = '../connectome_graph.csv'
output_csv_file = 'ordered-nodes.csv'

out_adj, in_adj, node_list, edges, node_to_index, index_to_node = create_adjacency_lists(csv_file_path)

n = len(node_list)
permutation, pos = generate_random_permutation(n)

#with open('ordered-nodes.csv', 'r', encoding='utf-8') as f:
#    csv_reader = csv.reader(f)
#    next(csv_reader)  # Skip header row
#    for (node, order) in csv_reader:
#        node=(int)(node)
#        pos[node_to_index[node]]=(int)(order)
    
state=pos

def initial_score():
    e = 0
    for source, target, weight in edges:
        src_idx = node_to_index[source]
        tgt_idx = node_to_index[target]
        if state[src_idx] < state[tgt_idx]:
            e += weight
    return e

def compute_change(nodes, prev, cur):
    delta = 0
    for i in range(len(nodes)):
        node = nodes[i]
        prevPos = prev[i]
        curPos = cur[i]
        for target, weight in out_adj[node]:
            if prevPos < state[target]:
                delta -= weight
            if curPos < state[target]:
                delta += weight
        for source, weight in in_adj[node]:
            if source in nodes:
                continue
            if state[source] < prevPos:
                delta -= weight
            if state[source] < curPos:
                delta += weight
    return delta

def move():
    # Select four random indices
    indices = random.sample(range(n), 4)
    current_order = [state[idx] for idx in indices]
    best_order = current_order[:]
    best_delta = 0    
    
    for perm in permutations(current_order):
        delta = compute_change(indices, current_order, perm)
        if delta > best_delta:
            best_delta = delta
            best_order = perm
        
    for i, node in enumerate(indices):
        state[node] = best_order[i]
    return best_delta

num_iters = 50000000000
output_frac = 100

score = initial_score()

try:
    for iter in range(0,num_iters):
        score+=move()
        if (iter%(output_frac))==0:
            print("Iteration",iter,score)

finally:
    write_ordered_nodes_to_csv(permutation, pos, output_csv_file, index_to_node)

    print(f"Ordered nodes saved to {output_csv_file}")
    print(f"Final score: {score}\n")
