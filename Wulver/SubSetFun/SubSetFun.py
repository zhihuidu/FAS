import pandas as pd
import networkx as nx
from networkx.algorithms.cycles import simple_cycles
import gurobipy as gp
from gurobipy import GRB
import sys
from datetime import datetime
import time
import csv
import random
from itertools import permutations


FileNameHead="SetFunc"

#given a graph file in the csv format, generate the graph data structure
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
            weight = int(weight) #only weight should be integer, vertex ID can be a string
            if source != target :
                edges.append((source, target, weight))
            node_list.add(source)
            node_list.add(target)

    node_list = list(node_list)

    #here we merge multiple edges
    merged_edges = {}
    for source, dest, weight in edges:
        if (source, dest) in merged_edges:
            merged_edges[(source, dest)] += weight
        else:
            merged_edges[(source, dest)] = weight
    merged_edges_list = [(source, dest, weight) for (source, dest), weight in merged_edges.items()]

    in_adj={}
    out_adj={}
    for node in node_list:
        out_adj[node]=[]
        in_adj[node]=[]
    for source, target, weight in merged_edges_list:
        out_adj[source].append((target, weight))
        in_adj[target].append((source, weight))

    print("node list",node_list)
    print("merged_edges_list",merged_edges_list)
    print("in_adj",in_adj)
    print("out_adj",out_adj)

    return node_list, merged_edges_list, in_adj, out_adj 

def get_lvs_mvs_rvs(nodes, in_edges,out_edges):
    lvs = [node for node in nodes if in_edges[node] == [] and out_edges[node]!=[] ]
    #vertices in the left vertex set (lvs) should have small index without refining

    rvs = [node for node in nodes if len(out_edges[node]) == 0 and len(in_edges[node]) > 0]
    #vertices in the right vertex set (rvs) should have large index withoug refining

    mvs = [node for node in nodes if len(in_edges[node])>0 and len(out_edges[node])>0]
    #vertices in the middle vertex set (mvs) should have middle index and we need to refine them

    return lvs, mvs, rvs


def total_weight(edges):
    return sum(w for u, v, w in edges)


def average_node_weight(nodes, edges):
    return total_weight(edges)/len(nodes)
    #the average weight for one node


def vertex_weight(node, in_edges,out_edges):
    return sum(w for (v,w) in in_edges[node])+ sum(w for (v,w) in out_edges[node])
    #all weights associated with one vertex


def split_set_by_weight(originalset, split_weight,in_edges,out_edges):
    #split a given set into two subsets, one with relative higher weight for each vertex
    #another set has lower weight for each vertex
    Hvs=[]
    Lvs=[]
    for i in originalset:
        if vertex_weight(i,in_edges,out_edges)>split_weight:
            Hvs.append(i)
        else:
            Lvs.append(i)
    return Hvs, Lvs


def init_segment_based_mapping (nodes, edges,in_edges,out_edges):
    lvs,mvs,rvs = get_lvs_mvs_rvs(nodes, in_edges,out_edges)
    average_weight = average_node_weight(nodes,edges)
    Hvs,Lvs = split_set_by_weight(mvs, average_weight,in_edges,out_edges)
    id_mapping = {node: idx for idx, node in enumerate(lvs + Hvs+Lvs + rvs)}
    #mapping from vertex ID/string to index

    reverse_mapping = {idx: node for node, idx in id_mapping.items()}
    #mapping from index to vertex ID

    return id_mapping,reverse_mapping



def sum_of_weight_current_mapping (edges,id_mapping):
    w = 0
    for source, target, weight in edges:
        src_idx = id_mapping[source]
        tgt_idx = id_mapping[target]
        if src_idx < tgt_idx:
            w += weight
    return w


def compute_change(nodes, prev, cur,in_adj, out_adj,id_mapping):
    #for the vertex id set nodes={a,b,c,d}, prev=[10,3,9,4] is their corresponding index
    #cur=[9,10,4,3] is their shuffled index, we calculate the weigh changes
    delta = 0
    for i in range(len(nodes)):
        node = nodes[i]
        prevPos = prev[i]
        curPos = cur[i]
        for target, weight in out_adj[node]:
            if prevPos < id_mapping[target]:
                delta -= weight
            if curPos < id_mapping[target]:
                delta += weight
        for source, weight in in_adj[node]:
            if source in nodes:
                continue
            if id_mapping[source] < prevPos:
                delta -= weight
            if id_mapping[source] < curPos:
                delta += weight
    return delta

def randommove(vertexset,in_edges,out_edges,id_mapping):
    # vertexset is the ID set whose ID will be randomly rearrange their indices.
    current_order=[]
    for i in vertexset:
        current_order.append(id_mapping[i])
    best_order = current_order[:]
    best_delta = 0    
    
    for perm in permutations(current_order):
        delta = compute_change(list(vertexset), current_order, perm,in_edges,out_edges,id_mapping)
        if delta > best_delta:
            best_delta = delta
            best_order = perm
        
    for i, node in enumerate(vertexset):
        id_mapping[node] = best_order[i]
    return best_delta


def exchange(pair,in_edges,out_edges,reverse_mapping):
    #here the pair give a index pair such as [1,2],or [4,3] instead of the vertex ID pair
    current_order = pair
    best_order = current_order[:]
    best_delta = 0    
    exchange[0],exchange[1]=current_order[1],current_order[0]
    
    delta = compute_change([reverse_mapping[pair[0]],reverse_mapping[pair[1]]], current_order, perm,in_edges,out_edges,id_mapping)
    if delta > best_delta:
            best_delta = delta
            best_order = perm
        
    for i, node in enumerate(vertexset):
        id_mapping[node] = best_order[i]
    return best_delta

output_frac=1000

def generate_random_permutation(n):
    permutation = list(range(n))
    random.shuffle(permutation)
    pos = [0] * n
    for idx, node in enumerate(permutation):
        pos[node] = idx
    return permutation, pos

 

# Write the original vertex ID and its relative order to a file
def write_relabelled_nodes_to_file(mapping, output_file):
    with open(output_file, 'w') as f:
        for node, order in mapping.items():
            f.write(f"{node},{order}\n")

# Main function to perform all steps

def read_ID_Mapping(file_path):
    df=pd.read_csv(file_path, header=None, names=['ID', 'Index'])
    data_table = df.values.tolist()
    data_dict = {key: value for key, value in data_table}
    return data_dict


def process_graph(file_path,theid_mapping):

    starttime = time.time()
    node_list, edges, in_edges, out_edges = create_adjacency_lists(file_path)
    endtime = time.time()
    executiontime=endtime-starttime
    current_time = datetime.now()
    time_string = current_time.strftime("%Y%m%d_%H%M%S")
    # Create the file name using the formatted time string
    oneoutputfile = f"{FileNameHead}-{time_string}.txt"

    with open(oneoutputfile, 'a') as f:
            f.write(f"reading file {file_path} and build data structure take {executiontime} seconds \n\n")


    starttime = time.time()
    lvs,mvs,rvs = get_lvs_mvs_rvs(node_list, in_edges,out_edges)
    average_weight = average_node_weight(node_list,edges)
    Hvs,Lvs = split_set_by_weight(mvs, average_weight,in_edges,out_edges)
    if len(theid_mapping) >0 :
        id_mapping=theid_mapping
    else:
        id_mapping = {node: idx for idx, node in enumerate(lvs + Hvs + Lvs + rvs)}


    num_iters=50000
    output_frac=1000
    score= sum_of_weight_current_mapping (edges,id_mapping)

    for iter in range(0,num_iters):
            if len(Hvs)>4 :
                vertexset = random.sample(Hvs, 4)
            else:
                if len(Hvs + Lvs) >4 :
                     vertexset = random.sample(Hvs+Lvs, 4)
                else:
                     vertexset =Hvs+Lvs
            score += randommove(vertexset,in_edges,out_edges,id_mapping) 
            for i in range(len(lvs),len(lvs+mvs)-2):
                pair=set()
                pair.add(reverse_mapping[i])
                pair.add(reverse_mapping[i+1])
                score += exchange(pair,in_edges,out_edges,reverse_mapping)
            if (iter%(output_frac))==0:
                print("Iteration",iter,score)

                output_file=f"{oneoutputfile}-ID-{iter}.csv"
                write_relabelled_nodes_to_file(id_mapping, output_file)




    total_weight = score
    executiontime=endtime-starttime
    with open(oneoutputfile, 'a') as f:
        f.write(f"Total Edge Weight:{total_weight}, Total number of Edges= {len(edges)}\n")
        f.write(f"Gained weight={score},Percentage of remined Edges Weight: {score/total_weight * 100}\n\n")

    starttime = time.time()
    current_time = datetime.now()
    time_string = current_time.strftime("%Y%m%d_%H%M%S")
    #Create the file name using the formatted time string
    output_file = f"{FileNameHead}-10-Relabel-{time_string}.csv"
    write_relabelled_nodes_to_file(id_mapping, output_file)
    endtime = time.time()
    executiontime=endtime-starttime
    with open(oneoutputfile, 'a') as f:
            f.write(f"write label of file {file_path} uses {executiontime} seconds \n\n")



file_path = sys.argv[1]
if len(sys.argv) >2: 
    myid_mapping = read_ID_Mapping(sys.argv[2])
else:
    myid_mapping={}
process_graph(file_path,myid_mapping)


