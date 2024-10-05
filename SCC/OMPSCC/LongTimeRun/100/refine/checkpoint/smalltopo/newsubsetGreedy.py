#!/usr/bin/env pypy

import csv
import random
from itertools import permutations
from collections import defaultdict
from datetime import datetime
import pandas as pd
import networkx as nx
from networkx.algorithms.cycles import simple_cycles
import gurobipy as gp
from gurobipy import GRB
import sys
import time


FileNameHead="SetFunc"


def get_lvs_mvs_rvs(originalnodes, in_edges, out_edges):
    nodes=[]
    for node in originalnodes:
       nodes.append(node_to_index[node])

    lvs = [node for node in nodes if len(in_edges[node]) == 0 and len(out_edges[node])>=0]
    #vertices in the left vertex list (lvs) should have small index without refining

    rvs = [node for node in nodes if len(out_edges[node]) == 0 and len(in_edges[node]) > 0]
    #vertices in the right vertex list (rvs) should have large index without refining

    mvs = [node for node in nodes if len(in_edges[node])>0 and len(out_edges[node])>0]
    #vertices in the middle vertex list (mvs) should have middle index and we need to refine them

    return lvs, mvs, rvs


def graph_total_weight(edges):
    return sum(w for u, v, w in edges)


def average_node_weight(nodes, edges):
    return graph_total_weight(edges)/len(nodes)
    #the average weight of one node


def vertex_weight(node, in_edges,out_edges):
    return sum(w for (v,w) in in_edges[node]) + sum(w for (v,w) in out_edges[node])
    #all weights associated with one vertex

def vertex_in_weight(node, in_edges):
    return sum(w for (v,w) in in_edges[node]) 
    #all in weights associated with one vertex

def vertex_out_weight(node, out_edges):
    return  sum(w for (v,w) in out_edges[node])
    #all out weights associated with one vertex

def split_set_by_weight(originalset, split_weight,in_edges,out_edges):
    #split a given set into two subsets, one with relative higher weight for each vertex
    #another set has lower weight for each vertex. split_weight is a relative number compared with the average weight
    Hvs=[]
    Lvs=[]
    sumweight=0.0
    averageweight=0.0
    for node in originalset:
        sumweight+= vertex_weight(node,in_edges,out_edges)
    if len(originalset) >1 :
        averageweight=sumweight/len(originalset)
    else:
        averageweight=sumweight

    for node in originalset:
        if vertex_weight(node,in_edges,out_edges)/averageweight>split_weight:
            Hvs.append(node)
        else:
            Lvs.append(node)
    return Hvs, Lvs

def split_set_by_weight_change(originalset, split_weight,in_edges,out_edges):
    #split a given set into two subsets, one with relative higher weight for each vertex
    #another set has lower weight for each vertex. split_weight is a relative number compared with the average weight
    HCvs=[]
    LCvs=[]
    sumin=0
    sumout=0
    for node in originalset:
        sumin  +=vertex_in_weight(node,in_edges)
        sumout +=vertex_out_weight(node,out_edges)
    if len(originalset) >1 :
        averagechange=abs(sumout-sumin)/len(originalset)
    else:
        averagechange=abs(sumout-sumin)


    for node in originalset:
        if abs(vertex_out_weight(node,out_edges)-vertex_in_weight(node,in_edges))/averagechange >split_weight:
            HCvs.append(node)
        else:
            LCvs.append(node)
    return HCvs, LCvs


def init_segment_based_mapping (nodes, edges,in_edges,out_edges):
    lvs,mvs,rvs = get_lvs_mvs_rvs(nodes, in_edges,out_edges)
    average_weight = average_node_weight(nodes,edges)
    id_mapping = {node: idx for idx, node in enumerate(lvs + mvs + rvs)}
    #mapping from vertex ID/string to index

    reverse_mapping = {idx: node for node, idx in id_mapping.items()}
    #mapping from index to vertex ID

    return id_mapping,reverse_mapping



def read_ID_Mapping(file_path):
    df=pd.read_csv(file_path, header=None, names=['ID', 'Index'])
    data_table = df.values.tolist()
    data_dict = {key: value for key, value in data_table}
    return data_dict


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

def sum_of_weight_for_current_mapping (edges,x):
    w = 0
    for source, target, weight in edges:
        if state [node_to_index[source]] < state [node_to_index[target]]:
            w += weight
    return w


def generate_random_permutation(n):
    permutation = list(range(n))
    random.shuffle(permutation)
    pos = [0] * n
    for idx, node in enumerate(permutation):
        pos[node] = idx
    return permutation, pos

def new_generate_random_permutation(n):
    lvs,mvs,rvs = get_lvs_mvs_rvs(node_list, in_adj,out_adj)
    # here we divide the vertex set into three part, the labels in left part should smaller than the right part
    total=graph_total_weight(edges)
    print(f"total number of nodes={len(node_list)}, total number of edges={len(edges)}")
    print(f"sum of weight={total},average weight per node={total/len(node_list)} average weight per edge={total/len(edges)}")
    Hvs,Lvs = split_set_by_weight(mvs, 1.5 ,in_adj,out_adj)
    HCvs,LCvs = split_set_by_weight_change(mvs,400.0 ,in_adj,out_adj)
    # more significant effect on the final score
    id_mapping={}
    print(f"size of lvs={len(lvs)}, mvs={len(mvs)}, Hvs={len(Hvs)}, Lvs={len(Lvs)}, HCvs={len(HCvs)}, LCvs={len(LCvs)}, rvs={len(rvs)}")
    id_mapping = {node: idx for idx, node in enumerate(lvs + mvs + rvs)}
    permutation=node_list.copy()
    return permutation, id_mapping,lvs, mvs,rvs,Hvs,Lvs,HCvs,LCvs



def write_ordered_nodes_to_csv(permutation, pos, output_csv_file, index_to_node):
    with open(output_csv_file, mode='w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['node_id', 'order in the permutation'])
        for idx in range(len(permutation)):
            node = index_to_node[idx]
            csvwriter.writerow([node, pos[idx]])
 


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
    actural={}

    for i in range(len(nodes)):
        node=nodes[i]
        curPos=cur[i]
        actural[node]=curPos
    for i in range(len(nodes)):
        node = nodes[i]
        prevPos = prev[i]
        curPos = cur[i]
        for target, weight in out_adj[node]:
            nextPos=state[target]
            if target in nodes:
                nextPos=actural[target]
            if prevPos < state[target]:
                delta -= weight
            if curPos < nextPos:
                delta += weight
        for source, weight in in_adj[node]:
            if source in nodes:
                continue
            if state[source] < prevPos:
                delta -= weight
            if state[source] < curPos:
                delta += weight
    return delta

def move(indices,number):
    # Select four random indices
    #indices = random.sample(nodes, number)
    current_order = [state[idx] for idx in indices]
    best_order = current_order[:]
    best_delta = 0    
    
    perms=list(permutations(current_order))
    perms=perms[1:]
    for perm in perms:
        delta = compute_change(indices, current_order, perm)
        if delta > best_delta:
            best_delta = delta
            best_order = perm
        
    for i, node in enumerate(indices):
        state[node] = best_order[i]
    return best_delta



def process_graph(file_path):

    current_time = datetime.now()
    time_string = current_time.strftime("%Y%m%d_%H%M%S")
    # Create the file name using the formatted time string
    oneoutputfile = f"{FileNameHead}-{time_string}.txt"

    with open(oneoutputfile, 'a') as f:
            f.write(f"reading file {file_path} \n\n")

    id_mapping={}
    score= sum_of_weight_for_current_mapping (edges,id_mapping)
    oldscore=score
    oldwrite=score
    no_improvement=0
    no_increasement=0

    total=graph_total_weight(edges)
    vertexlist=[]
    for iter in range(0,num_iters):
            originalresult=score
            for j in range(1000):
                if len(Hvs)>3 :
                    vertexlist = random.sample(Hvs, 3)
                else:
                    vertexlist = [0]
                score += move(vertexlist,len(vertexlist))
            directresult=score
            Hvsgain=directresult-originalresult


            originalresult=directresult 
            for j in range(1000):
                if len(HCvs)>3 :
                    vertexlist = random.sample(HCvs, 3)
                else:
                    vertexlist = random.sample(mvs, 3)
                score += move (vertexlist,len(vertexlist))
          
            directresult = score
            HCvsgain=directresult-originalresult
            originalresult=directresult 
            score=directresult
            for j in range(1000):
                if len(mvs)>3 :
                    vertexlist = random.sample(mvs, 3)
                else:
                    vertexlist = [0]
                score += move(vertexlist,len(vertexlist))
          
            directresult = score
            mvsgain=directresult-originalresult
            originalresult=directresult 

            smallest_value = min(HCvsgain, Hvsgain,mvsgain)
            if smallest_value ==0:
                    smallest_value=1
            # Create a list of tuples with variable names, values, and their ratios
            # Create a list of tuples with variable names, values, and their ratios
            variables = [
              ('HCvsgain', HCvsgain, HCvsgain / smallest_value),
              ('Hvsgain', Hvsgain, Hvsgain / smallest_value),
              ('mvsgain', mvsgain, mvsgain / smallest_value)
            ]

            # Sort the list by the second element in each tuple (the value)
            variables_sorted = sorted(variables, key=lambda x: x[1])
            for name, value, ratio in variables_sorted:
                print(f"{name} = {value}, Ratio compared to the smallest = {ratio:.2f}")
            print(f"Iter={iter}, Percentage= {score/total*100}, Weight={score}\n\n")

            reverse_mapping={idx: node for node, idx in state.items()}
            if iter%10 ==0 or (score/total>0.8 and iter%2 ==0) :
                for i in range(len(lvs),len(lvs+mvs)-2):
                    pair=[]
                    pair.append(reverse_mapping[i])
                    pair.append(reverse_mapping[i+1])
                    score += move(pair,2)
                directresult=score
                neighbourgain=directresult-originalresult
                print(f"After {len(mvs)-1} times  neighbour exchange in [{len(lvs)},{len(lvs+mvs)-2}], the gained weight is {score} about {score/total*100}, the gained percentage is {neighbourgain/total*100}")

                originalresult=directresult 
                for j in range(1000):
                    if len(LCvs)>3 :
                        vertexlist = random.sample(LCvs, 3)
                    else:
                        vertexlist = [0]
                    score += move (vertexlist,len(vertexlist))
          
                directresult = score
                LCvsgain=directresult-originalresult


                originalresult=directresult 
                for j in range(1000):
                    if len(Lvs)>3 :
                        vertexlist = random.sample(Lvs, 3)
                    else:
                        vertexlist = [0]
                    score += move (vertexlist,len(vertexlist))
          
                directresult = score
                Lvsgain=directresult-originalresult

                originalresult=directresult 
                smallest_value = min(HCvsgain, Hvsgain,mvsgain,neighbourgain,Lvsgain,LCvsgain)
                if smallest_value==0:
                    smallest_value=1
                # Create a list of tuples with variable names, values, and their ratios
                variables = [
                   ('HCvsgain', HCvsgain, HCvsgain / smallest_value),
                   ('Hvsgain', Hvsgain, Hvsgain / smallest_value),
                   ('LCvsgain', LCvsgain, LCvsgain / smallest_value),
                   ('Lvsgain', Lvsgain, Lvsgain / smallest_value),
                   ('mvsgain', mvsgain, mvsgain / smallest_value),
                   ('neighbourgain', neighbourgain, neighbourgain / smallest_value)
                ]

                # Sort the list by the second element in each tuple (the value)
                variables_sorted = sorted(variables, key=lambda x: x[1])
                for name, value, ratio in variables_sorted:
                     print(f"{name} = {value}, Ratio compared to the smallest = {ratio:.2f}")

                print("\n\n")

            if (iter%(output_frac))==0 or (score-oldwrite)/total >0.05 or ((score-oldwrite)/total >0.01 and iter >100):
                print(f"Iteration,{iter},{score},{score/total*100}")
                output_file=f"{oneoutputfile}-ID-{iter}.csv"
                #write_ordered_nodes_to_csv(permutation, pos, output_file, index_to_node)

                print(f"write the current {score/total*100}, increased  {(score-oldwrite)/total*100 } compared with last write\n\n")                    
                write_ordered_nodes_to_csv(permutation, pos, output_file, index_to_node)
                oldwrite=score
            if score > oldscore:
                 #print(f"Iteration,{iter},{score},{score/total*100}")
                 oldscore = score
                 no_increasement=0
            else:
                 no_increasement+=1
            if no_increasement > 50 :
               break




    executiontime=endtime-starttime
    with open(oneoutputfile, 'a') as f:
        f.write(f"Total Edge Weight:{total}, Total number of Edges= {len(edges)}\n")
        f.write(f"Gained weight={score},Percentage of remined Edges Weight: {score/total * 100}\n\n")


    write_ordered_nodes_to_csv(permutation, pos, output_csv_file, index_to_node)




csv_file_path = sys.argv[1]
current_time = datetime.now()
time_string = current_time.strftime("%Y%m%d_%H%M%S")
output_csv_file = f"IDMapping-{time_string}.csv"
out_adj, in_adj, node_list, edges, node_to_index, index_to_node = create_adjacency_lists(csv_file_path)
n = len(node_list)
permutation, pos,lvs,mvs,rvs,Hvs,Lvs,HCvs,LCvs = new_generate_random_permutation(n)
state=pos
num_iters = 5000000
output_frac = 100
score = initial_score()

process_graph(csv_file_path)


'''
try:
    for iter in range(0,num_iters,1000):
        if (iter %1000 <300):
              score+=move(Hvs,3)
        else:
           if (iter %1000 >600):
              score+=move(mvs,3)
           else:
              score+=move(HCvs,3)
           
        if (iter%(output_frac))==0:
            print("Iteration",iter,score,score/41912141.0*100.0)
            directresult= sum_of_weight_for_current_mapping (edges)
            if directresult !=score :
                 if directresult > score :
                     print(f"Large    Real  result {directresult} > {score}, the accumulated value\n")
                 else :
                     print(f"Small    Real  result {directresult} < {score}, the accumulated value\n")
            score=directresult


finally:
    write_ordered_nodes_to_csv(permutation, pos, output_csv_file, index_to_node)

    print(f"Ordered nodes saved to {output_csv_file}")
    print(f"Final score: {score}\n")


'''



#file_path = sys.argv[1]
#if len(sys.argv) >2: 
#    myid_mapping = read_ID_Mapping(sys.argv[2])
#else:
#    myid_mapping={}
#process_graph(file_path)

