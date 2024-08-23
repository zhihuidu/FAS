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

#given a graph file in the csv format (each line is (source,destination, weight)), generate the graph data structure

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

    return node_list, merged_edges_list, in_adj, out_adj 

def get_lvs_mvs_rvs(nodes, in_edges, out_edges):
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



def sum_of_weight_for_current_mapping (edges,id_mapping):
    w = 0
    for source, target, weight in edges:
        if id_mapping [source] < id_mapping [target]:
            w += weight
    return w


def compute_change(nodes, in_adj, out_adj,id_mapping, new_id_mapping):
    #for the vertex ID list such as [1,50,10,3],we shuffle the IDs and calculate the best increased weights based on different mappings
    #new_id_mapping is used to calculate the gained weight for changed mapping

    delta = 0
    for node in nodes:
        for target, weight in out_adj[node]:
            if id_mapping[node] < id_mapping [target]:
                delta -= weight
            if new_id_mapping[node] < new_id_mapping[target]:
                delta += weight
        for source, weight in in_adj[node]:
            if source in nodes:
                continue
            if id_mapping [source] < id_mapping[node] :
                delta -= weight
            if new_id_mapping [source] < new_id_mapping[node] :
                delta += weight
    return delta

def listshuffle(vertexlist,in_edges,out_edges,id_mapping,new_id_mapping):
    # vertexlist is the ID list whose ID will be permuated to calculate the best weight sum 

    current_order = [id_mapping[idx] for idx in vertexlist]
    best_order = current_order.copy()
    best_delta = 0    
    
    perms = list(permutations(vertexlist))
    perms=perms[1:] #remove the original one without any changes
    for perm in perms:
        for index,node in enumerate (perm) :
              new_id_mapping[perm[index]]=id_mapping[vertexlist[index]]
        # update the mapping 

        delta = compute_change(vertexlist, in_edges,out_edges,id_mapping, new_id_mapping)
        for index,node in enumerate (perm) :
              new_id_mapping[vertexlist[index]]=id_mapping[vertexlist[index]]
        # restore the mapping

        if delta > best_delta and delta > 0.00000001:
            best_delta = delta
            for index,node in enumerate (perm):
                best_order[index] = id_mapping[perm[index]]
        
    if best_delta >0 :
        for i, node in enumerate(vertexlist):
            id_mapping[node] = best_order[i]
            new_id_mapping[node]=best_order[i]

    return best_delta


def exchange(pair,in_edges,out_edges,id_mapping,new_id_mapping):
    #here the pair give a index pair such as [1,2], instead of the vertex ID pair
    
    new_id_mapping[pair[0]]=id_mapping[pair[1]]
    new_id_mapping[pair[1]]=id_mapping[pair[0]]

    delta = compute_change(pair,in_edges,out_edges,id_mapping,new_id_mapping)

    if delta > 0:
        id_mapping[pair[0]]  = new_id_mapping[pair[0]]
        id_mapping[pair[1]]  = new_id_mapping[pair[1]]
    else:
        new_id_mapping[pair[0]]=id_mapping[pair[0]]
        new_id_mapping[pair[1]]=id_mapping[pair[1]]
        delta=0
    return delta



# Write the original vertex ID and its relative order to a file
def write_labelled_nodes_to_file(id_mapping, output_file):
    with open(output_file, 'w') as f:
        f.write(f"Node Id, Order\n")
        for node, order in id_mapping.items():  
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
    # here we divide the vertex set into three part, the labels in left part should smaller than the right part
    total=graph_total_weight(edges)
    print(f"total number of nodes={len(node_list)}, total number of edges={len(edges)}")
    print(f"sum of weight={total},average weight per node={total/len(node_list)} average weight per edge={total/len(edges)}")
    Hvs,Lvs = split_set_by_weight(mvs, 1.5 ,in_edges,out_edges)
    HCvs,LCvs = split_set_by_weight_change(mvs,1.5 ,in_edges,out_edges)
    # we use average weight to further divided the middle set into two parts. The high weight part should have 
    # more significant effect on the final score
    id_mapping={}
    print(f"size of lvs={len(lvs)}, mvs={len(mvs)}, Hvs={len(Hvs)}, Lvs={len(Lvs)}, HCvs={len(HCvs)}, LCvs={len(LCvs)}, rvs={len(rvs)}")
    label={}
    if len(theid_mapping) >0 :
        print("start from existing mapping")
        id_mapping=theid_mapping
    else:
        id_mapping = {node: idx for idx, node in enumerate(lvs + mvs + rvs)}

    num_iters=50000
    output_frac=100
    score= sum_of_weight_for_current_mapping (edges,id_mapping)
    oldscore=score
    oldwrite=score
    no_improvement=0
    no_increasement=0
    new_id_mapping=id_mapping.copy()

    vertexlist=[]
    for iter in range(0,num_iters):
            originalresult = sum_of_weight_for_current_mapping (edges,id_mapping)
            for j in range(700):
                if len(Hvs)>3 :
                    vertexlist = random.sample(Hvs, 3)
                else:
                    vertexlist = random.sample(mvs, 3)
                score += listshuffle(vertexlist,in_edges,out_edges,id_mapping,new_id_mapping)
          
            directresult = sum_of_weight_for_current_mapping (edges,id_mapping)
            Hvsgain=directresult-originalresult
            if iter %2 ==0:
                print(f"Iter={iter}, Score= {score/total*100}, Weight={score}, After 700 times random search in Hvs, the gained percentage is {Hvsgain/total*100}\n\n")
            originalresult=directresult 
            if directresult !=score :
                 if directresult > score :
                     print(f"Large    Hvs Search: Real  result {directresult} > {score},the accumulated value\n")
                 else :
                     print(f"Small    Hvs Search: Real  result {directresult} < {score}, the accumulated value\n")

                 score=directresult

            for j in range(200):
                if len(HCvs)>3 :
                    vertexlist = random.sample(HCvs, 3)
                else:
                    vertexlist = random.sample(mvs, 3)
                score += listshuffle(vertexlist,in_edges,out_edges,id_mapping,new_id_mapping)
          
            directresult = sum_of_weight_for_current_mapping (edges,id_mapping)
            HCvsgain=directresult-originalresult
            if iter %2 ==0:
                print(f"Iter={iter}, Score= {score/total*100}, Weight={score}, After 200 times random search in HCvs, the gained percentage is {HCvsgain/total*100}\n\n")
            originalresult=directresult 
            if directresult !=score :
                 if directresult > score :
                     print(f"Large    HCvs Search: Real  result {directresult} > {score}, the accumulated value\n")
                 else :
                     print(f"Small    HCvs Search: Real  result {directresult} < {score}, the accumulated value\n")
                 score=directresult
            for j in range(100):
                if len(mvs)>3 :
                    vertexlist = random.sample(mvs, 3)
                else:
                    vertexlist = random.sample(node_list, 3)
                score += listshuffle(vertexlist,in_edges,out_edges,id_mapping,new_id_mapping)
          
            directresult = sum_of_weight_for_current_mapping (edges,id_mapping)
            mvsgain=directresult-originalresult
            if iter %2 ==0:
                print(f"Iter={iter}, Score= {score/total*100}, Weight={score}, After 100 times random search in mvs,the gained percentage is {mvsgain/total*100}\n\n")
            originalresult=directresult 
            if directresult !=score :
                 if directresult > score :
                     print(f"Large    mvs Search: Real  result {directresult} > {score}, the accumulated value\n")
                 else :
                     print(f"Small    mvs Search: Real  result {directresult} < {score}, the accumulated value\n")
                 score=directresult




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
                print(f"{name}: Value = {value}, Ratio compared to the smallest = {ratio:.2f}")

            reverse_mapping={idx: node for node, idx in id_mapping.items()}
            if iter%10 ==0 :
                for i in range(len(lvs),len(lvs+mvs)-2):
                    pair=[]
                    pair.append(reverse_mapping[i])
                    pair.append(reverse_mapping[i+1])
                    score += exchange(pair,in_edges,out_edges,id_mapping,new_id_mapping)
                directresult = sum_of_weight_for_current_mapping (edges,id_mapping)
                neighbourgain=directresult-originalresult
                print(f"After {len(mvs)-1} times  neighbour exchange in [{len(lvs)},{len(lvs+mvs)-2}], the gained weight is {score} about {score/total*100}, the gained percentage is {neighbourgain/total*100}")
                originalresult=directresult 
                if directresult !=score :
                    if directresult > score :
                        print(f"Large    Neighbor Search: Real  result {directresult} > {score}, the accumulated value\n")
                    else :
                        print(f"Small    Neighbor Search: Real  result {directresult} < {score}, the accumulated value\n")
                    score=directresult
                smallest_value = min(HCvsgain, Hvsgain,mvsgain,neighbourgain)
                if smallest_value==0:
                    smallest_value=1
                # Create a list of tuples with variable names, values, and their ratios
                variables = [
                   ('HCvsgain', HCvsgain, HCvsgain / smallest_value),
                   ('Hvsgain', Hvsgain, Hvsgain / smallest_value),
                   ('mvsgain', mvsgain, mvsgain / smallest_value),
                   ('neighbourgain', neighbourgain, neighbourgain / smallest_value)
                ]

                # Sort the list by the second element in each tuple (the value)
                variables_sorted = sorted(variables, key=lambda x: x[1])
                for name, value, ratio in variables_sorted:
                     print(f"{name}: Value = {value}, Ratio compared to the smallest = {ratio:.2f}\n")


            if (iter%(output_frac))==0 or (score-oldwrite)/total >0.05 or ((score-oldwrite)/total >0.01 and iter >1000):
                print(f"Iteration,{iter},{score},{score/total*100}")
                output_file=f"{oneoutputfile}-ID-{iter}.csv"
                #write_ordered_nodes_to_csv(permutation, pos, output_file, index_to_node)

                print(f"write the current {score/total*100}, increased  {(score-oldwrite)/total*100 } compared with last write\n\n")                    
                write_labelled_nodes_to_file(id_mapping, output_file)
                oldwrite=score
            if score > oldscore:
                 #print(f"Iteration,{iter},{score},{score/total*100}")
                 oldscore = score
                 no_increasement=0
            else:
                 no_increasement+=1
            if no_increasement > 100000 :
               break




    executiontime=endtime-starttime
    with open(oneoutputfile, 'a') as f:
        f.write(f"Total Edge Weight:{total}, Total number of Edges= {len(edges)}\n")
        f.write(f"Gained weight={score},Percentage of remined Edges Weight: {score/total * 100}\n\n")

    starttime = time.time()
    current_time = datetime.now()
    time_string = current_time.strftime("%Y%m%d_%H%M%S")
    #Create the file name using the formatted time string
    output_file = f"{FileNameHead}-10-Relabel-{time_string}.csv"
    write_labelled_nodes_to_file(id_mapping, output_file)
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


