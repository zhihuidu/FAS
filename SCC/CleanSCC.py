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


FileNameHead="Clean-SCC-DFS"

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

    in_adj={}
    out_adj={}
    for node in node_list:
        out_adj[node]=[]
        in_adj[node]=[]
    for source, target in merged_edges:
        out_adj[source].append((target, merged_edges[(source,target)]))
        in_adj[target].append((source, merged_edges[(source,target)]))

    return node_list, merged_edges, in_adj, out_adj 

def read_removed_edges(csv_file, edge_weights,edge_flag):
    removed_weight=0
    with open(csv_file, mode='r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            source, target = row
            source=int(source)
            target=int(target)
            if edge_flag[(source,target)] ==1 :
                removed_weight+=edge_weights[(source,target)]
                edge_flag[(source,target)]=0
    return removed_weight

def build_dag(edge_weights,edge_flag):
    G = nx.DiGraph()
    for (u,v) in edge_weights :
         if edge_flag[(u,v)]==1:
              G.add_edge(u,v,weight=edge_weights[(u,v)])
    return G

def build_small_graph(shG,smallcom,edge_flag):
    smallG = nx.DiGraph()
    for u, v, data in shG.edges(data=True):
         if  u in smallcom and v in smallcom and edge_flag[(u,v)]==1:
              smallG.add_edge(u,v,weight=data['weight'])
    return smallG
def build_new_shrunk_graph (G,edge_flag):
    shrunkG = nx.DiGraph()
    for u, v, data in G.edges(data=True):
         if edge_flag[(u,v)]==1:
              shrunkG.add_edge(u,v,weight=data['weight'])
    return shrunkG

def build_graph(edge_weights):
    G = nx.DiGraph()
    for (u,v) in edge_weights :
        G.add_edge(u,v,weight=edge_weights[(u,v)])
    return G

# Relabel the vertices in the DAG
def relabel_dag(G_dag):
    topological_order = list(nx.topological_sort(G_dag))
    mapping = {node: i for i, node in enumerate(topological_order)}
    return mapping

# Write the original vertex ID and its relative order to a file
def write_relabelled_nodes_to_file(mapping, output_file):
    with open(output_file, 'w') as f:
        for node, order in mapping.items():
            f.write(f"{node},{order}\n")

def read_ID_Mapping(file_path):
    df=pd.read_csv(file_path, header=None, names=['ID', 'Index'])
    data_table = df.values.tolist()
    data_dict = {key: value for key, value in data_table}
    return data_dict

num=0
def calculate_updated_weight(nodes,edge_weights,edge_flag,out_adj,updated_weights):
    print(f"search in the size 2,3,4 and 5 loops")
    global num
    for node in nodes:
        for neighbor, weight in out_adj[node]:
            if neighbor < node or edge_flag[(node,neighbor)]==0:
                  continue
            if (neighbor,node) in edge_flag:
                 if edge_flag[(neighbor,node)]==1:
                        if weight < edge_weights[(neighbor,node)]:
                              smallone=weight
                        else:
                              smallone=edge_weights[(neighbor,node)]
                        updated_weights[(node,neighbor)]-=smallone
                        updated_weights[(neighbor,node)]-=smallone
            for third, weight1 in out_adj[neighbor]:
                if third < node or edge_flag[(neighbor,third)]==0:
                        continue
                if (third,node) in edge_flag:
                      if edge_flag[(third,node)]==1:
                          weight2=edge_weights[(third,node)]
                          smallone==min(weight,weight1,weight2)
                          updated_weights[(node,neighbor)]-=smallone
                          updated_weights[(neighbor,third)]-=smallone
                          updated_weights[(third,node)]-=smallone
                for four,weight2 in out_adj[third]:
                         if four < node  or edge_flag[(third,four)]==0:
                                continue
                         if (four,node) in edge_flag:
                             if edge_flag[(four,node)]==1:
                                 weight3=edge_weights[(four,node)]
                                 smallone=min(weight,weight1,weight2,weight3)
                                 updated_weights[(node,neighbor)]-=smallone
                                 updated_weights[(neighbor,third)]-=smallone
                                 updated_weights[(third,four)]-=smallone
                                 updated_weights[(four,node)]-=smallone
                         for five, weight3 in out_adj[four]:
                               if five < node  or edge_flag[(four,five)]==0:
                                      continue
                               if (five,node) in edge_flag:
                                   if edge_flag[(five,node)]==1:
                                       weight4=edge_weights[(five,node)]
                                       smallone=min(weight,weight1,weight2,weight3,weight4)
                                       updated_weights[(node,neighbor)]-=smallone
                                       updated_weights[(neighbor,third)]-=smallone
                                       updated_weights[(third,four)]-=smallone
                                       updated_weights[(four,five)]-=smallone
                                       updated_weights[(five,node)]-=smallone


    print(f"after update weight of size 2,3,4 and 5 loops")


def remove_small_cycles(nodes,edge_weights,out_adj,edge_flag):
    updated_weights=edge_weights.copy()
    print(f"first calculate the updated weights in  the size 2,3,4 and 5 loops")
    calculate_updated_weight(nodes,edge_weights,edge_flag,out_adj,updated_weights)
    print(f"finish calculate the updated weights in  the size 2,3,4 and 5 loops")
    print(f"clean the size 2,3,4 and 5 loops")
    removed_weight=0
    global num
    for node in nodes:
        for neighbor, weight in out_adj[node]:
            u_weight=updated_weights[(node,neighbor)]
            if neighbor < node or edge_flag[(node,neighbor)]==0:
                  continue
            if (neighbor,node) in edge_flag:
                 if edge_flag[(neighbor,node)]==1:
                        if updated_weights[(node,neighbor)] < updated_weights[(neighbor,node)]:
                              edge_flag[(node,neighbor)]=0
                              removed_weight+=weight
                        else:
                              edge_flag[(neighbor,node)]=0
                              removed_weight+=edge_weights[(neighbor,node)]
                              print(f"removed {num} edges ({neighbor},{node})")
                        num+=1
            for third, weight1 in out_adj[neighbor]:
                u_weight1=updated_weights[(neighbor,third)]
                if third < node or edge_flag[(neighbor,third)]==0:
                        continue
                if (third,node) in edge_flag:
                      if edge_flag[(third,node)]==1:
                          weight2=edge_weights[(third,node)]
                          u_weight2=updated_weights[(third,node)]
                          if u_weight==min(u_weight,u_weight1,u_weight2):
                               edge_flag[(node,neighbor)]=0
                               removed_weight+=weight
                               print(f"removed {num} edges ({node},{neighbor})")
                               num+=1
                          else:
                              if u_weight1==min(u_weight,u_weight1,u_weight2):
                                   edge_flag[(neighbor,third)]=0
                                   removed_weight+=weight1
                                   print(f"removed {num} edges ({neighbor},{third})")
                                   num+=1
                              else:
                                   edge_flag[(third,node)]=0
                                   print(f"removed {num} edges ({third},{node})")
                                   removed_weight+=edge_weights[(third,node)]
                                   num+=1
                for four,weight2 in out_adj[third]:
                         u_weight2=updated_weights[(third,four)]   
                         if four < node  or edge_flag[(third,four)]==0:
                                continue
                         if (four,node) in edge_flag:
                             if edge_flag[(four,node)]==1:
                                 weight3=edge_weights[(four,node)]
                                 u_weight3=updated_weights[(four,node)]
                                 if u_weight==min(u_weight,u_weight1,u_weight2,u_weight3):
                                      edge_flag[(node,neighbor)]=0
                                      removed_weight+=weight
                                      print(f"removed {num} edges ({node},{neighbor})")
                                      num+=1
                                 else:
                                      if u_weight1==min(u_weight,u_weight1,u_weight2,u_weight3):
                                           edge_flag[(neighbor,third)]=0
                                           removed_weight+=weight1
                                           print(f"removed {num} edges ({node},{neighbor})")
                                           num+=1
                                      else:
                                          if u_weight2==min(u_weight,u_weight1,u_weight2,u_weight3):
                                                edge_flag[(third,four)]=0
                                                removed_weight+=weight2
                                                print(f"removed {num} edges ({third},{four})")
                                                num+=1
                                          else:
                                                edge_flag[(four,node)]=0
                                                removed_weight+=edge_weights[(four,node)]
                                                print(f"removed {num} edges ({four},{node})")
                                                num+=1


                         for five, weight3 in out_adj[four]:
                               u_weight3=updated_weight[(four,five)]
                               if five < node  or edge_flag[(four,five)]==0:
                                      continue
                               if (five,node) in edge_flag:
                                   if edge_flag[(five,node)]==1:
                                       weight4=edge_weights[(five,node)]
                                       u_weight4=updated_weights[(five,node)]
                                       if u_weight==min(u_weight,u_weight1,u_weight2,u_weight3,u_weight4):
                                            edge_flag[(node,neighbor)]=0
                                            removed_weight+=weight
                                            print(f"removed {num} edges ({node},{neighbor})")
                                            num+=1
                                       else:
                                            if u_weight1==min(u_weight,u_weight1,u_weight2,u_weight3,u_weight4):
                                                 edge_flag[(neighbor,third)]=0
                                                 removed_weight+=weight1
                                                 print(f"removed {num} edges ({node},{neighbor})")
                                                 num+=1
                                            else:
                                                if u_weight2==min(u_weight,u_weight1,u_weight2,u_weight3,u_weight4):
                                                      edge_flag[(third,four)]=0
                                                      removed_weight+=weight2
                                                      print(f"removed {num} edges ({third},{four})")
                                                      num+=1
                                                else:
                                                      if u_weight3==min(u_weight,u_weight1,u_weight2,u_weight3,u_weight4):
                                                            edge_flag[(four,five)]=0
                                                            removed_weight+=weight3
                                                            print(f"removed {num} edges ({four},{node})")
                                                            num+=1
                                                      else:
                                                            edge_flag[(five,node)]=0
                                                            removed_weight+=edge_weights[(five,node)]
                                                            print(f"removed {num} edges ({five},{node})")
                                                            num+=1


    print(f"after clean the size 2,3,4 and 5 loops")
    return removed_weight


def sccdfs_remove_cycle_edges(nodes,edge_weights,out_adj,edge_flag):
    oldnum=num
    removed_weight=0
    def dfs(node, stack, rec_stack ):
        global num
        nonlocal removed_weight
        rec_stack.add(node)
        stack.append(node)
        for neighbor, weight in out_adj[node]:
            if neighbor not in nodes :
                continue

            if neighbor in rec_stack :
                if edge_flag[(node, neighbor)]==0:
                     continue
                cycle = stack[stack.index(neighbor):] + [neighbor]
                cycle_edges=[]
                skip=0
                for i in range(len(cycle) - 1) :
                     if edge_flag[(cycle[i], cycle[i + 1])]==1:
                          cycle_edges.append((cycle[i], cycle[i + 1]))      
                     else:
                         skip=1
                         break
                if skip==0:
                    cycle_weights=[]
                    print(f"find cycle len is {len(cycle)} with vertices {cycle}")
                    for u, v in cycle_edges:
                        cycle_weights.append((u,v,edge_weights[(u,v)]))
                    min_edge = min(cycle_weights, key=lambda x: x[2])
                    removed_weight+=min_edge[2]
                    #removed_edges.add(min_edge)
                    print(f"removed {num+1} edges= {min_edge}")
                    num=num+1
                    edge_flag[(min_edge[0],min_edge[1])]=0
            else:
                    dfs(neighbor, stack, rec_stack)
                    if neighbor in nodes:
                         nodes.remove(neighbor)
        restorenode=stack.pop()
        rec_stack.remove(node)
        if node in nodes:
            nodes.remove(node)
        return False
    
    removed_edges = set()
    rec_stack = set()
    removed_weight=0
    print(f"start from node {nodes[0]} and we have {len(nodes)} nodes")
    while len(nodes)>0: 
        node = nodes[0]
        #print(f"start from node {node} and we have {len(nodes)} nodes now")
        #print(f"enter dfs")
        dfs(node, [], rec_stack)
        if node in nodes:
            nodes.remove(node)


    return removed_weight 

def again_dfs_remove_cycle_edges(nodes,edge_weights,out_adj,edge_flag):
    removed_edges = set()
    def dfs(node, stack,  rec_stack):
        global num
        nonlocal removed_weight
        nonlocal tovisit
        rec_stack.add(node)
        stack.append(node)
        for neighbor, weight in out_adj[node]:
            if neighbor in rec_stack :
                if edge_flag[(node, neighbor)]==0:
                     continue
                cycle = stack[stack.index(neighbor):] + [neighbor]
                cycle_edges=[]
                skip=0
                for i in range(len(cycle) - 1) :
                     if edge_flag[(cycle[i], cycle[i + 1])]==1:
                          cycle_edges.append((cycle[i], cycle[i + 1]))      
                     else:
                         skip=1
                         break
                if skip==0:
                    cycle_weights=[]
                    print(f"find cycle len is {len(cycle)} with vertices {cycle}")
                    for u, v in cycle_edges:
                        cycle_weights.append((u,v,edge_weights[(u,v)]))
                    min_edge = min(cycle_weights, key=lambda x: x[2])
                    removed_weight+=min_edge[2]
                    #removed_edges.add(min_edge)
                    print(f"removed {num+1} edges {min_edge}")
                    num=num+1
                    edge_flag[(min_edge[0],min_edge[1])]=0
            elif neighbor not in tovisit:
                    dfs(neighbor, stack, rec_stack)
                    if neighbor in tovisit:
                         tovisit.remove(neighbor)
        rec_stack.remove(node)
        restorenode=stack.pop()
        if node in tovisit:
                         tovisit.remove(node)
        #visited.remove(restorenode)
        return False
    

    # Open the CSV file
    print(f"read removed edges file removed.txt")
    removed_weight=0
    with open('removed.txt', 'r') as file:
              reader = csv.reader(file)
              for row in reader:
                 x1,x2,x3  = row # Strip any leading/trailing whitespace
                 x11,x12=x1.split(maxsplit=1)
                 x12=int(x12)
                 x2=int(x2)
                 if edge_flag[(x12,x2)] ==1:
                     edge_flag[(x12,x2)] = 0
                     removed_weight+=edge_weights[(x12,x2)]

    print(f"after read removed edges, weight={removed_weight}, remained={41912141-removed_weight} dfs again")
    return removed_weight 
    iternum=0
    new_removed_weight=removed_weight
    while iternum ==0 :
        visited = set()
        rec_stack = set()
        tovisit =nodes.copy()
        random.shuffle(tovisit)
        while len(tovisit)>0:
            node = tovisit.pop(0)
            print(f"start from node {node} and we have {len(tovisit)} nodes now")
            if node not in visited:
                dfs(node, [], rec_stack)
            if node in tovisit:
                tovisit.remove(node)
        new_removed_weight = sum(w for u, v,w in removed_edges)
        removed_weight += new_removed_weight
        iternum=iternum+1
    print("finish  again bfs")
    return removed_weight 



def process_graph(file_path):
    global num
    print(f"read data")
    node_list, edge_weights, in_edges, out_edges = create_adjacency_lists(file_path)
    G=build_graph(edge_weights)
    total=sum(edge_weights[(u,v)] for (u,v) in edge_weights)
    print(f"total number of nodes={len(node_list)}, total number of edges={len(edge_weights)}")
    print(f"sum of weight={total}")

    removed_weight=0
    edge_flag={(u,v):1 for (u,v) in edge_weights }
    #removed_weight+=again_dfs_remove_cycle_edges(node_list, edge_weights,out_edges,edge_flag )
    removed_weight+=remove_small_cycles(node_list,edge_weights,out_edges,edge_flag)
    #removed_weight=read_removed_edges("removed_edge.csv",edge_weights,edge_flag)
    current_time = datetime.now()
    time_string = current_time.strftime("%Y%m%d_%H%M%S")
    output_file=FileNameHead+"removed_edge"+time_string+".csv"
    with open(output_file, 'w') as f:
        for u, v in edge_flag:
            if edge_flag[(u,v)]==0:
                 f.write(f"{u},{v}\n")

    removednum=0
    removed_weight=0
    for u,v in edge_flag:
        if edge_flag[(u,v)]==0:
           removednum+=1
           removed_weight+=edge_weights[(u,v)]
    print(f"to here removed {removednum} edges and the weight is {removed_weight}, percentage is {removed_weight/total*100}")

    shG=build_new_shrunk_graph (G,edge_flag)



    newsize=9000
    while not nx.is_directed_acyclic_graph(shG):
        print("the graph is not a DAG.")
        print(f"strongly connected components")
        scc=list(nx.strongly_connected_components(shG))
        print(f"number of scc is {len(scc)}")
        numcomponent=0
        for component in scc:
            if len(component)==1:
                 continue
            print(f"handle the {numcomponent}th component with size {len(component)}")
            subnum=0
            oldnum=num
            while len(component) >newsize:
                   print(f"handle the {subnum}th random part of {numcomponent}th component with size {len(component)}")
                   subnum += 1
                   smallcom=random.sample(component, newsize)
                   component.difference_update(smallcom)
                   #smallG=build_small_graph(shG,smallcom,edge_flag)
                   removed_weight1=sccdfs_remove_cycle_edges(list(smallcom), edge_weights,out_edges,edge_flag )
                   removed_weight+=removed_weight1
                   addnum=max(num-oldnum,1)
                   if addnum < 100:
                        newsize=newsize*2
                   newsize=min(len(component),newsize)

            removed_weight1 = sccdfs_remove_cycle_edges(list(component), edge_weights,out_edges,edge_flag )
            removed_weight+=removed_weight1
            print(f"removed weight is {removed_weight1}, totally removed {removed_weight}, percentage is {removed_weight/total*100}\n\n")
            numcomponent+=1

            removednum=0
            for u,v in edge_flag:
                if edge_flag[(u,v)]==0:
                    removednum+=1
            print(f"to here removed {removednum} edges")

        print(f"sum of removed weight={removed_weight},percentage of remained  weight ={(total-removed_weight)/total *100}")

        print(f"build dag")
        shG=build_dag(edge_weights,edge_flag)
        dag_weight=sum(data['weight'] for u, v, data in shG.edges(data=True))
        print(f"dag weight={dag_weight}")
        if dag_weight+removed_weight != total:
            print("something wrong with the removed edge weights, remained weights, and the total weights")



    print(f"relabel dag")
    mapping = relabel_dag(shG)

    print(f"write file")
    current_time = datetime.now()
    time_string = current_time.strftime("%Y%m%d_%H%M%S")
    output_file = f"{FileNameHead}-Relabel-{time_string}.csv"
    write_relabelled_nodes_to_file(mapping, output_file)




file_path = sys.argv[1]
sys.setrecursionlimit(900000)
process_graph(file_path)

