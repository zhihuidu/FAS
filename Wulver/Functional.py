import pandas as pd
import networkx as nx
from networkx.algorithms.cycles import simple_cycles
import gurobipy as gp
from gurobipy import GRB
import sys
from datetime import datetime
import time

FileNameHead="Functional"
# Read the CSV file and create a directed graph
def read_graph_from_csv(file_path):
    #df = pd.read_csv(file_path)
    df=pd.read_csv(file_path, header=None, names=['source', 'destination', 'weight'])
    G = nx.DiGraph()
    for index, row in df.iterrows():
        G.add_edge(row['source'], row['destination'], weight=int(row['weight']))
        #print("source=",row['source'], "dest=",row['destination'], "weight=",row['weight'])
    return G

def read_ID_Mapping(file_path):
    df=pd.read_csv(file_path, header=None, names=['ID', 'Index'])
    data_table = df.values.tolist()
    data_dict = {key: value for key, value in data_table}
    return data_dict

# Find all directed cycles in the graph
def find_cycles(G):
    return list(simple_cycles(G))
    #  next_n_cycles = list(itertools.islice(all_cycles, start, start + number))

def build_cycle_graph(original_graph):
    """
    Build a new directed graph based on cycles from the original directed graph.
    
    Parameters:
    original_graph (networkx.DiGraph): The original directed graph.
    
    Returns:
    networkx.DiGraph: A new directed graph containing edges from cycles.
    """
    cycle_graph = nx.DiGraph()
    
    # Get all cycles in the original graph
    cycles = nx.simple_cycles(original_graph)

    # Add edges from each cycle to the new graph
    for cycle in cycles:
        # Add edges of the cycle to the new graph
        for i in range(len(cycle)):
            source = cycle[i]
            destination = cycle[(i + 1) % len(cycle)]  # wrap around to the start
            # Check if the edge exists in the original graph
            if original_graph.has_edge(source, destination):
                weight = original_graph[source][destination]['weight']
                cycle_graph.add_edge(source, destination, weight=weight)

    return cycle_graph



# Calculate the total weight of the edges
def total_edge_weight(G):
    return sum(data['weight'] for u, v, data in G.edges(data=True))

# Calculate the weight of the removed edges
def removed_edge_weight(G, removed_edges):
    return sum(G[u][v]['weight'] for u, v in removed_edges)

# Identify the removed edges and construct a DAG
def construct_dag(G, removed_edges):
    G_dag = G.copy()
    G_dag.remove_edges_from(removed_edges)
    return G_dag

# Relabel the vertices in the DAG
def relabel_dag(G_dag):
    topological_order = list(nx.topological_sort(G_dag))
    mapping = {node: i for i, node in enumerate(topological_order)}
    G_dag = nx.relabel_nodes(G_dag, mapping)
    return G_dag, mapping

# Write the original vertex ID and its relative order to a file
def write_relabelled_nodes_to_file(mapping, output_file):
    with open(output_file, 'w') as f:
        for node, order in mapping.items():
            f.write(f"{node},{order}\n")





def calculate_vertex_gain(graph, id_mapping, node):
    gain = 0
    for neighbor, data in graph[node].items():
        if id_mapping[neighbor] > id_mapping[node]:
            gain += data['weight']
    for pred in graph.predecessors(node):
        if id_mapping[pred] < id_mapping[node]:
            gain += graph[pred][node]['weight']
    return gain

def calculate_set_weight(graph, id_mapping, r):
    sumweight = 0
    for u, v, data in graph.edges(data=True):
        if (   (id_mapping[u] < id_mapping[v]) and   
               ((id_mapping[u]>=r[0] and id_mapping[u]<=r[1]) or 
                (id_mapping[v]>=r[0] and id_mapping[v]<=r[1]))):
                sumweight += graph[u][v]['weight']
    '''
    for u in vertex_set:
        predecessors = list(graph.predecessors(u))
        for v in predecessors:
            if ((id_mapping[v] < id_mapping[u]) and (v not in vertex_set or v<u)):
                sumweight += graph[v][u]['weight']
        successors = list(graph[u])
        for v in successors:
            if ((id_mapping[u] < id_mapping[v]) and (v not in vertex_set or u<v)):
                sumweight += graph[u][v]['weight']
'''

    return sumweight

def calculate_gained_weight(graph, id_mapping):
    total_score = 0
    for u, v, data in graph.edges(data=True):
        if id_mapping[u] < id_mapping[v]:
            total_score += data['weight']
    return total_score


def optimize_ids(graph, lvs, mvs, rvs,oneoutputfile,theid_mapping):
    start_time=time.time()
    id_mapping = theid_mapping.copy()
    if len(id_mapping)==0: 
        id_mapping = {node: idx for idx, node in enumerate(lvs + mvs + rvs)}

    reverse_mapping = {idx: node for node, idx in id_mapping.items()}
    #best_id_mapping = id_mapping.copy()
    percentage=0.0
    
    no_improvement = 1
    writenum=0
    total_weight=total_edge_weight(graph)
    #previous_score = calculate_set_weight(graph, id_mapping, [len(lvs),len(lvs)+len(mvs)-1])
    #previous_weight = calculate_set_weight(graph, id_mapping, [len(lvs),len(lvs)+len(mvs)-1])
    previous_weight = calculate_gained_weight(graph, id_mapping)
    percentage=previous_weight/total_weight * 100.0
    
    while no_improvement < 20 :
        with open(oneoutputfile, 'a') as f:
            f.write(f"gained weight ={previous_weight} total weight ={total_weight} and percentage ={percentage} \n")
            f.write(f"no improved iterations ={no_improvement} \n")
            f.write(f"current time is {datetime.now()} \n\n")

        for x in mvs:
            xgain = calculate_vertex_gain(graph, id_mapping, x)
            rgain = 0
            lgain = 0
            if id_mapping[x] < len(lvs) + len(mvs) - 1:
                if id_mapping[x] < len(lvs) + len(mvs) -no_improvement :
                    r = reverse_mapping[id_mapping[x] + no_improvement ]
                else :
                    r = reverse_mapping[id_mapping[x] + 1 ]
                id_mapping[x], id_mapping[r] = id_mapping[r], id_mapping[x]
                xrgain = calculate_vertex_gain(graph, id_mapping, x)+calculate_vertex_gain(graph, id_mapping, r)
                id_mapping[x], id_mapping[r] = id_mapping[r], id_mapping[x]  # Swap back
                rgain = calculate_vertex_gain(graph, id_mapping, r)
            else:
                xrgain = 0
                
            if id_mapping[x] > len(lvs):
                if id_mapping[x] > len(lvs)+no_improvement:
                    l = reverse_mapping[id_mapping[x] - no_improvement]
                else:
                    l = reverse_mapping[id_mapping[x] - 1]
                id_mapping[x], id_mapping[l] = id_mapping[l], id_mapping[x]
                xlgain = calculate_vertex_gain(graph, id_mapping, x)+calculate_vertex_gain(graph, id_mapping, l)
                id_mapping[x], id_mapping[l] = id_mapping[l], id_mapping[x]  # Swap back
                lgain =calculate_vertex_gain(graph, id_mapping, l)
            else:
                xlgain = 0
                
            if xrgain > xgain + rgain and (xrgain > xlgain or xlgain < xgain+lgain) :
                    id_mapping[x], id_mapping[r] = id_mapping[r], id_mapping[x]
            elif xlgain > xgain + lgain :
                    id_mapping[x], id_mapping[l] = id_mapping[l], id_mapping[x]
        
        ##new_weight = calculate_set_weight(graph, id_mapping, [len(lvs),len(lvs)+len(mvs)-1])
        new_weight = calculate_gained_weight(graph, id_mapping)
        if new_weight <= previous_weight:
            no_improvement += 1
        else:
            no_improvement = 0
            previous_weight = new_weight
            percentage=new_weight/total_weight
        current_time=time.time()
        if (current_time-start_time) >1800:
            writenum+=1
            output_file=f"{oneoutputfile}-ID-{writenum}.csv"
            write_relabelled_nodes_to_file(id_mapping, output_file)
            start_time=current_time

    return id_mapping





def get_lvs_mvs_rvs(graph):
    lvs = [node for node in graph.nodes if graph.in_degree(node) == 0 and graph.out_degree(node) > 0]
    rvs = [node for node in graph.nodes if graph.out_degree(node) == 0 and graph.in_degree(node) > 0]
    mvs = [node for node in graph.nodes if node not in lvs and node not in rvs]
    return lvs, mvs, rvs




# Main function to perform all steps
def process_graph(file_path,theid_mapping):

    print("read file")
    starttime = time.time()
    G = read_graph_from_csv(file_path)
    endtime = time.time()
    executiontime=endtime-starttime
    current_time = datetime.now()
    time_string = current_time.strftime("%Y%m%d_%H%M%S")
    # Create the file name using the formatted time string
    oneoutputfile = f"{FileNameHead}-{time_string}.txt"

    with open(oneoutputfile, 'a') as f:
            f.write(f"reading file {file_path} takes {executiontime} seconds \n\n")

    starttime = time.time()
    lvs,mvs,rvs=get_lvs_mvs_rvs(G)
    endtime = time.time()
    executiontime=endtime-starttime

    with open(oneoutputfile, 'a') as f:
            f.write(f"calculating vertex set {file_path} takes {executiontime} seconds \n\n")
            f.write(f"lvs size={len(lvs)}, mvs={len(mvs)}, rvs={len(rvs)}\n\n")


    starttime = time.time()
    #print("before call, mapping_weight",calculate_gained_weight(G,theid_mapping))
    mapping = optimize_ids(G, lvs, mvs, rvs,oneoutputfile,theid_mapping)
    mapping_weight=calculate_gained_weight(G,mapping)

    print("after call, mapping_weight",calculate_gained_weight(G,mapping))
    total_weight = total_edge_weight(G)
    endtime = time.time()
    executiontime=endtime-starttime
    with open(oneoutputfile, 'a') as f:
        f.write(f"Total Edge Weight:{total_weight}, Total number of Edges= {G.number_of_edges()}\n")
        f.write(f"Gained weight={mapping_weight},Percentage of remined Edges Weight: {mapping_weight/total_weight * 100}\n\n")

    starttime = time.time()
    current_time = datetime.now()
    time_string = current_time.strftime("%Y%m%d_%H%M%S")
    #Create the file name using the formatted time string
    output_file = f"{FileNameHead}-07-Relabel-{time_string}.csv"
    write_relabelled_nodes_to_file(mapping, output_file)
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


