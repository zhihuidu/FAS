#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <time.h>
#include <sys/time.h>

#define MAX_NODES 136648
#define MAX_EDGES 5657719
#define MAX_CYCLE_LENGTH 100

typedef struct {
    int source;
    int target;
    int weight;
} Edge;

typedef struct {
    int* edges;  // Dynamic array of edge indices
    int count;   // Number of edges
    int capacity; // Capacity of the dynamic array
} EdgeInfo;

typedef struct {
    int num_nodes;
    int num_edges;
    int vertices[MAX_NODES];  // List of vertices
    Edge edges[MAX_EDGES];    // Static array of all edges

    EdgeInfo out_edge[MAX_NODES];  // Outgoing edges information
    EdgeInfo in_edge[MAX_NODES];   // Incoming edges information
} Graph;

int updated_weights[MAX_EDGES];
int removed_edges[MAX_EDGES];


// Function to initialize the graph
void initialize_graph(Graph* graph) {
    graph->num_nodes = 0;
    graph->num_edges = 0;

    for (int i = 0; i < MAX_NODES; i++) {
        graph->vertices[i] = -1;
        graph->out_edge[i].edges = NULL;
        graph->out_edge[i].count = 0;
        graph->out_edge[i].capacity = 0;

        graph->in_edge[i].edges = NULL;
        graph->in_edge[i].count = 0;
        graph->in_edge[i].capacity = 0;
    }

    for (int i = 0; i < MAX_EDGES; i++) {
        updated_weights[i] = 0;
        removed_edges[i] = 0;
    }
}

// Function to find or add a vertex
int find_or_add_vertex(Graph* graph, int vertex) {
    for (int i = 0; i < graph->num_nodes; i++) {
        if (graph->vertices[i] == vertex) {
            return i;
        }
    }
    graph->vertices[graph->num_nodes] = vertex;
    return graph->num_nodes++;
}

// Function to dynamically add an edge to EdgeInfo
void add_edge_to_info(EdgeInfo* edge_info, int edge_index) {
    if (edge_info->count == edge_info->capacity) {
        edge_info->capacity = edge_info->capacity == 0 ? 2 : edge_info->capacity * 2;
        edge_info->edges = realloc(edge_info->edges, edge_info->capacity * sizeof(int));
    }
    edge_info->edges[edge_info->count++] = edge_index;
}

// Function to add an edge to the graph
void add_edge(Graph* graph, int source, int target, int weight) {
    int edge_index = graph->num_edges++;
    graph->edges[edge_index].source = source;
    graph->edges[edge_index].target = target;
    graph->edges[edge_index].weight = weight;

    updated_weights[edge_index] = weight;

    // Update out_edge info
    int source_index = find_or_add_vertex(graph, source);
    add_edge_to_info(&graph->out_edge[source_index], edge_index);

    // Update in_edge info
    //int target_index = find_or_add_vertex(graph, target);
    //add_edge_to_info(&graph->in_edge[target_index], edge_index);
}

// Function to read the graph from a CSV file
void read_graph_from_csv(const char* filename, Graph* graph) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Cannot open file %s\n", filename);
        exit(1);
    }

    int source, target, weight;
    while (fscanf(file, "%d,%d,%d\n", &source, &target, &weight) != EOF) {
        //printf("source=%d,target=%d,weight=%d\n",source,target,weight);
        add_edge(graph, source, target, weight);
    }
    printf("add all edges to the graph\n");

    fclose(file);
}


// Function to find cycles up to a maximum length
void find_cycles(Graph* graph, int start, int current, int length, int* path, bool* visited , int * num_cycle) {
    if (length > MAX_CYCLE_LENGTH) return;

    path[length - 1] = current;

    int source_index = find_or_add_vertex(graph, current);
    visited[source_index] = true;
    EdgeInfo* out_edges = &graph->out_edge[source_index];



    for (int i = 0; i < out_edges->count; i++) {
        int edge_index = out_edges->edges[i];
        Edge* edge = &graph->edges[edge_index];
        int target_index = find_or_add_vertex(graph, edge->target);
        if (edge->target == start && length > 1) {
            // Cycle detected
            int min_weight = INT_MAX;
            num_cycle[0]+=1;
            printf("find %d(th) cycle starting from vertex %d with length %d\n",num_cycle[0], start,length);
            for (int j = 0; j < length - 1; j++) {
                int u = path[j];
                int v = path[j + 1];
                int u_index = find_or_add_vertex(graph, u);
                EdgeInfo* u_out_edges = &graph->out_edge[u_index];
                for (int k = 0; k < u_out_edges->count; k++) {
                    int idx = u_out_edges->edges[k];
                    if (graph->edges[idx].target == v) {
                        if (graph->edges[idx].weight < min_weight) {
                            min_weight = graph->edges[idx].weight;
                        }
                    }
                }
            }


            // Update the weights in the updated_weights array
            for (int j = 0; j < length - 1; j++) {
                int u = path[j];
                int v = path[j + 1];
                int u_index = find_or_add_vertex(graph, u);
                EdgeInfo* u_out_edges = &graph->out_edge[u_index];
                for (int k = 0; k < u_out_edges->count; k++) {
                    int idx = u_out_edges->edges[k];
                    if (graph->edges[idx].target == v) {
                        updated_weights[idx] -= min_weight;
                    }
                }
            }
            updated_weights[edge_index] -= min_weight;
        } else if ( !visited[target_index] && edge->target > start) {
            find_cycles(graph, start, edge->target, length + 1, path, visited,num_cycle);
        }
    }

    visited[source_index] = false;
}

// Function to mark removed edges
void mark_removed_edges(Graph* graph, int start, int current, int length, int* path, bool* visited,int * num_edges) {
    if (length > MAX_CYCLE_LENGTH) return;

    path[length - 1] = current;

    int source_index = find_or_add_vertex(graph, current);
    visited[source_index] = true;
    EdgeInfo* out_edges = &graph->out_edge[source_index];

    for (int i = 0; i < out_edges->count; i++) {
        int edge_index = out_edges->edges[i];
        Edge* edge = &graph->edges[edge_index];
        int target_index = find_or_add_vertex(graph, edge->target);
        if (edge->target == start && length > 1 && removed_edges[edge_index] == 0) {
            // Cycle detected
            int min_weight = INT_MAX;
            int has_cycle = 1;
            int min_edge_index = -1;
            for (int j = 0; j < length - 1; j++) {
                int u = path[j];
                int v = path[j + 1];
                int u_index = find_or_add_vertex(graph, u);
                EdgeInfo* u_out_edges = &graph->out_edge[u_index];
                for (int k = 0; k < u_out_edges->count; k++) {
                    int idx = u_out_edges->edges[k];
                    if (graph->edges[idx].target == v) {
                        if (removed_edges[idx] ==1) {
                             has_cycle=0;
                             j=length;
                             printf("edge (%d, %d) has been removed\n",u,v);
                             break;
                        }
                        if (updated_weights[idx] < min_weight) {
                            min_weight = updated_weights[idx];
                            min_edge_index = idx;
                        }
                    }
                }
            }

            // Mark the edge as removed
            if (has_cycle==1) {
                removed_edges[min_edge_index] = 1;
                num_edges[0]+=1;
                printf("removed %d(th) edge (%d,%d)\n",num_edges[0], graph->edges[min_edge_index].source,graph->edges[min_edge_index].target);
            }
        } else if (!visited[target_index] && edge->target > start) {
            mark_removed_edges(graph, start, edge->target, length + 1, path, visited, num_edges);
        }
    }

    visited[source_index] = false;
}


int main(int argc, char * argv[]) {
    const char* input_filename = argv[1];
    //const char* input_filename = "graph.csv";
    //char* output_filename = "removed-edge.txt";

    struct timeval start, end;
    long seconds, useconds;
    double elapsed;

    time_t now = time(NULL);
    struct tm *t = localtime(&now);

    // Create a filename with the timestamp
    char output_filename[100];
    strftime(output_filename, sizeof(output_filename), "removed-edge-%Y%m%d-%H%M%S.txt", t);
    Graph graph;
    printf("read file name=%s, writefile name=%s\n",input_filename,output_filename);

    gettimeofday(&start, NULL);

    initialize_graph(&graph);
    printf("read data\n");
    read_graph_from_csv(input_filename, &graph);
    printf("finish read data\n");

    gettimeofday(&end, NULL);
    seconds  = end.tv_sec  - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;
    elapsed = seconds + useconds/1000000.0;
    printf("Initial graph and read data taken: %f seconds\n\n", elapsed);

    exit(0);
    gettimeofday(&start, NULL);

    // Find cycles and update weights
    int number_cycles=0;
    for (int i = 0; i < graph.num_nodes; i++) {
        bool visited[MAX_NODES] = {0};
        int path[MAX_CYCLE_LENGTH];
        //int min_cycle_weight = INT_MAX;
        find_cycles(&graph, graph.vertices[i], graph.vertices[i], 1, path, visited,&number_cycles);
    }

    gettimeofday(&end, NULL);
    seconds  = end.tv_sec  - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;
    elapsed = seconds + useconds/1000000.0;
    printf("Find %d Cycles with the largest length %d, Time taken: %f seconds\n\n",number_cycles, MAX_CYCLE_LENGTH, elapsed);

    // Mark removed edges
    gettimeofday(&start, NULL);
    int number_edges=0;
    for (int i = 0; i < graph.num_nodes; i++) {
        bool visited[MAX_NODES] = {0};
        int path[MAX_CYCLE_LENGTH];
        mark_removed_edges(&graph, i, i, 1, path, visited,&number_edges);
    }

    gettimeofday(&end, NULL);
    seconds  = end.tv_sec  - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;
    elapsed = seconds + useconds/1000000.0;
    printf("Mark %d removed edges in Cycles with the largest length %d, Time taken: %f seconds\n\n",number_edges, MAX_CYCLE_LENGTH, elapsed);
    // Output removed edges to a file
    FILE* output_file = fopen(output_filename, "w");
    if (output_file == NULL) {
        printf("Error: Cannot open file %s for writing\n", output_filename);
        exit(1);
    }

    for (int i = 0; i < graph.num_edges; i++) {
        if (removed_edges[i] == 1) {
            fprintf(output_file, "%d,%d\n", graph.edges[i].source, graph.edges[i].target);
        }
    }

    fclose(output_file);
    printf("Removed edges have been successfully written to %s\n", output_filename);

    // Free dynamically allocated memory
    for (int i = 0; i < graph.num_nodes; i++) {
        free(graph.out_edge[i].edges);
        //free(graph.in_edge[i].edges);
    }

    return 0;
}


