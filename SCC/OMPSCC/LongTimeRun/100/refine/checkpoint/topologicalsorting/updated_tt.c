#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <limits.h>

#define MAX_EDGES 1000000  // Maximum number of edges (adjust as needed)

// Structure to represent an edge in a directed, weighted graph
typedef struct {
    int source;
    int target;
    int weight;
} Edge;

// Graph representation: Adjacency list (using dynamic arrays)
typedef struct {
    int vertex_count;
    int edge_count;
    Edge* edges; // Array of edges
} Graph;

// Function to add an edge to the graph
void add_edge(Graph *graph, int source, int target, int weight) {
    graph->edges[graph->edge_count].source = source;
    graph->edges[graph->edge_count].target = target;
    graph->edges[graph->edge_count].weight = weight;
    graph->edge_count++;
}

// Initialize graph
Graph* init_graph(int vertex_count, int max_edges) {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->vertex_count = vertex_count;
    graph->edge_count = 0;
    graph->edges = (Edge*)malloc(max_edges * sizeof(Edge));
    return graph;
}

// Free memory
void free_graph(Graph* graph) {
    free(graph->edges);
    free(graph);
}

// Function to calculate the feedback arc set weight for a given vertex ordering
int calculate_feedback_arc_set(Graph *graph, int* ordering) {
    int total_weight = 0;

    #pragma omp parallel for reduction(+:total_weight)
    for (int i = 0; i < graph->edge_count; i++) {
        int source_pos = ordering[graph->edges[i].source];
        int target_pos = ordering[graph->edges[i].target];

        if (source_pos > target_pos) {
            total_weight += graph->edges[i].weight;
        }
    }
    return total_weight;
}

// Function to construct a DAG by removing edges in the feedback arc set
void build_DAG(Graph* graph, int* ordering, int** adj_list, int* in_degree) {
    // Initialize adjacency list and in-degree count
    for (int i = 0; i < graph->vertex_count; i++) {
        in_degree[i] = 0;
    }

    // Build the adjacency list based on the current ordering
    for (int i = 0; i < graph->edge_count; i++) {
        int u = graph->edges[i].source;
        int v = graph->edges[i].target;
        int source_pos = ordering[u];
        int target_pos = ordering[v];

        if (source_pos < target_pos) {
            adj_list[u][in_degree[u]++] = v; // Store v as a neighbor of u
        }
    }
}

// Function to perform topological sort using Kahn's algorithm
void topological_sort(int vertex_count, int** adj_list, int* in_degree, int* topo_order) {
    int* queue = (int*)malloc(vertex_count * sizeof(int));
    int front = 0, rear = 0;

    // Initialize queue with nodes having zero in-degree
    for (int i = 0; i < vertex_count; i++) {
        if (in_degree[i] == 0) {
            queue[rear++] = i;
        }
    }

    int index = 0;
    while (front < rear) {
        // Taking the first node from the queue, but we can shuffle for non-determinism
        int random_index = front + rand() % (rear - front);
        int node = queue[random_index];
        queue[random_index] = queue[front]; // Swap to remove it from the queue
        queue[front++] = node;

        topo_order[index++] = node;

        // Reduce in-degree of all neighbors
        for (int i = 0; i < in_degree[node]; i++) {
            int neighbor = adj_list[node][i];
            in_degree[neighbor]--;
            if (in_degree[neighbor] == 0) {
                queue[rear++] = neighbor;
            }
        }
    }

    free(queue);
}

// Main optimization routine
void optimize_feedback_arc_set(Graph *graph, int *ordering, int *current_score) {
    int max_iterations = 100;
    int progress_interval = 10;
    int iteration = 0;

    // Temporary storage for DAG
    int* in_degree = (int*)calloc(graph->vertex_count, sizeof(int));
    int** adj_list = (int**)malloc(graph->vertex_count * sizeof(int*));
    for (int i = 0; i < graph->vertex_count; i++) {
        adj_list[i] = (int*)malloc(graph->vertex_count * sizeof(int)); // Adjust size as needed
    }

    int* topo_order = (int*)malloc(graph->vertex_count * sizeof(int));

    while (iteration < max_iterations) {
        int improved = 0;
        int best_delta_score = 0;

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < 10000; i++) {
            // Implement heuristic search or optimization (e.g., select and permute nodes)

            int delta_score = calculate_feedback_arc_set(graph, ordering);

            #pragma omp critical
            {
                if (delta_score > best_delta_score) {
                    best_delta_score = delta_score;
                    improved = 1;
                }
            }
        }

        if (improved) {
            *current_score += best_delta_score;
            printf("Iteration %d, New score: %d\n", iteration, *current_score);
        }

        // Construct DAG and perform topological sorting
        build_DAG(graph, ordering, adj_list, in_degree);
        topological_sort(graph->vertex_count, adj_list, in_degree, topo_order);

        // Update ordering based on topological sort
        for (int i = 0; i < graph->vertex_count; i++) {
            ordering[topo_order[i]] = i;
        }

        iteration++;
        if (iteration % progress_interval == 0) {
            printf("Progress: %d iterations completed.\n", iteration);
        }
    }

    // Free temporary storage
    free(in_degree);
    for (int i = 0; i < graph->vertex_count; i++) {
        free(adj_list[i]);
    }
    free(adj_list);
    free(topo_order);
}

// Function to map long node ids to integer ids
void map_node_ids(FILE* order_file, long* original_to_mapped, int* ordering) {
    char line[256];
    int count = 0;

    // Skip header
    fgets(line, sizeof(line), order_file);

    // Read order file and map node ids
    while (fgets(line, sizeof(line), order_file)) {
        long node;
        int order;
        sscanf(line, "%ld,%d", &node, &order);
        ordering[count] = order;
        original_to_mapped[node] = count++;
    }
}

// Function to write the ordering to a file
void write_order_to_file(const char* filename, int* ordering, int vertex_count) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error opening output file");
        return;
    }

    fprintf(file, "Vertex,Order\n");
    for (int i = 0; i < vertex_count; i++) {
        fprintf(file, "%d,%d\n", i, ordering[i]);
    }
    fclose(file);
}

// Main function
int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <graph_file> <order_file>\n", argv[0]);
        return 1;
    }

    // Open input files
    FILE* graph_file = fopen(argv[1], "r");
    FILE* order_file = fopen(argv[2], "r");
    if (!graph_file || !order_file) {
        perror("Error opening file");
        return 1;
    }

    // Skip the first line of the graph file
    char line[256];
    fgets(line, sizeof(line), graph_file);

    // Read edges from the graph file
    int vertex_count = 0;
    int edge_count = 0;
    long max_node = 0;
    Edge edges[MAX_EDGES];
    while (fgets(line, sizeof(line), graph_file)) {
        long source, target;
        int weight;
        sscanf(line, "%ld,%ld,%d", &source, &target, &weight);
        edges[edge_count].source = source;
        edges[edge_count].target = target;
        edges[edge_count].weight = weight;
        if (source > max_node) max_node = source;
        if (target > max_node) max_node = target;
        edge_count++;
    }
    vertex_count = max_node + 1; // Max node id + 1 is the vertex count

    // Initialize graph
    Graph* graph = init_graph(vertex_count, edge_count);
    for (int i = 0; i < edge_count; i++) {
        add_edge(graph, edges[i].source, edges[i].target, edges[i].weight);
    }

    // Map long node ids to internal integer ids
    long* original_to_mapped = (long*)calloc(vertex_count, sizeof(long));
    int* ordering = (int*)malloc(vertex_count * sizeof(int));
    map_node_ids(order_file, original_to_mapped, ordering);

    int current_score = 0;
    current_score = calculate_feedback_arc_set(graph, ordering);
    printf("Initial score: %d\n", current_score);

    // Perform optimization
    optimize_feedback_arc_set(graph, ordering, &current_score);

    // Write the resulting ordering to a file
    write_order_to_file("output_ordering.csv", ordering, vertex_count);

    // Clean up
    free(ordering);
    free(original_to_mapped);
    free_graph(graph);
    fclose(graph_file);
    fclose(order_file);

    return 0;
}

