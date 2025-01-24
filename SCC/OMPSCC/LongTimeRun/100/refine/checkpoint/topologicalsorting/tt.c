#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <limits.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>

// Structure to represent an edge in the graph
typedef struct {
    int source;
    int target;
    int weight;
} Edge;

// Graph representation using adjacency list
typedef struct {
    int vertex_count;
    int edge_count;
    Edge *edges;
} Graph;

// A node in the hash map for large-to-small ID mapping
typedef struct NodeIDMap {
    long int original_id;
    int mapped_id;
    struct NodeIDMap *next;
} NodeIDMap;

// Function to create a hash map node
NodeIDMap *create_map_node(long int original_id, int mapped_id) {
    NodeIDMap *node = (NodeIDMap *)malloc(sizeof(NodeIDMap));
    node->original_id = original_id;
    node->mapped_id = mapped_id;
    node->next = NULL;
    return node;
}

// Hash function for large IDs
int hash_function(long int id, int size) {
    return id % size;
}

// Function to insert into the hash map
void insert_id_mapping(NodeIDMap **hash_map, int size, long int original_id, int mapped_id) {
    int index = hash_function(original_id, size);
    NodeIDMap *new_node = create_map_node(original_id, mapped_id);
    if (hash_map[index] == NULL) {
        hash_map[index] = new_node;
    } else {
        NodeIDMap *current = hash_map[index];
        while (current->next != NULL) {
            current = current->next;
        }
        current->next = new_node;
    }
}

// Function to get mapped ID from hash map
int get_mapped_id(NodeIDMap **hash_map, int size, long int original_id) {
    int index = hash_function(original_id, size);
    NodeIDMap *current = hash_map[index];
    while (current != NULL) {
        if (current->original_id == original_id) {
            return current->mapped_id;
        }
        current = current->next;
    }
    return -1; // Return -1 if not found
}

// Function to initialize graph
Graph* init_graph(int vertex_count, int max_edges) {
    Graph *graph = (Graph *)malloc(sizeof(Graph));
    graph->vertex_count = vertex_count;
    graph->edge_count = 0;
    graph->edges = (Edge *)malloc(max_edges * sizeof(Edge));
    return graph;
}

// Function to free graph memory
void free_graph(Graph* graph) {
    free(graph->edges);
    free(graph);
}

// Function to add edge to graph
void add_edge(Graph *graph, int source, int target, int weight) {
    graph->edges[graph->edge_count].source = source;
    graph->edges[graph->edge_count].target = target;
    graph->edges[graph->edge_count].weight = weight;
    graph->edge_count++;
}

// Function to read the graph and map node IDs from file
Graph* read_graph_from_file(const char *filename, int *vertex_count, NodeIDMap **id_mapping, int *next_mapped_id, int hash_size) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening graph file.\n");
        exit(EXIT_FAILURE);
    }

    char buffer[256];
    fgets(buffer, sizeof(buffer), file); // Skip the header

    Graph *graph = init_graph(0, 1000000); // Assuming large graph
    long int source_id, target_id;
    int weight;

    while (fgets(buffer, sizeof(buffer), file)) {
        sscanf(buffer, "%ld,%ld,%d", &source_id, &target_id, &weight);

        // Map large node IDs to smaller internal IDs
        int mapped_source, mapped_target;
        if ((mapped_source = get_mapped_id(id_mapping, hash_size, source_id)) == -1) {
            mapped_source = (*next_mapped_id)++;
            insert_id_mapping(id_mapping, hash_size, source_id, mapped_source);
        }

        if ((mapped_target = get_mapped_id(id_mapping, hash_size, target_id)) == -1) {
            mapped_target = (*next_mapped_id)++;
            insert_id_mapping(id_mapping, hash_size, target_id, mapped_target);
        }

	printf("mapping original source %ld, to %d, original target %ld to %d\n",source_id,mapped_source,target_id, mapped_target);
        add_edge(graph, mapped_source, mapped_target, weight);
    }

    fclose(file);
    *vertex_count = *next_mapped_id;  // Set the total number of vertices
    return graph;
}

// Function to read the initial order and map node IDs
int* read_order_from_file(const char *filename, int vertex_count, NodeIDMap **id_mapping, int hash_size) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening order file.\n");
        exit(EXIT_FAILURE);
    }

    char buffer[256];
    fgets(buffer, sizeof(buffer), file);  // Skip the header

    int *ordering = (int *)malloc(vertex_count * sizeof(int));
    long int node_id;
    int order;

    while (fgets(buffer, sizeof(buffer), file)) {
        sscanf(buffer, "%ld,%d", &node_id, &order);
        int mapped_id = get_mapped_id(id_mapping, hash_size, node_id);
        ordering[mapped_id] = order;
	printf("mapping original ID %ld, to %d, and order is  %d\n",node_id,mapped_id, order);
    }

    fclose(file);
    return ordering;
}

// Function to write the new order to a file
void write_order_to_file(const char *filename, int *ordering, int vertex_count, NodeIDMap **id_mapping, int hash_size) {
    FILE *outfile = fopen(filename, "w");
    if (!outfile) {
        fprintf(stderr, "Error opening file %s for writing.\n", filename);
        exit(EXIT_FAILURE);
    }

    fprintf(outfile, "Node,Order\n");
    for (int i = 0; i < vertex_count; i++) {
        // Find the original ID for each mapped ID
        for (int j = 0; j < hash_size; j++) {
            NodeIDMap *current = id_mapping[j];
            while (current != NULL) {
                if (current->mapped_id == i) {
                    fprintf(outfile, "%ld,%d\n", current->original_id, ordering[i]);
                }
                current = current->next;
            }
        }
    }

    fclose(outfile);
}

// Function to calculate the feedforward arc set weight
int calculate_feedforward_arc_set(Graph *graph, int *ordering) {
    int total_weight = 0;

    #pragma omp parallel for reduction(+:total_weight)
    for (int i = 0; i < graph->edge_count; i++) {
        int source_pos = ordering[graph->edges[i].source];
        int target_pos = ordering[graph->edges[i].target];

        if (source_pos < target_pos) {
            total_weight += graph->edges[i].weight;
        }
    }
    return total_weight;
}

// Function to construct a directed acyclic graph (DAG) from the current ordering
void build_DAG(Graph *graph, int *ordering, int **adj_list, int *in_degree) {
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
            adj_list[u][in_degree[v]++] = v; // Store v as a neighbor of u
        }
    }
}




// Function to perform topological sort using Kahn's algorithm with randomization
void topological_sort(int vertex_count, int **adj_list, int *in_degree, int *topo_order) {
    int *queue = (int *)malloc(vertex_count * sizeof(int));
    int front = 0, rear = 0;

    // Initialize the random seed
    srand(time(NULL));

    // Initialize queue with nodes having zero in-degree
    for (int i = 0; i < vertex_count; i++) {
        if (in_degree[i] == 0) {
            queue[rear++] = i;
        }
    }

    int index = 0;
    while (front < rear) {
        // Select a random node from the queue
        int rand_index = front + rand() % (rear - front);
        int node = queue[rand_index];

        // Swap the randomly picked node with the front node
        queue[rand_index] = queue[front];
        queue[front] = node;

        // Remove the selected node from the queue
        front++;

        // Add node to the topological order
        topo_order[index++] = node;

        // Reduce in-degree of all neighbors and add new zero in-degree nodes to the queue
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
void optimize_feedback_arc_set(Graph *graph, int *ordering, int *current_score, int vertex_count, NodeIDMap **id_mapping, int hash_size) {
    int max_iterations = 100;
    int iteration;

    // Temporary storage for DAG
    int *in_degree = (int *)calloc(graph->vertex_count, sizeof(int));
    int **adj_list = (int **)malloc(graph->vertex_count * sizeof(int *));
    for (int i = 0; i < graph->vertex_count; i++) {
        adj_list[i] = (int *)malloc(graph->vertex_count * sizeof(int));
    }

    getchar();
    // Use OpenMP to parallelize the iterations
    //#pragma omp parallel private(iteration) shared(ordering, current_score, graph, adj_list, in_degre)
    {
        // Each thread will handle different iterations
        //#pragma omp for
        for (iteration = 0; iteration < max_iterations; iteration++) {
            int thread_id = omp_get_thread_num();
            int total_threads = omp_get_num_threads();

	    printf("iteration %d thread %d of %d \n",iteration,thread_id,total_threads);
            fflush(stdout);
            // Build the DAG based on the current ordering
            build_DAG(graph, ordering, adj_list, in_degree);
	    printf("build dag\n");
            fflush(stdout);
            
            // Perform topological sort
            int *topo_order = (int *)malloc(graph->vertex_count * sizeof(int));
            topological_sort(graph->vertex_count, adj_list, in_degree, topo_order);
	    printf("sort \n");
            fflush(stdout);

            // Calculate the new score
            int new_score = calculate_feedback_arc_set(graph, topo_order);
	    printf("new score \n");

            fflush(stdout);


            // Lock to ensure that only one thread updates the ordering and score
            #pragma omp critical
            {
                if (new_score < *current_score) {
                    *current_score = new_score;
                    memcpy(ordering, topo_order, graph->vertex_count * sizeof(int));
                    printf("Iteration %d: New score = %d\n", iteration, new_score);
		    // Write new solution to CSV file
                    char filename[50];
                    sprintf(filename, "%d.csv", new_score);
                    write_order_to_file(filename, ordering, vertex_count, id_mapping, hash_size);
                } else {
                    printf("Iteration %d: No improvement\n", iteration);
                }
            }

            free(topo_order);
        }
    }








    // Free allocated memory
    for (int i = 0; i < graph->vertex_count; i++) {
        free(adj_list[i]);
    }
    free(adj_list);
    free(in_degree);
}


// Main function
int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <graph_file> <initial_order_file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int vertex_count;
    int hash_size = 1000000;  // Hash map size, should be chosen based on expected range of large IDs
    NodeIDMap **id_mapping = (NodeIDMap **)calloc(hash_size, sizeof(NodeIDMap *));
    int next_mapped_id = 0;

    Graph *graph = read_graph_from_file(argv[1], &vertex_count, id_mapping, &next_mapped_id, hash_size);
    int *ordering = read_order_from_file(argv[2], vertex_count, id_mapping, hash_size);

    int current_score = calculate_feedback_arc_set(graph, ordering);
    printf("Initial score: %d\n", current_score);

    optimize_feedback_arc_set(graph, ordering, &current_score,vertex_count, id_mapping, hash_size);

    write_order_to_file("optimized_order.csv", ordering, vertex_count, id_mapping, hash_size);

    free(ordering);
    free_graph(graph);

    return 0;
}





