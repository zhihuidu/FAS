#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#define MAX_PERMUTATIONS 24

typedef struct {
    int source;
    int target;
    int weight;
} Edge;

typedef struct {
    int64_t original_id; //only the original ID is an int64, the mapped id is an int
    int mapped_id;
} NodeMap;

typedef struct {
    int node_count;
    int edge_count;
    Edge *edges;
    NodeMap *node_map;
} Graph;

typedef struct {
    int *ordering;    // mapping from position to node_id, node_id and position are int
    int *position;    // mapping from node_id to position
    int size;
} Ordering;

// Function prototypes
void read_graph(const char *filename, Graph *graph);
void read_ordering(const char *filename, Ordering *ordering, Graph *graph);
int compute_initial_score(Graph *graph, Ordering *ordering);
void generate_permutations(int *nodes, int start, int end, int permutations[][4], int *count);
void improve_solution(Graph *graph, Ordering *ordering, int *current_score);

void __attribute__((constructor)) premain() {
    printf("Program started.\n");
    fflush(stdout);
}

int main() {
    Graph graph;
    Ordering ordering;
    int current_score;

    // Initialize random seed
    srand(time(NULL));

    printf("Starting to read the graph.\n");
    fflush(stdout);
    // Read the graph
    read_graph("connectome_graph.csv", &graph);
    printf("Finished reading the graph.\n");
    fflush(stdout);

    // Print the size of the input graph
    printf("Graph has %d vertices and %d edges.\n", graph.node_count, graph.edge_count);
    fflush(stdout);

    printf("Starting to read the initial feasible solution.\n");
    fflush(stdout);
    // Read the initial feasible solution
    read_ordering("best.csv", &ordering, &graph);
    printf("Finished reading the initial feasible solution.\n");
    fflush(stdout);

    // Compute the initial score
    current_score = compute_initial_score(&graph, &ordering);
    printf("Initial score: %.0f\n", current_score);
    fflush(stdout);

    // Start improving the solution
    printf("Starting to improve the solution.\n");
    fflush(stdout);
    improve_solution(&graph, &ordering, &current_score);

    // Free allocated memory
    free(graph.edges);
    free(graph.node_map);
    free(ordering.ordering);
    free(ordering.position);

    printf("Program completed successfully.\n");
    fflush(stdout);
    return 0;
}

void read_graph(const char *filename, Graph *graph) {
    printf("Entering read_graph.\n");
    fflush(stdout);

    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening graph file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Dynamically allocate line buffer
    char *line = malloc(8192);
    if (!line) {
        fprintf(stderr, "Failed to allocate memory for line buffer.\n");
        exit(EXIT_FAILURE);
    }

    int edge_capacity = 10000;
    int node_capacity = 10000;
    int edge_count = 0;
    int node_id_count = 0;

    Edge *edges = malloc(edge_capacity * sizeof(Edge));
    NodeMap *node_map = malloc(node_capacity * sizeof(NodeMap));
    int64_t *node_ids = malloc(node_capacity * sizeof(int64_t)); //here is the original node id

    if (!edges || !node_map || !node_ids) {
        fprintf(stderr, "Memory allocation failed while initializing graph structures.\n");
        exit(EXIT_FAILURE);
    }

    // Skip header
    if (fgets(line, 8192, file) == NULL) {
        fprintf(stderr, "Graph file is empty or header is missing.\n");
        exit(EXIT_FAILURE);
    }

    while (fgets(line, 8192, file)) {
        int64_t source_id, target_id;
        int weight;
        char *token;

        // Remove newline character
        line[strcspn(line, "\r\n")] = 0;

        // Skip empty lines
        if (strlen(line) == 0) {
            continue;
        }

        // Parse source ID
        token = strtok(line, ",");
        if (!token) {
            fprintf(stderr, "Malformed line in graph file: %s\n", line);
            continue;
        }
        source_id = atoll(token);

        // Parse target ID
        token = strtok(NULL, ",");
        if (!token) {
            fprintf(stderr, "Malformed line in graph file: %s\n", line);
            continue;
        }
        target_id = atoll(token);

        // Parse weight
        token = strtok(NULL, ",");
        if (!token) {
            fprintf(stderr, "Malformed line in graph file: %s\n", line);
            continue;
        }
        weight = atoi(token);

        // Map source_id and target_id to indices
        int source_mapped = -1;
        int target_mapped = -1;

        // Use a hash map or binary search for better performance in large graphs
        // For simplicity, we use a linear search here
        for (int i = 0; i < node_id_count; i++) {
            if (node_ids[i] == source_id) {
                source_mapped = i;
            }
            if (node_ids[i] == target_id) {
                target_mapped = i;
            }
            if (source_mapped != -1 && target_mapped != -1) {
                break;
            }
        }

        if (source_mapped == -1) {
            if (node_id_count == node_capacity) {
                node_capacity *= 2;
                node_ids = realloc(node_ids, node_capacity * sizeof(int64_t));
                node_map = realloc(node_map, node_capacity * sizeof(NodeMap));
                if (!node_ids || !node_map) {
                    fprintf(stderr, "Memory allocation failed while expanding node arrays.\n");
                    exit(EXIT_FAILURE);
                }
            }
            source_mapped = node_id_count;
            node_ids[node_id_count++] = source_id;
        }

        if (target_mapped == -1) {
            if (node_id_count == node_capacity) {
                node_capacity *= 2;
                node_ids = realloc(node_ids, node_capacity * sizeof(int64_t));
                node_map = realloc(node_map, node_capacity * sizeof(NodeMap));
                if (!node_ids || !node_map) {
                    fprintf(stderr, "Memory allocation failed while expanding node arrays.\n");
                    exit(EXIT_FAILURE);
                }
            }
            target_mapped = node_id_count;
            node_ids[node_id_count++] = target_id;
        }

        if (edge_count == edge_capacity) {
            edge_capacity *= 2;
            edges = realloc(edges, edge_capacity * sizeof(Edge));
            if (!edges) {
                fprintf(stderr, "Memory allocation failed while expanding edges array.\n");
                exit(EXIT_FAILURE);
            }
        }

        edges[edge_count].source = source_mapped;
        edges[edge_count].target = target_mapped;
        edges[edge_count].weight = weight;
        edge_count++;
    }

    fclose(file);
    free(line);

    // Build node mapping
    for (int i = 0; i < node_id_count; i++) {
        node_map[i].original_id = node_ids[i];
        node_map[i].mapped_id = i;
    }

    graph->edges = edges;
    graph->edge_count = edge_count;
    graph->node_count = node_id_count;
    graph->node_map = node_map;

    free(node_ids);

    printf("Exiting read_graph.\n");
    fflush(stdout);
}

void read_ordering(const char *filename, Ordering *ordering, Graph *graph) {
    printf("Entering read_ordering.\n");
    fflush(stdout);

    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening ordering file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Dynamically allocate line buffer
    char *line = malloc(8192);
    if (!line) {
        fprintf(stderr, "Failed to allocate memory for line buffer.\n");
        exit(EXIT_FAILURE);
    }

    int size = graph->node_count;
    int *order = malloc(size * sizeof(int));//here the id is not original id
    int *position = malloc(size * sizeof(int));

    if (!order || !position) {
        fprintf(stderr, "Memory allocation failed while initializing ordering structures.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize order and position arrays
    for (int i = 0; i < size; i++) {
        order[i] = -1;
        position[i] = -1;
    }

    // Skip header
    if (fgets(line, 8192, file) == NULL) {
        fprintf(stderr, "Ordering file is empty or header is missing.\n");
        exit(EXIT_FAILURE);
    }

    while (fgets(line, 8192, file)) {
        int64_t node_id;
        int pos;
        char *token;

        // Remove newline character
        line[strcspn(line, "\r\n")] = 0;

        // Skip empty lines
        if (strlen(line) == 0) {
            continue;
        }

        token = strtok(line, ",");
        if (!token) {
            fprintf(stderr, "Malformed line in ordering file: %s\n", line);
            continue;
        }
        node_id = atoll(token);

        token = strtok(NULL, ",");
        if (!token) {
            fprintf(stderr, "Malformed line in ordering file: %s\n", line);
            continue;
        }
        pos = atoi(token);

        // Map node_id to mapped_id
        int mapped_id = -1;
        for (int i = 0; i < graph->node_count; i++) {
            if (graph->node_map[i].original_id == node_id) {
                mapped_id = graph->node_map[i].mapped_id;
                break;
            }
        }

        if (mapped_id == -1) {
            fprintf(stderr, "Node ID %lld in ordering not found in graph.\n", node_id);
            exit(EXIT_FAILURE);
        }

        if (pos < 0 || pos >= size) {
            fprintf(stderr, "Invalid position %d for node ID %lld.\n", pos, node_id);
            exit(EXIT_FAILURE);
        }

        if (order[pos] != -1) {
            fprintf(stderr, "Duplicate position %d for node ID %lld.\n", pos, node_id);
            exit(EXIT_FAILURE);
        }

        order[pos] = mapped_id;
        position[mapped_id] = pos;
    }

    fclose(file);
    free(line);

    // Check if all positions are filled
    for (int i = 0; i < size; i++) {
        if (order[i] == -1) {
            fprintf(stderr, "Position %d is not assigned in the ordering.\n", i);
            exit(EXIT_FAILURE);
        }
    }

    ordering->ordering = order;
    ordering->position = position;
    ordering->size = size;

    printf("Exiting read_ordering.\n");
    fflush(stdout);
}

int compute_initial_score(Graph *graph, Ordering *ordering) {
    printf("Computing initial score.\n");
    fflush(stdout);

    int score = 0;
    for (int i = 0; i < graph->edge_count; i++) {
        int u = graph->edges[i].source;
        int v = graph->edges[i].target;
        int w = graph->edges[i].weight;
        if (ordering->position[u] < ordering->position[v]) {
            score += w;
        }
    }

    printf("Initial score computed.\n");
    fflush(stdout);
    return score;
}

void generate_permutations(int *nodes, int start, int end, int permutations[][4], int *count) {
    if (start == end) {
        for (int i = 0; i <= end; i++) {
            permutations[*count][i] = nodes[i];
        }
        (*count)++;
    } else {
        for (int i = start; i <= end; i++) {
            // Swap
            int temp = nodes[start];
            nodes[start] = nodes[i];
            nodes[i] = temp;

            generate_permutations(nodes, start + 1, end, permutations, count);

            // Swap back
            temp = nodes[start];
            nodes[start] = nodes[i];
            nodes[i] = temp;
        }
    }
}

void improve_solution(Graph *graph, Ordering *ordering, int *current_score) {
    printf("Entering improve_solution.\n");
    fflush(stdout);

    int iteration = 0;
    int progress_interval = 10000;

    int node_count = graph->node_count;
    int edge_count = graph->edge_count;

    // Build adjacency lists for nodes
    int **node_edges = malloc(node_count * sizeof(int *));//given node mapped id, list all its edges
    int *node_edge_counts = calloc(node_count, sizeof(int));//given node has how many edges
    if (!node_edges || !node_edge_counts) {
        fprintf(stderr, "Memory allocation failed while initializing adjacency lists.\n");
        exit(EXIT_FAILURE);
    }
    int *edge_counts = calloc(node_count, sizeof(int)); //how many edges connected to a node
    if (!edge_counts) {
        fprintf(stderr, "Memory allocation failed while initializing edge counts.\n");
        exit(EXIT_FAILURE);
    }

    // First pass to count edges per node
    for (int i = 0; i < edge_count; i++) {
        int u = graph->edges[i].source;
        int v = graph->edges[i].target;
        edge_counts[u]++;
        edge_counts[v]++;
    }
    // Allocate memory
    for (int i = 0; i < node_count; i++) {
        node_edges[i] = malloc(edge_counts[i] * sizeof(int));
        if (!node_edges[i]) {
            fprintf(stderr, "Memory allocation failed for adjacency list of node %d.\n", i);
            exit(EXIT_FAILURE);
        }
        node_edge_counts[i] = 0; // Reset for second pass
    }
    // Second pass to fill edge indices
    for (int i = 0; i < edge_count; i++) {
        int u = graph->edges[i].source;
        int v = graph->edges[i].target;
        node_edges[u][node_edge_counts[u]++] = i;
        node_edges[v][node_edge_counts[v]++] = i;
    }
    free(edge_counts);

    printf("Initialization of adjacency lists completed.\n");
    fflush(stdout);

    while (1) {
        // Randomly pick four distinct nodes
        int selected_nodes[4];
        int selected_count = 0;
        while (selected_count < 4) {
            int rand_node = rand() % node_count;
            int duplicate = 0;
            for (int i = 0; i < selected_count; i++) {
                if (selected_nodes[i] == rand_node) {
                    duplicate = 1;
                    break;
                }
            }
            if (!duplicate) {
                selected_nodes[selected_count++] = rand_node;
		printf("select node %d\n",rand_node);
            }
        }

        // Generate all permutations of these four nodes
        int permutations[MAX_PERMUTATIONS][4];
        int perm_count = 0;
        int nodes_copy[4];
        memcpy(nodes_copy, selected_nodes, 4 * sizeof(int));
        generate_permutations(nodes_copy, 0, 3, permutations, &perm_count);
	for (int i=0;i<perm_count;i++) {
		for (int j=0;i<4;j++){
			printf(" %d ",permutations[i][j]);
		}
		printf("\n");
	}

        // Collect all edges involving these nodes
        int *edges_to_check = NULL;
        int edges_to_check_count = 0;
        int edges_to_check_capacity = 0;
        for (int i = 0; i < 4; i++) {
            int node = selected_nodes[i];
            for (int j = 0; j < node_edge_counts[node]; j++) {
                int edge_index = node_edges[node][j];
                // Add edge_index to edges_to_check if not already added
                int already_added = 0;
                for (int k = 0; k < edges_to_check_count; k++) {
                    if (edges_to_check[k] == edge_index) {
                        already_added = 1;
                        break;
                    }
                }
                if (!already_added) {
                    if (edges_to_check_count == edges_to_check_capacity) {
                        edges_to_check_capacity = edges_to_check_capacity == 0 ? 10 : edges_to_check_capacity * 2;
                        edges_to_check = realloc(edges_to_check, edges_to_check_capacity * sizeof(int));
                        if (!edges_to_check) {
                            fprintf(stderr, "Memory allocation failed while expanding edges_to_check array.\n");
                            exit(EXIT_FAILURE);
                        }
                    }
                    edges_to_check[edges_to_check_count++] = edge_index;//list all unique edges to check
                }
            }
        }

        // Save original positions
        int original_positions[4];
        for (int i = 0; i < 4; i++) {
            original_positions[i] = ordering->position[selected_nodes[i]];
        }

        // Try all permutations
        int improved = 0;
        for (int p = 0; p < MAX_PERMUTATIONS; p++) {
            int *perm = permutations[p];
            int delta_score = 0;

            // Apply permutation to positions
            for (int i = 0; i < 4; i++) {
                int node = perm[i]; // The permuted node
                int pos = original_positions[i]; // The original position
                ordering->position[node] = pos;
                ordering->ordering[pos] = node;
            }

            // Calculate delta score
            for (int i = 0; i < edges_to_check_count; i++) {
                int edge_index = edges_to_check[i];
                int u = graph->edges[edge_index].source;
                int v = graph->edges[edge_index].target;
                int w = graph->edges[edge_index].weight;

                int pos_u_old = ordering->position[u];
                int pos_v_old = ordering->position[v];

                int contributes_old = pos_u_old < pos_v_old ? 1 : 0;

                // Since we've updated positions, get the new positions
                int pos_u_new = ordering->position[u];
                int pos_v_new = ordering->position[v];

                int contributes_new = pos_u_new < pos_v_new ? 1 : 0;

                delta_score += (contributes_new - contributes_old) * w;
            }

            if (delta_score > 0) {
                // Update current score
                *current_score += delta_score;
                printf("Iteration %d, New score: %.0f\n", iteration, *current_score);
                fflush(stdout);

                // Write new solution to CSV file
                char filename[50];
                sprintf(filename, "%.0f.csv", *current_score);

                FILE *outfile = fopen(filename, "w");
                if (!outfile) {
                    fprintf(stderr, "Error opening file %s for writing.\n", filename);
                    exit(EXIT_FAILURE);
                }
                fprintf(outfile, "Node ID,Order\n");
                for (int i = 0; i < ordering->size; i++) {
                    int node_id = ordering->ordering[i];
                    int64_t original_id = graph->node_map[node_id].original_id;
                    fprintf(outfile, "%lld,%d\n", original_id, i);
                }
                fclose(outfile);

                improved = 1;
                //break; // Exit permutations loop
            } 
	    {
                // Revert positions
                for (int i = 0; i < 4; i++) {
                    int node = selected_nodes[i];
                    int pos = original_positions[i];
                    ordering->position[node] = pos;
                    ordering->ordering[pos] = node;
                }
            }
        }

        free(edges_to_check);

        iteration++;
        if (iteration % progress_interval == 0) {
            printf("Progress: %d iterations completed.\n", iteration);
            fflush(stdout);
        }

        if (improved) {
            // Reset iteration count after improvement
            //iteration = 0;
        }
    }

    // Free adjacency lists
    for (int i = 0; i < node_count; i++) {
        free(node_edges[i]);
    }
    free(node_edges);
    free(node_edge_counts);

    printf("Exiting improve_solution.\n");
    fflush(stdout);
}
