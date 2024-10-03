#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <pthread.h>

#define MAX_PERMUTATIONS 24

typedef struct {
    size_t source;
    size_t target;
    double weight;
} Edge;

typedef struct {
    int64_t original_id;
    size_t mapped_id;
} NodeMap;

typedef struct {
    size_t node_count;
    size_t edge_count;
    Edge *edges;
    NodeMap *node_map;
} Graph;

typedef struct {
    size_t *ordering;    // mapping from position to node_id
    size_t *position;    // mapping from node_id to position
    size_t size;
} Ordering;

// Shared data structure for threads
typedef struct {
    size_t perm_index;
    int permutations[MAX_PERMUTATIONS][4];
    size_t perm_count;
    size_t selected_nodes[4];
    size_t original_positions[4];
    size_t *edges_to_check;
    size_t edges_to_check_count;
    Graph *graph;
    Ordering *ordering;
    double delta_score;
    int improved;
    pthread_mutex_t *mutex;
    pthread_cond_t *cond;
    int *improvement_found;
    double *current_score;
    size_t ordering_size;
} ThreadData;

// Function prototypes
void read_graph(const char *filename, Graph *graph);
void read_ordering(const char *filename, Ordering *ordering, Graph *graph);
double compute_initial_score(Graph *graph, Ordering *ordering);
void generate_permutations(size_t *nodes, size_t start, size_t end, int permutations[][4], size_t *count);
void improve_solution(Graph *graph, Ordering *ordering, double *current_score, int num_threads);
void *evaluate_permutation(void *arg);

int main(int argc, char *argv[]) {
    Graph graph;
    Ordering ordering;
    double current_score;

    // Initialize random seed
    srand(time(NULL));

    // Default number of threads
    int num_threads = 4;

    // Parse command-line arguments
    if (argc > 2) {
        fprintf(stderr, "Usage: %s [num_threads]\n", argv[0]);
        exit(EXIT_FAILURE);
    } else if (argc == 2) {
        num_threads = atoi(argv[1]);
        if (num_threads <= 0) {
            fprintf(stderr, "Number of threads must be a positive integer.\n");
            exit(EXIT_FAILURE);
        }
    }

    printf("Using %d thread(s).\n", num_threads);
    fflush(stdout);

    printf("Starting to read the graph.\n");
    fflush(stdout);
    // Read the graph
    read_graph("connectome_graph.csv", &graph);
    printf("Finished reading the graph.\n");
    fflush(stdout);

    // Print the size of the input graph
    printf("Graph has %zu vertices and %zu edges.\n", graph.node_count, graph.edge_count);
    fflush(stdout);

    printf("Starting to read the initial feasible solution.\n");
    fflush(stdout);
    // Read the initial feasible solution
    read_ordering("best.csv", &ordering, &graph);
    printf("Finished reading the initial feasible solution.\n");
    fflush(stdout);

    // Compute the initial score
    current_score = compute_initial_score(&graph, &ordering);
    printf("Initial score: %.6f\n", current_score);
    fflush(stdout);

    // Start improving the solution
    printf("Starting to improve the solution.\n");
    fflush(stdout);
    improve_solution(&graph, &ordering, &current_score, num_threads);

    // Free allocated memory
    free(graph.edges);
    free(graph.node_map);
    free(ordering.ordering);
    free(ordering.position);

    printf("Program completed successfully.\n");
    fflush(stdout);
    return 0;
}

// Function to read the graph from a CSV file
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
        fclose(file);
        exit(EXIT_FAILURE);
    }

    size_t edge_capacity = 10000;
    size_t node_capacity = 10000;
    size_t edge_count = 0;
    size_t node_id_count = 0;

    Edge *edges = malloc(edge_capacity * sizeof(Edge));
    NodeMap *node_map = malloc(node_capacity * sizeof(NodeMap));
    int64_t *node_ids = malloc(node_capacity * sizeof(int64_t));

    if (!edges || !node_map || !node_ids) {
        fprintf(stderr, "Memory allocation failed while initializing graph structures.\n");
        fclose(file);
        free(line);
        exit(EXIT_FAILURE);
    }

    // Skip header
    if (fgets(line, 8192, file) == NULL) {
        fprintf(stderr, "Graph file is empty or header is missing.\n");
        fclose(file);
        free(line);
        exit(EXIT_FAILURE);
    }

    while (fgets(line, 8192, file)) {
        int64_t source_id, target_id;
        double weight;
        char *token;
        char *endptr;

        // Remove newline character
        line[strcspn(line, "\r\n")] = 0;

        // Skip empty lines
        if (strlen(line) == 0) {
            continue;
        }

        // Parse source ID
        token = strtok(line, ",");
        if (!token) {
            fprintf(stderr, "Malformed line in graph file (missing source ID): %s\n", line);
            continue;
        }
        source_id = strtoll(token, &endptr, 10);
        if (*endptr != '\0') {
            fprintf(stderr, "Invalid source ID in graph file: %s\n", token);
            continue;
        }

        // Parse target ID
        token = strtok(NULL, ",");
        if (!token) {
            fprintf(stderr, "Malformed line in graph file (missing target ID): %s\n", line);
            continue;
        }
        target_id = strtoll(token, &endptr, 10);
        if (*endptr != '\0') {
            fprintf(stderr, "Invalid target ID in graph file: %s\n", token);
            continue;
        }

        // Parse weight
        token = strtok(NULL, ",");
        if (!token) {
            fprintf(stderr, "Malformed line in graph file (missing weight): %s\n", line);
            continue;
        }
        weight = strtod(token, &endptr);
        if (*endptr != '\0') {
            fprintf(stderr, "Invalid weight in graph file: %s\n", token);
            continue;
        }

        // Map source_id and target_id to indices
        size_t source_mapped = SIZE_MAX;
        size_t target_mapped = SIZE_MAX;

        // Use a linear search here; for large graphs, consider using a hash map
        for (size_t i = 0; i < node_id_count; i++) {
            if (node_ids[i] == source_id) {
                source_mapped = i;
            }
            if (node_ids[i] == target_id) {
                target_mapped = i;
            }
            if (source_mapped != SIZE_MAX && target_mapped != SIZE_MAX) {
                break;
            }
        }

        if (source_mapped == SIZE_MAX) {
            if (node_id_count == node_capacity) {
                node_capacity *= 2;
                node_ids = realloc(node_ids, node_capacity * sizeof(int64_t));
                node_map = realloc(node_map, node_capacity * sizeof(NodeMap));
                if (!node_ids || !node_map) {
                    fprintf(stderr, "Memory allocation failed while expanding node arrays.\n");
                    fclose(file);
                    free(line);
                    exit(EXIT_FAILURE);
                }
            }
            source_mapped = node_id_count;
            node_ids[node_id_count++] = source_id;
        }

        if (target_mapped == SIZE_MAX) {
            if (node_id_count == node_capacity) {
                node_capacity *= 2;
                node_ids = realloc(node_ids, node_capacity * sizeof(int64_t));
                node_map = realloc(node_map, node_capacity * sizeof(NodeMap));
                if (!node_ids || !node_map) {
                    fprintf(stderr, "Memory allocation failed while expanding node arrays.\n");
                    fclose(file);
                    free(line);
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
                fclose(file);
                free(line);
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
    for (size_t i = 0; i < node_id_count; i++) {
        node_map[i].original_id = node_ids[i];
        node_map[i].mapped_id = i; // size_t
    }

    graph->edges = edges;
    graph->edge_count = edge_count;
    graph->node_count = node_id_count;
    graph->node_map = node_map;

    free(node_ids);

    printf("Exiting read_graph.\n");
    fflush(stdout);
}

// Function to read the initial ordering from a CSV file
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
        fclose(file);
        exit(EXIT_FAILURE);
    }

    size_t size = graph->node_count;
    size_t *order = malloc(size * sizeof(size_t));
    size_t *position = malloc(size * sizeof(size_t));

    if (!order || !position) {
        fprintf(stderr, "Memory allocation failed while initializing ordering structures.\n");
        fclose(file);
        free(line);
        exit(EXIT_FAILURE);
    }

    // Initialize order and position arrays
    for (size_t i = 0; i < size; i++) {
        order[i] = SIZE_MAX;
        position[i] = SIZE_MAX;
    }

    // Skip header
    if (fgets(line, 8192, file) == NULL) {
        fprintf(stderr, "Ordering file is empty or header is missing.\n");
        fclose(file);
        free(line);
        exit(EXIT_FAILURE);
    }

    while (fgets(line, 8192, file)) {
        int64_t node_id;
        size_t pos;
        char *token;
        char *endptr;

        // Remove newline character
        line[strcspn(line, "\r\n")] = 0;

        // Skip empty lines
        if (strlen(line) == 0) {
            continue;
        }

        token = strtok(line, ",");
        if (!token) {
            fprintf(stderr, "Malformed line in ordering file (missing node ID): %s\n", line);
            continue;
        }
        node_id = strtoll(token, &endptr, 10);
        if (*endptr != '\0') {
            fprintf(stderr, "Invalid node ID in ordering file: %s\n", token);
            continue;
        }

        token = strtok(NULL, ",");
        if (!token) {
            fprintf(stderr, "Malformed line in ordering file (missing position): %s\n", line);
            continue;
        }
        pos = strtoull(token, &endptr, 10);
        if (*endptr != '\0') {
            fprintf(stderr, "Invalid position in ordering file: %s\n", token);
            continue;
        }

        // Map node_id to mapped_id
        size_t mapped_id = SIZE_MAX;
        for (size_t i = 0; i < graph->node_count; i++) {
            if (graph->node_map[i].original_id == node_id) {
                mapped_id = graph->node_map[i].mapped_id;
                break;
            }
        }

        if (mapped_id == SIZE_MAX) {
            fprintf(stderr, "Node ID %lld in ordering not found in graph.\n", node_id);
            fclose(file);
            free(line);
            exit(EXIT_FAILURE);
        }

        if (pos >= size) {
            fprintf(stderr, "Invalid position %zu for node ID %lld.\n", pos, node_id);
            fclose(file);
            free(line);
            exit(EXIT_FAILURE);
        }

        if (order[pos] != SIZE_MAX) {
            fprintf(stderr, "Duplicate position %zu for node ID %lld.\n", pos, node_id);
            fclose(file);
            free(line);
            exit(EXIT_FAILURE);
        }

        order[pos] = mapped_id;
        position[mapped_id] = pos;
    }

    fclose(file);
    free(line);

    // Check if all positions are filled
    for (size_t i = 0; i < size; i++) {
        if (order[i] == SIZE_MAX) {
            fprintf(stderr, "Position %zu is not assigned in the ordering.\n", i);
            exit(EXIT_FAILURE);
        }
    }

    ordering->ordering = order;
    ordering->position = position;
    ordering->size = size;

    printf("Exiting read_ordering.\n");
    fflush(stdout);
}

// Function to compute the initial score
double compute_initial_score(Graph *graph, Ordering *ordering) {
    printf("Computing initial score.\n");
    fflush(stdout);

    double score = 0.0;
    for (size_t i = 0; i < graph->edge_count; i++) {
        size_t u = graph->edges[i].source;
        size_t v = graph->edges[i].target;
        double w = graph->edges[i].weight;
        if (ordering->position[u] < ordering->position[v]) {
            score += w;
        }
    }

    printf("Initial score computed.\n");
    fflush(stdout);
    return score;
}

// Function to generate all permutations of the selected nodes
void generate_permutations(size_t *nodes, size_t start, size_t end, int permutations[][4], size_t *count) {
    if (start == end) {
        for (size_t i = 0; i <= end; i++) {
            permutations[*count][i] = nodes[i];
        }
        (*count)++;
    } else {
        for (size_t i = start; i <= end; i++) {
            // Swap
            size_t temp = nodes[start];
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

// Thread function to evaluate a single permutation
void *evaluate_permutation(void *arg) {
    ThreadData *data = (ThreadData *)arg;

    while (1) {
        size_t perm_index;

        // Protect access to perm_index
        pthread_mutex_lock(data->mutex);
        if (*(data->improvement_found)) {
            pthread_mutex_unlock(data->mutex);
            break;
        }
        perm_index = data->perm_index++;
        pthread_mutex_unlock(data->mutex);

        if (perm_index >= data->perm_count) {
            break;
        }

        int *perm = data->permutations[perm_index];
        double delta_score = 0.0;

        // Apply permutation to positions (thread-local copy)
        size_t *temp_positions = malloc(data->ordering_size * sizeof(size_t));
        size_t *temp_ordering = malloc(data->ordering_size * sizeof(size_t));
        if (!temp_positions || !temp_ordering) {
            fprintf(stderr, "Thread memory allocation failed.\n");
            exit(EXIT_FAILURE);
        }

        memcpy(temp_positions, data->ordering->position, data->ordering_size * sizeof(size_t));
        memcpy(temp_ordering, data->ordering->ordering, data->ordering_size * sizeof(size_t));

        for (int i = 0; i < 4; i++) {
            size_t node = perm[i];
            size_t pos = data->original_positions[i];
            temp_positions[node] = pos;
            temp_ordering[pos] = node;
        }

        // Calculate delta score
        for (size_t i = 0; i < data->edges_to_check_count; i++) {
            size_t edge_index = data->edges_to_check[i];
            size_t u = data->graph->edges[edge_index].source;
            size_t v = data->graph->edges[edge_index].target;
            double w = data->graph->edges[edge_index].weight;

            size_t pos_u_old = data->ordering->position[u];
            size_t pos_v_old = data->ordering->position[v];

            int contributes_old = pos_u_old < pos_v_old ? 1 : 0;

            size_t pos_u_new = temp_positions[u];
            size_t pos_v_new = temp_positions[v];

            int contributes_new = pos_u_new < pos_v_new ? 1 : 0;

            delta_score += (contributes_new - contributes_old) * w;
        }

        // Check if improvement is found
        pthread_mutex_lock(data->mutex);
        if (delta_score > 0 && !*(data->improvement_found)) {
            // Update shared ordering and current score
            *(data->improvement_found) = 1;
            data->delta_score = delta_score;

            memcpy(data->ordering->position, temp_positions, data->ordering_size * sizeof(size_t));
            memcpy(data->ordering->ordering, temp_ordering, data->ordering_size * sizeof(size_t));
            *(data->current_score) += delta_score;

            pthread_cond_broadcast(data->cond);
            pthread_mutex_unlock(data->mutex);

            free(temp_positions);
            free(temp_ordering);
            break;
        }
        pthread_mutex_unlock(data->mutex);

        free(temp_positions);
        free(temp_ordering);
    }

    return NULL;
}

// Function to improve the solution using multithreading
void improve_solution(Graph *graph, Ordering *ordering, double *current_score, int num_threads) {
    printf("Entering improve_solution.\n");
    fflush(stdout);

    size_t iteration = 0;
    size_t progress_interval = 10000;

    size_t node_count = graph->node_count;
    size_t edge_count = graph->edge_count;

    // Build adjacency lists for nodes
    size_t **node_edges = malloc(node_count * sizeof(size_t *));
    size_t *node_edge_counts = calloc(node_count, sizeof(size_t));
    if (!node_edges || !node_edge_counts) {
        fprintf(stderr, "Memory allocation failed while initializing adjacency lists.\n");
        exit(EXIT_FAILURE);
    }
    size_t *edge_counts = calloc(node_count, sizeof(size_t));
    if (!edge_counts) {
        fprintf(stderr, "Memory allocation failed while initializing edge counts.\n");
        exit(EXIT_FAILURE);
    }

    // First pass to count edges per node
    for (size_t i = 0; i < edge_count; i++) {
        size_t u = graph->edges[i].source;
        size_t v = graph->edges[i].target;
        edge_counts[u]++;
        edge_counts[v]++;
    }
    // Allocate memory
    for (size_t i = 0; i < node_count; i++) {
        node_edges[i] = malloc(edge_counts[i] * sizeof(size_t));
        if (!node_edges[i]) {
            fprintf(stderr, "Memory allocation failed for adjacency list of node %zu.\n", i);
            exit(EXIT_FAILURE);
        }
        node_edge_counts[i] = 0; // Reset for second pass
    }
    // Second pass to fill edge indices
    for (size_t i = 0; i < edge_count; i++) {
        size_t u = graph->edges[i].source;
        size_t v = graph->edges[i].target;
        node_edges[u][node_edge_counts[u]++] = i;
        node_edges[v][node_edge_counts[v]++] = i;
    }
    free(edge_counts);

    printf("Initialization of adjacency lists completed.\n");
    fflush(stdout);

    pthread_mutex_t mutex;
    pthread_cond_t cond;
    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&cond, NULL);

    // Main loop
    while (1) {
        // Randomly pick four distinct nodes
        size_t selected_nodes[4];
        size_t selected_count = 0;
        while (selected_count < 4) {
            size_t rand_node = rand() % node_count;
            int duplicate = 0;
            for (size_t i = 0; i < selected_count; i++) {
                if (selected_nodes[i] == rand_node) {
                    duplicate = 1;
                    break;
                }
            }
            if (!duplicate) {
                selected_nodes[selected_count++] = rand_node;
            }
        }

        // Generate all permutations of these four nodes
        int permutations[MAX_PERMUTATIONS][4];
        size_t perm_count = 0;
        size_t nodes_copy[4];
        memcpy(nodes_copy, selected_nodes, 4 * sizeof(size_t));
        generate_permutations(nodes_copy, 0, 3, permutations, &perm_count);

        // Collect all edges involving these nodes
        size_t *edges_to_check = NULL;
        size_t edges_to_check_count = 0;
        size_t edges_to_check_capacity = 0;
        for (size_t i = 0; i < 4; i++) {
            size_t node = selected_nodes[i];
            for (size_t j = 0; j < node_edge_counts[node]; j++) {
                size_t edge_index = node_edges[node][j];
                // Add edge_index to edges_to_check if not already added
                int already_added = 0;
                for (size_t k = 0; k < edges_to_check_count; k++) {
                    if (edges_to_check[k] == edge_index) {
                        already_added = 1;
                        break;
                    }
                }
                if (!already_added) {
                    if (edges_to_check_count == edges_to_check_capacity) {
                        edges_to_check_capacity = edges_to_check_capacity == 0 ? 10 : edges_to_check_capacity * 2;
                        edges_to_check = realloc(edges_to_check, edges_to_check_capacity * sizeof(size_t));
                        if (!edges_to_check) {
                            fprintf(stderr, "Memory allocation failed while expanding edges_to_check array.\n");
                            exit(EXIT_FAILURE);
                        }
                    }
                    edges_to_check[edges_to_check_count++] = edge_index;
                }
            }
        }

        // Save original positions
        size_t original_positions[4];
        for (size_t i = 0; i < 4; i++) {
            original_positions[i] = ordering->position[selected_nodes[i]];
        }

        // Thread data initialization
        ThreadData thread_data;
        thread_data.perm_index = 0;
        memcpy(thread_data.permutations, permutations, perm_count * sizeof(permutations[0]));
        thread_data.perm_count = perm_count;
        memcpy(thread_data.selected_nodes, selected_nodes, 4 * sizeof(size_t));
        memcpy(thread_data.original_positions, original_positions, 4 * sizeof(size_t));
        thread_data.edges_to_check = edges_to_check;
        thread_data.edges_to_check_count = edges_to_check_count;
        thread_data.graph = graph;
        thread_data.ordering = ordering;
        thread_data.delta_score = 0.0;
        thread_data.improved = 0;
        thread_data.mutex = &mutex;
        thread_data.cond = &cond;
        int improvement_found = 0;
        thread_data.improvement_found = &improvement_found;
        thread_data.current_score = current_score;
        thread_data.ordering_size = ordering->size;

        pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
        if (!threads) {
            fprintf(stderr, "Memory allocation failed for threads.\n");
            exit(EXIT_FAILURE);
        }

        // Create threads
        int actual_threads = num_threads < perm_count ? num_threads : perm_count;
        for (int i = 0; i < actual_threads; i++) {
            int rc = pthread_create(&threads[i], NULL, evaluate_permutation, (void *)&thread_data);
            if (rc) {
                fprintf(stderr, "Error creating thread %d\n", i);
                exit(EXIT_FAILURE);
            }
        }

        // Wait for threads to finish or an improvement to be found
        pthread_mutex_lock(&mutex);
        while (!improvement_found && thread_data.perm_index < perm_count) {
            pthread_cond_wait(&cond, &mutex);
        }
        pthread_mutex_unlock(&mutex);

        // Join threads
        for (int i = 0; i < actual_threads; i++) {
            pthread_join(threads[i], NULL);
        }

        free(threads);
        free(edges_to_check);

        if (improvement_found) {
            printf("Iteration %zu, New score: %.6f\n", iteration, *current_score);
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
            for (size_t i = 0; i < ordering->size; i++) {
                size_t node_id = ordering->ordering[i];
                int64_t original_id = graph->node_map[node_id].original_id;
                fprintf(outfile, "%lld,%zu\n", original_id, i);
            }
            fclose(outfile);

            // Reset iteration count after improvement
            iteration = 0;
        } else {
            // If no improvement was found, restore original positions
            for (size_t i = 0; i < 4; i++) {
                size_t node = selected_nodes[i];
                size_t pos = original_positions[i];
                ordering->position[node] = pos;
                ordering->ordering[pos] = node;
            }
        }

        iteration++;
        if (iteration % progress_interval == 0) {
            printf("Progress: %zu iterations completed.\n", iteration);
            fflush(stdout);
        }

        // For testing, add a termination condition
        /*
        if (iteration >= 100000) {
            printf("Reached maximum number of iterations. Exiting.\n");
            break;
        }
        */
    }

    // Free adjacency lists
    for (size_t i = 0; i < node_count; i++) {
        free(node_edges[i]);
    }
    free(node_edges);
    free(node_edge_counts);

    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);

    printf("Exiting improve_solution.\n");
    fflush(stdout);
}
