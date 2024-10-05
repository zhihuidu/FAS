#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>


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
    printf("Initial score: %d\n", current_score);
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
    //for (int i =0; i<ordering->size;i++){
    //	    printf("inner vertex %d's position is %d \n",i, ordering->position[i]);
    //}

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
	//printf("edge (%d,%d,%d)\n",u,v,w);
	//printf("order (%d,%d)\n",ordering->position[u],ordering->position[v]);

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
    int earlyexit=0;
    while(1) {
    #pragma omp parallel for schedule(dynamic)
    for (int paralli=0; paralli < 1000; paralli++) {
        int thread_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();
        int local_current_score= *current_score;
        //printf("thread id is %d, of %d, loop index %d\n",thread_id,total_threads,paralli);
        srand(time(NULL) + thread_id); // Seed each thread differently

	    
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
		//printf("select node %d\n",rand_node);
            }
        }

        // Generate all permutations of these four nodes
        int permutations[MAX_PERMUTATIONS][4];
        int perm_count = 0;
        int nodes_copy[4];
        memcpy(nodes_copy, selected_nodes, 4 * sizeof(int));
        generate_permutations(nodes_copy, 0, 3, permutations, &perm_count);

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
        int new_positions[4];
        for (int i = 0; i < 4; i++) {
            original_positions[i] = ordering->position[selected_nodes[i]];
        }

        // Try all permutations
        int improved = 0;
        for (int p = 0; p < MAX_PERMUTATIONS; p++) {
	    if (earlyexit==1) {
		    break;
	    }
            int *perm = permutations[p];
            int delta_score = 0;

            // Apply permutation to positions
            for (int i = 0; i < 4; i++) {
                int node = selected_nodes[i]; // The permuted node
                int pos = ordering->position[node];
		new_positions[i]=pos;
            }

            // Calculate delta score
            for (int i = 0; i < edges_to_check_count; i++) {
                int edge_index = edges_to_check[i];
                int u = graph->edges[edge_index].source;
                int v = graph->edges[edge_index].target;
                int w = graph->edges[edge_index].weight;

                int pos_u_new = -1;
                int pos_v_new = -1;
                int pos_u_old = -1;
                int pos_v_old = -1;

		for (int k=0;k<4;k++) {
			if (perm[k]==u) {
				pos_u_new=new_positions[k];
			}
			if (perm[k]==v) {
				pos_v_new=new_positions[k];
			}
			if (selected_nodes[k]==u) {
				pos_u_old=original_positions[k];
			}
			if (selected_nodes[k]==v) {
				pos_v_old=original_positions[k];
			}
		}
		if (pos_u_new==-1) {
                      pos_u_new = ordering->position[u];
		}
		if (pos_v_new==-1) {
                      pos_v_new = ordering->position[v];
		}
		if (pos_u_old==-1) {
                      pos_u_old = ordering->position[u];
		}
		if (pos_v_old==-1) {
                      pos_v_old = ordering->position[v];
		}

                int contributes_new = pos_u_new < pos_v_new ? 1 : 0;
                int contributes_old = pos_u_old < pos_v_old ? 1 : 0;

                delta_score += (contributes_new - contributes_old) * w;
            }

	    if (earlyexit==1) {
		    break;
	    }
            if (delta_score > 0) {
                // Update current score
                local_current_score += delta_score;
                printf("Iteration %d,  New score: %d, thread %d, loop index %d \n", iteration, local_current_score,thread_id, paralli);
                fflush(stdout);
                int writeflag=0;
                #pragma omp critical
                {
                      if (local_current_score  > *current_score) {
			     writeflag=1;
			     earlyexit=1;
			     *current_score= local_current_score;
			     for (int ii=0;ii<4;ii++){
			         ordering->position[perm[ii]]=new_positions[ii];
				 ordering->ordering[new_positions[ii]]=perm[ii];
			     }
                      }
                } 
		if (writeflag==1) {

                            printf("write new score: %d, thread %d, loop index %d \n", local_current_score,thread_id, paralli);
                             // Write new solution to CSV file
                            char filename[50];
                            sprintf(filename, "%d.csv", local_current_score);

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
                } 

                break; // Exit permutations loop
            } 
            local_current_score= *current_score;
        }
	earlyexit=0;

        local_current_score= *current_score;
        free(edges_to_check);

    }
        iteration++;
        if (iteration % progress_interval == 0) {
            printf("%d(th) iteration  * 1000 finished\n",iteration);
            fflush(stdout);
        }
        if (iteration >1000000) {
            iteration=0;
            printf("reset  iteration to 0 after 1000000 searches\n");
            fflush(stdout);
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




#define MAX_VERTICES 1000
#define MAX_EDGES 10000

typedef struct {
    int u, v;      // Edge from u to v
    double weight; // Edge weight
} Edge;

typedef struct {
    int vertex;    // Vertex number
    double weight; // Total weight of outgoing edges removed to break cycles
} FeedbackArcSet;

int n, m; // Number of vertices, edges
Edge edges[MAX_EDGES];   // Edges of the graph
double feedbackArcWeight = 0.0; // Weight of current feedback arc set
int ordering[MAX_VERTICES]; // Current vertex ordering
int inDAG[MAX_EDGES]; // 1 if edge is part of the DAG, 0 if in feedback arc set

void readGraph() {
    printf("Enter the number of vertices and edges: ");
    scanf("%d %d", &n, &m);
    
    printf("Enter the edges (u, v, weight):\n");
    for (int i = 0; i < m; i++) {
        scanf("%d %d %lf", &edges[i].u, &edges[i].v, &edges[i].weight);
    }

    printf("Enter the initial vertex order:\n");
    for (int i = 0; i < n; i++) {
        scanf("%d", &ordering[i]);
    }
}

// Find the feedback arc set based on the current order
void findFeedbackArcSet() {
    feedbackArcWeight = 0.0;
    for (int i = 0; i < m; i++) {
        int u_pos = -1, v_pos = -1;
        for (int j = 0; j < n; j++) {
            if (ordering[j] == edges[i].u) u_pos = j;
            if (ordering[j] == edges[i].v) v_pos = j;
        }

        // If the edge is backward in the current order, add it to the feedback arc set
        if (u_pos > v_pos) {
            inDAG[i] = 0;
            feedbackArcWeight += edges[i].weight;
        } else {
            inDAG[i] = 1; // Part of the DAG
        }
    }
}

// Topological sort using DFS
void dfsTopSort(int v, int *visited, int *topSort, int *pos, int adjMatrix[MAX_VERTICES][MAX_VERTICES]) {
    visited[v] = 1;
    for (int i = 0; i < n; i++) {
        if (adjMatrix[v][i] && !visited[i]) {
            dfsTopSort(i, visited, topSort, pos, adjMatrix);
        }
    }
    topSort[(*pos)--] = v;
}

// Create a new order using topological sort of the current DAG
void topologicalSort() {
    int adjMatrix[MAX_VERTICES][MAX_VERTICES] = {0};
    
    // Create adjacency matrix for the current DAG
    for (int i = 0; i < m; i++) {
        if (inDAG[i]) {
            adjMatrix[edges[i].u][edges[i].v] = 1;
        }
    }

    int visited[MAX_VERTICES] = {0};
    int topSort[MAX_VERTICES];
    int pos = n - 1;
    
    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            dfsTopSort(i, visited, topSort, &pos, adjMatrix);
        }
    }

    // Update ordering based on topological sort
    for (int i = 0; i < n; i++) {
        ordering[i] = topSort[i];
    }
}

void improveOrdering() {
    findFeedbackArcSet(); // Compute the feedback arc set initially

    double oldFeedbackWeight = feedbackArcWeight;
    
    while (1) {
        topologicalSort(); // Generate a new order using topological sort
        findFeedbackArcSet(); // Compute the feedback arc set for the new order

        // If we have improved, continue the process
        if (feedbackArcWeight < oldFeedbackWeight) {
            oldFeedbackWeight = feedbackArcWeight;
        } else {
            break; // Stop if no improvement
        }
    }

    printf("Final Feedback Arc Set Weight: %.2f\n", feedbackArcWeight);
}

int main() {
    readGraph();
    improveOrdering();
    
    return 0;
}





#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Structure to represent an edge in a directed, weighted graph
typedef struct {
    int source;
    int target;
    double weight;
} Edge;

// Graph representation: Adjacency list (with dynamic arrays)
typedef struct {
    int vertex_count;
    int edge_count;
    Edge* edges;  // Array of edges
} Graph;

// Function to add an edge to the graph
void add_edge(Graph *graph, int source, int target, double weight) {
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

// Function to calculate feedback arc set for given vertex ordering
double calculate_feedback_arc_set(Graph *graph, int* ordering) {
    double total_weight = 0.0;

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

// Main optimization routine
void optimize_feedback_arc_set(Graph *graph, int *ordering, double *current_score) {
    int max_iterations = 1000;
    int progress_interval = 10;
    int iteration = 0;

    while (iteration < max_iterations) {
        int improved = 0;
        double best_delta_score = 0.0;
        int best_permutation[4];

        // Select random nodes and permute
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < 10000; i++) {
            // (code for selecting and permuting nodes)

            double delta_score = calculate_feedback_arc_set(graph, ordering);

            #pragma omp critical
            {
                if (delta_score > best_delta_score) {
                    best_delta_score = delta_score;
                    // Store best permutation (if applicable)
                    improved = 1;
                }
            }
        }

        // Update the solution if improvement found
        if (improved) {
            *current_score += best_delta_score;
            printf("Iteration %d, New score: %.6f\n", iteration, *current_score);
        }

        iteration++;
        if (iteration % progress_interval == 0) {
            printf("Progress: %d iterations completed.\n", iteration);
        }
    }
}

// Main function
int main() {
    int vertex_count = 100000;  // Example vertex count
    int max_edges = 500000;     // Example edge count

    // Initialize graph
    Graph* graph = init_graph(vertex_count, max_edges);

    // Add edges to the graph (add_edge() function is used)

    int* ordering = (int*)malloc(vertex_count * sizeof(int));
    // Initialize vertex ordering
    for (int i = 0; i < vertex_count; i++) {
        ordering[i] = i;
    }

    double current_score = 0.0;

    // Perform optimization
    optimize_feedback_arc_set(graph, ordering, &current_score);

    // Clean up
    free(ordering);
    free_graph(graph);

    return 0;
}




#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// Structure to represent an edge in a directed, weighted graph
typedef struct {
    int source;
    int target;
    double weight;
} Edge;

// Graph representation: Adjacency list (using dynamic arrays)
typedef struct {
    int vertex_count;
    int edge_count;
    Edge* edges; // Array of edges
} Graph;

// Function to add an edge to the graph
void add_edge(Graph *graph, int source, int target, double weight) {
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
double calculate_feedback_arc_set(Graph *graph, int* ordering) {
    double total_weight = 0.0;

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
        int node = queue[front++];
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
void optimize_feedback_arc_set(Graph *graph, int *ordering, double *current_score) {
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
        double best_delta_score = 0.0;

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < 10000; i++) {
            // Implement heuristic search or optimization (e.g., select and permute nodes)

            double delta_score = calculate_feedback_arc_set(graph, ordering);

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
            printf("Iteration %d, New score: %.6f\n", iteration, *current_score);
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

// Main function
int main() {
    int vertex_count = 100000;  // Example vertex count
    int max_edges = 500000;     // Example edge count

    // Initialize graph
    Graph* graph = init_graph(vertex_count, max_edges);

    // Add edges to the graph (add_edge() function is used)

    int* ordering = (int*)malloc(vertex_count * sizeof(int));
    // Initialize vertex ordering
    for (int i = 0; i < vertex_count; i++) {
        ordering[i] = i;
    }

    double current_score = 0.0;

    // Perform optimization
    optimize_feedback_arc_set(graph, ordering, &current_score);

    // Clean up
    free(ordering);
    free_graph(graph);

    return 0;
}

