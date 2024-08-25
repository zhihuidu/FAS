#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <time.h>
#include <omp.h>
#include <stdatomic.h>
#include <sys/time.h>

#define MAX_NODES 136648
#define MAX_EDGES 5657719
int MAX_CYCLE_LENGTH=100;

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
    //EdgeInfo in_edge[MAX_NODES];   // Incoming edges information
} Graph;

int updated_weights[MAX_EDGES];
int total=0;
Edge mapped_edges[MAX_EDGES];    // Static array of all edges but vertices are mapped to 0-MAX_NODES-1
EdgeInfo mapped_out_edge[MAX_NODES];  // Outgoing edges information
bool removed_edges[MAX_EDGES];

int verbosity=0;

#define TABLE_SIZE 5657723  // Size of the hash table

// Define a structure for the pairs (u, v)
typedef struct {
    int u, v;
} Pair;

// Define a structure for the hash table entries
typedef struct HashEntry {
    Pair pair;
    int edge_number;
    struct HashEntry* next;  // For handling collisions (chaining)
} HashEntry;

// Define the hash table
typedef struct {
    HashEntry** entries;  // Array of pointers to HashEntry (chaining)
} HashTable;


unsigned int hash_function(int u, int v) {
    return (u * 31 + v) % TABLE_SIZE;  // Simple hash function combining u and v
}
HashTable* create_table() {
    HashTable* table = (HashTable*)malloc(sizeof(HashTable));
    table->entries = (HashEntry**)malloc(TABLE_SIZE * sizeof(HashEntry*));

    for (int i = 0; i < TABLE_SIZE; i++) {
        table->entries[i] = NULL;  // Initialize all entries to NULL
    }

    return table;
}
void insert(HashTable* table, int u, int v, int edge_number) {
    unsigned int index = hash_function(u, v);

    // Create a new entry for the pair
    HashEntry* new_entry = (HashEntry*)malloc(sizeof(HashEntry));
    new_entry->pair.u = u;
    new_entry->pair.v = v;
    new_entry->edge_number = edge_number;
    new_entry->next = NULL;

    // Insert the entry into the hash table (chaining)
    if (table->entries[index] == NULL) {
        table->entries[index] = new_entry;
    } else {
        // Handle collision with chaining
        HashEntry* current = table->entries[index];
        while (current->next != NULL) {
            current = current->next;
        }
        current->next = new_entry;
    }
}
int get_edge_number(HashTable* table, int u, int v) {
    unsigned int index = hash_function(u, v);
    HashEntry* entry = table->entries[index];

    // Search for the pair in the linked list at this index
    while (entry != NULL) {
        if (entry->pair.u == u && entry->pair.v == v) {
            return entry->edge_number;
        }
        entry = entry->next;
    }

    return -1;  // Return -1 if the pair is not found
}
void free_table(HashTable* table) {
    for (int i = 0; i < TABLE_SIZE; i++) {
        HashEntry* entry = table->entries[i];
        while (entry != NULL) {
            HashEntry* temp = entry;
            entry = entry->next;
            free(temp);
        }
    }
    free(table->entries);
    free(table);
}

// Function to initialize the graph
void initialize_graph(Graph* graph) {
    graph->num_nodes = 0;
    graph->num_edges = 0;

    for (int i = 0; i < MAX_NODES; i++) {
        graph->vertices[i] = -1;
        graph->out_edge[i].edges = NULL;
        graph->out_edge[i].count = 0;
        graph->out_edge[i].capacity = 0;

        //graph->in_edge[i].edges = NULL;
        //graph->in_edge[i].count = 0;
        //graph->in_edge[i].capacity = 0;
    }

    for (int i = 0; i < MAX_EDGES; i++) {
        //updated_weights[i] = 0;
        removed_edges[i] = false;
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

    int source_index = find_or_add_vertex(graph, source);
    int target_index = find_or_add_vertex(graph, target);
    mapped_edges[edge_index].source=source_index;        
    mapped_edges[edge_index].target=target_index;        
    mapped_edges[edge_index].weight=weight;        
    if (verbosity >1){
             printf("map %d,%d,%d, to %d,%d,%d\n",source,target,weight,source_index,target_index,weight);
    }
    updated_weights[edge_index] = weight;

    // Update out_edge info
    add_edge_to_info(&graph->out_edge[source_index], edge_index);

    // Update in_edge info
    //int target_index = find_or_add_vertex(graph, target);
    //add_edge_to_info(&graph->in_edge[target_index], edge_index);
}

// Function to remmap  an edge using relabelled vertices
void remap_vertex(Graph* graph) {
    for (int i = 0; i < graph->num_edges ; i++) {
         int source=graph->edges[i].source;
         int target=graph->edges[i].target;
         int weight=graph->edges[i].weight;
         int source_index = find_or_add_vertex(graph, graph->edges[i].source);
         int target_index = find_or_add_vertex(graph, graph->edges[i].target);
         mapped_edges[i].source=source_index;        
         mapped_edges[i].target=target_index;        
         mapped_edges[i].weight=weight;        
         if (verbosity >1){
             printf("map %d,%d,%d, to %d,%d,%d\n",source,target,weight,source_index,target_index,weight);
         }
    }
}
// Function to read the graph from a CSV file
void read_graph_from_csv(const char* filename, Graph* graph,HashTable* table) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Cannot open file %s\n", filename);
        exit(1);
    }

    int source, target, weight;
    while (fscanf(file, "%d,%d,%d\n", &source, &target, &weight) != EOF) {
        //printf("source=%d,target=%d,weight=%d\n",source,target,weight);
        add_edge(graph, source, target, weight);
        total+=weight;
    }
    printf("add all edges to the graph\n\n");

    printf("remap vertices\n\n");
    for (int i = 0; i < graph->num_edges ; i++) {
         int source=mapped_edges[i].source;
         int target=mapped_edges[i].target;
         insert(table,source,target,i);
         if (verbosity >1){
             printf("map (%d,%d) to edge %d\n",source,target,i);
         }
    }
    printf("insert edges to hash table\n\n");
    fclose(file);
}


// Function to find cycles up to a maximum length
void find_cycles(Graph* graph, int start, int current, int length, int* path, bool* visited , int * num_cycle, HashTable * table,int min_weight) {
    if (length > MAX_CYCLE_LENGTH) return;

    path[length - 1] = current;

    visited[current] = true;
    EdgeInfo* out_edges = &graph->out_edge[current];


    int thread_id=omp_get_thread_num();
    for (int i = 0; i < out_edges->count; i++) {
        int edge_index = out_edges->edges[i];
        Edge* edge = &mapped_edges[edge_index];
        int target_index = edge->target;
        if (removed_edges[edge_index] == true) {
            continue;
        } 
        int weight=edge->weight;
        if (weight<min_weight) {
              min_weight=weight;
        }
        if (edge->target == start && length > 1) {
            // Cycle detected
            num_cycle[0]+=1;
            printf("Thread %d, find %d(th) cycle starting from vertex %d with length %d\n",thread_id,num_cycle[0], start,length);

            // Update the weights in the updated_weights array
            for (int j = 0; j < length - 1; j++) {
                int u = path[j];
                int v = path[j + 1];
                int idx = get_edge_number(table,u,v);
                atomic_fetch_sub(&updated_weights[idx],min_weight);
                if (verbosity >1){
                    printf("Thread %d, update %d(th) cycle's (length=%d) edge weight (%d,%d,%d) to (%d,%d,%d) \n",thread_id,num_cycle[0],length, u,v,graph->edges[idx].weight,u,v,updated_weights[idx]);
                }
            }
            updated_weights[edge_index] -= min_weight;
            if (verbosity >1){
                printf("Thread %d, update %d(th) cycle's (length=%d) edge weight (%d,%d,%d) to (%d,%d,%d) \n",thread_id,num_cycle[0], length, current,target_index,graph->edges[edge_index].weight,current, target_index, updated_weights[edge_index]);
            }
        } else if ( !visited[edge->target] && edge->target > start) {
            find_cycles(graph, start, edge->target, length + 1, path, visited,num_cycle,table,min_weight);
        }
    }

    visited[current] = false;
}

// Function to mark removed edges
void mark_removed_edges(Graph* graph, int start, int current, int length, int* path, bool* visited,int * num_edges,HashTable * table,int min_updated_weight,int min_updated_edge_index) {
    if (length > MAX_CYCLE_LENGTH) return;

    path[length - 1] = current;

    //int source_index = find_or_add_vertex(graph, current);
    visited[current] = true;
    EdgeInfo* out_edges = &graph->out_edge[current];

    int thread_id=omp_get_thread_num();
    for (int i = 0; i < out_edges->count; i++) {
        int edge_index = out_edges->edges[i];
        Edge* edge = &mapped_edges[edge_index];
        int target_index = edge->target;
        if (removed_edges[edge_index] == true) {
            continue;
        } 
        int weight=edge->weight;
        if (weight<min_updated_weight) {
              min_updated_weight=weight;
              min_updated_edge_index=edge_index;
        }

        if (edge->target == start && length > 1) {
            // Cycle detected

            atomic_store(& removed_edges[min_updated_edge_index], true);
            num_edges[0]+=1;
            printf("Thread %d, removed %d(th) edge (%d,%d,%d)\n",thread_id,num_edges[0], graph->edges[min_updated_edge_index].source,graph->edges[min_updated_edge_index].target,graph->edges[min_updated_edge_index].weight);
        } else if (!visited[target_index] && edge->target > start) {
            mark_removed_edges(graph, start, edge->target, length + 1, path, visited, num_edges,table,min_updated_weight,min_updated_edge_index);
        }
    }

    visited[current] = false;
}


int main(int argc, char * argv[]) {
    const char* input_filename = argv[1];
    //const char* input_filename = "graph.csv";
    //char* output_filename = "removed-edge.txt";
    if (argc>2) {
         MAX_CYCLE_LENGTH=atoi(argv[2]);
         if (argc >3) {
            verbosity=2;
         }
    } 
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

    printf("initialize graph\n");
    initialize_graph(&graph);
    printf("create hash table\n");
    HashTable* table =create_table();
    printf("read data\n");
    read_graph_from_csv(input_filename, &graph,table);
    printf("finish read data\n");

    gettimeofday(&end, NULL);
    seconds  = end.tv_sec  - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;
    elapsed = seconds + useconds/1000000.0;
    printf("Initial graph and read data taken: %f seconds\n\n", elapsed);

    gettimeofday(&start, NULL);

    // Find cycles and update weights
    int number_cycles=0;
    #pragma omp parallel for reduction(+:number_cycles) schedule(dynamic)
    for (int i = 0; i < graph.num_nodes; i++) {
        bool visited[MAX_NODES] = {0};
        int path[MAX_CYCLE_LENGTH];
        int num_threads=omp_get_num_threads();
        printf("Totally %d threads are executing the loop\n",num_threads);
        find_cycles(&graph, i, i, 1, path, visited, &number_cycles,table,INT_MAX);

    }

    gettimeofday(&end, NULL);
    seconds  = end.tv_sec  - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;
    elapsed = seconds + useconds/1000000.0;
    printf("Find %d Cycles with the largest length %d, Time taken: %f seconds\n\n",number_cycles, MAX_CYCLE_LENGTH, elapsed);

    // Mark removed edges
    gettimeofday(&start, NULL);
    int number_edges=0;
    #pragma omp parallel for reduction(+:number_edges) schedule(dynamic)
    for (int i = 0; i < graph.num_nodes; i++) {
        bool visited[MAX_NODES] = {0};
        int path[MAX_CYCLE_LENGTH];
        mark_removed_edges(&graph, i, i, 1, path, visited,&number_edges,table,INT_MAX,0);
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
    int removed_edge_num=0;
    int removed_edge_weight=0;
    for (int i = 0; i < graph.num_edges; i++) {
        if (removed_edges[i] == 1) {
            fprintf(output_file, "%d,%d,%d\n", graph.edges[i].source, graph.edges[i].target,graph.edges[i].weight);
            removed_edge_num+=1;
            removed_edge_weight+=graph.edges[i].weight;
        }
    }

    fclose(output_file);
    printf("Removed %d edges with weigt sum %d , gained percentage %f, have been successfully written to %s\n", removed_edge_num,removed_edge_weight, (total-removed_edge_weight)*1.0/total*100, output_filename);

    // Free dynamically allocated memory
    for (int i = 0; i < graph.num_nodes; i++) {
        free(graph.out_edge[i].edges);
        //free(graph.in_edge[i].edges);
    }
    free_table(table);

    return 0;
}


