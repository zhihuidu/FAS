#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <omp.h>
#include <stdatomic.h>
#include <sys/time.h>
#include <time.h>

#define MAX_NODES 136648
#define MAX_EDGES 5657719

int MAX_CYCLE_LENGTH=3;
int MIN_CYCLE_LENGTH=2;
int ALL_CYCLE_NUM=5;
int MAX_SEARCH_LEN=2000;

typedef struct {
    int source;
    int target;
    int weight;
} Edge;

typedef struct {
    long int source;
    long int target;
    int weight;
} LongEdge; //for storing original graph data

typedef struct {
    int* edges;  // Dynamic array of edge indices
    int count;   // Number of edges
    int capacity; // Capacity of the dynamic array
} EdgeInfo;// for in or out edges

typedef struct {
    int num_nodes;
    int num_edges;
    long int vertices[MAX_NODES];  // List of vertices
    LongEdge edges[MAX_EDGES];    // Static array of all edges
    EdgeInfo out_edge[MAX_NODES];  // Outgoing edges information
} Graph;

long int shared_weights[MAX_EDGES]={0};     //shared weight among different cycles

int total_weight=0;              // total weight of the graph

Edge mapped_edges[MAX_EDGES];    // Static array of all edges but vertices are mapped to 0-MAX_NODES-1

bool removed_edges[MAX_EDGES]={false};       //if the edge has been removed

int verbosity=0; //1 highly recommended; 2 recommended; >2 debugging info

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
    if (new_entry ==NULL) {
        printf("cannot allocate memory\n");
        exit(0);
    }
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

    }

}

// Function to find or add a vertex
int find_or_add_vertex(Graph* graph, long int vertex) {
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
        if (edge_info->edges ==NULL) {
              printf("cannot allocate memory\n");
              exit(0);
        }
    }
    edge_info->edges[edge_info->count++] = edge_index;
}

// Function to add an edge to the graph
void add_edge(Graph* graph, long int source, long int target, int weight) {
    int edge_index = graph->num_edges++;
    graph->edges[edge_index].source = source;
    graph->edges[edge_index].target = target;
    graph->edges[edge_index].weight = weight;

    int source_index = find_or_add_vertex(graph, source);
    int target_index = find_or_add_vertex(graph, target);

    mapped_edges[edge_index].source=source_index;        
    mapped_edges[edge_index].target=target_index;        
    mapped_edges[edge_index].weight=weight;        

    if (verbosity >5){
             printf("map %20ld,%20ld,%12d, to %10d,%10d,%12d\n",source,target,weight,source_index,target_index,weight);
    }

    // Update out_edge info
    add_edge_to_info(&graph->out_edge[source_index], edge_index);

}

// Function to read the graph from a CSV file
void read_graph_from_csv(const char* filename, Graph* graph,HashTable* table) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Cannot open file %s\n", filename);
        exit(1);
    }

    long int source, target;
    int weight;
    while (fscanf(file, "%ld,%ld,%d\n", &source, &target, &weight) != EOF) {
        if (verbosity >5) {
            printf("source=%ld,target=%ld,weight=%d\n",source,target,weight);
        }
        add_edge(graph, source, target, weight);
        total_weight+=weight;
    }
    printf("add all edges to the graph, total weight is %d\n\n",total_weight);

    for (int i = 0; i < graph->num_edges ; i++) {
         int source=mapped_edges[i].source;
         int target=mapped_edges[i].target;
         insert(table,source,target,i);
         if (verbosity >5){
             printf("map (%8d,%8d) to edge %7d\n",source,target,i);
         }
    }
    fclose(file);
}


// Function to find cycles up to a maximum length
void find_fix_cycles(Graph* graph, int start, int current, int length, int* path, bool* visited , long int * num_cycle, HashTable * table,int min_weight) {
    if (length > MAX_CYCLE_LENGTH) return;

    path[length - 1] = current;

    visited[current] = true;

    EdgeInfo* out_edges = &graph->out_edge[current];


    int thread_id=omp_get_thread_num();
    for (int i = 0; i < out_edges->count; i++) {
        int edge_index = out_edges->edges[i];
        Edge* edge = &mapped_edges[edge_index];
        int target_index = edge->target;
        if (edge->target == start && length >= MIN_CYCLE_LENGTH && removed_edges[edge_index] == false) {
            // Cycle detected
            if (edge->weight<min_weight) {
              min_weight=edge->weight;
            }
            atomic_fetch_add(&num_cycle[0],1);
            if (verbosity >2) {
                printf("Thread %3d, find %8d(th) cycle starting from vertex %7d with length %8d through %7d\n",thread_id,num_cycle[0], start,length,target_index);
            }
            // Update the weights in the shared_weights array
            for (int j = 0; j < length - 1; j++) {
                int u = path[j];
                int v = path[j + 1];
                int idx = get_edge_number(table,u,v);
                atomic_fetch_sub(&shared_weights[idx],min_weight);
                if (verbosity >4){
                    printf("Thread %3d, update %8d(th) cycle's (len=%5d) weight %6d to %12ld of edge (%6d,%6d) \n",thread_id,num_cycle[0],length, graph->edges[idx].weight,shared_weights[idx],u,v);
                }
            }
            atomic_fetch_sub(&shared_weights[edge_index],min_weight);
            if (verbosity >4){
                printf("Thread %3d, update %8d(th) cycle's (len=%5d) weight %6d to %12ld of edge (%6d,%6d) \n",thread_id,num_cycle[0], length,graph->edges[edge_index].weight,shared_weights[edge_index],current, target_index);
            }
        } else if ( !visited[edge->target] && edge->target > start && removed_edges[edge_index] == false) {
            if (edge->weight<min_weight) {
                 min_weight=edge->weight;
            }
            find_fix_cycles(graph, start, edge->target, length + 1, path, visited,num_cycle,table,min_weight);
        }
    }

    visited[current] = false;
}

// Function to find cycles up to a maximum length
void find_mix_cycles(Graph* graph, int start, int current, int length, int* path, bool* visited , long int * num_cycle, HashTable * table,int min_weight, long int * cur_large_cycles) {
    if (length > MAX_SEARCH_LEN || cur_large_cycles[0] >= ALL_CYCLE_NUM )  {
	    if (length >1) {
		    visited[path[length-1]]=false;
            }
            return;
    }

    path[length - 1] = current;

    visited[current] = true;

    EdgeInfo* out_edges = &graph->out_edge[current];


    int thread_id=omp_get_thread_num();
    for (int i = 0; i < out_edges->count; i++) {
        int edge_index = out_edges->edges[i];
        Edge* edge = &mapped_edges[edge_index];
        int target_index = edge->target;
        if (edge->target == start && length > MAX_CYCLE_LENGTH && removed_edges[edge_index] == false) {
            // Cycle detected
	    if (cur_large_cycles[0] >=ALL_CYCLE_NUM){
		    break;
	    }
            if (edge->weight<min_weight) {
                 min_weight=edge->weight;
            }
            atomic_fetch_add(&cur_large_cycles[0],1);
            if (verbosity >2) {
                printf("Thread %3d, find %8d(th) cycle starting from vertex %7d with length %8d through %7d\n",thread_id,num_cycle[0], start,length,target_index);
            }
            // Update the weights in the shared_weights array
            for (int j = 0; j < length - 1; j++) {
                int u = path[j];
                int v = path[j + 1];
                int idx = get_edge_number(table,u,v);
                atomic_fetch_sub(&shared_weights[idx],min_weight);
                if (verbosity >4){
                    printf("Thread %3d, update %5d(th) edge in the cycle (len=%5d) from weight %6d to %12ld of edge (%6d,%6d) \n",thread_id,j, length, graph->edges[idx].weight,shared_weights[idx],u,v);
                }
            }
            atomic_fetch_sub(&shared_weights[edge_index],min_weight);
            if (verbosity >4){
                printf("Thread %3d, update %5d(th) edge in the cycle (len=%5d) from weight %6d to %12ld of edge (%6d,%6d) \n",thread_id,length, length, graph->edges[edge_index].weight,shared_weights[edge_index],path[length-1],start);
            }
        } else if ( !visited[edge->target] && edge->target > start && removed_edges[edge_index] == false) {
            if (edge->weight<min_weight) {
              min_weight=edge->weight;
            }
            find_mix_cycles(graph, start, edge->target, length + 1, path, visited, num_cycle,table,min_weight,cur_large_cycles);
        }
    }

    visited[current] = false;
}

// Function to mark removed edges
void mark_fix_removed_edges(Graph* graph, int start, int current, int length, int* path, bool* visited,int * num_edges,HashTable * table,long int min_shared_weight,int min_shared_edge_index) {
    if (length > MAX_CYCLE_LENGTH) return;

    path[length - 1] = current;

    visited[current] = true;
    EdgeInfo* out_edges = &graph->out_edge[current];

    int thread_id=omp_get_thread_num();
    for (int i = 0; i < out_edges->count; i++) {
        int edge_index = out_edges->edges[i];
        Edge* edge = &mapped_edges[edge_index];
        int target_index = edge->target;
        long int longweight=edge->weight;

        if (edge->target == start && length >=MIN_CYCLE_LENGTH &&  removed_edges[edge_index]==false ) {
            // Cycle detected
            if (edge->weight+shared_weights[edge_index]<min_shared_weight) {
                  min_shared_weight=edge->weight+shared_weights[edge_index];
                  min_shared_edge_index=edge_index;
            }
            atomic_store(& removed_edges[min_shared_edge_index], true);
            atomic_fetch_add(&num_edges[0],1);
            if (verbosity >2){
                printf("Thread %3d, removed %8d(th) edge (%18ld,%18ld,%8d)\n",thread_id,num_edges[0], graph->edges[min_shared_edge_index].source,graph->edges[min_shared_edge_index].target,graph->edges[min_shared_edge_index].weight);
            }
        } else if (!visited[target_index] && edge->target > start && removed_edges[edge_index] == false ) {
            if (edge->weight+shared_weights[edge_index]<min_shared_weight) {
                  min_shared_weight=edge->weight+shared_weights[edge_index];
                  min_shared_edge_index=edge_index;
            }
            mark_fix_removed_edges(graph, start, edge->target, length + 1, path, visited, num_edges,table,min_shared_weight,min_shared_edge_index);
        }
    }

    visited[current] = false;
}




// Function to mark removed edges
void mark_mix_removed_edges(Graph* graph, int start, int current, int length, int* path, bool* visited,int * num_edges,HashTable * table,long int min_shared_weight,int min_shared_edge_index, long int* cur_large_cycles) {
    if (length > MAX_SEARCH_LEN ||cur_large_cycles[0] >= ALL_CYCLE_NUM )  {
	    if (length >1) {
		    visited[path[length-1]]=false;
            }
            return;
    }

    path[length - 1] = current;

    visited[current] = true;
    EdgeInfo* out_edges = &graph->out_edge[current];

    int thread_id=omp_get_thread_num();
    for (int i = 0; i < out_edges->count; i++) {
        int edge_index = out_edges->edges[i];
        Edge* edge = &mapped_edges[edge_index];
        int target_index = edge->target;
        long int longweight=edge->weight;

        if (edge->target == start && length >MAX_CYCLE_LENGTH &&  removed_edges[edge_index]==false ) {
            // Cycle detected
            if (cur_large_cycles[0] >= ALL_CYCLE_NUM ) {
                  break;
            }
            if (edge->weight+shared_weights[edge_index]<min_shared_weight) {
                  min_shared_weight=edge->weight+shared_weights[edge_index];
                  min_shared_edge_index=edge_index;
            }
            atomic_store(& removed_edges[min_shared_edge_index], true);
            atomic_fetch_add(&cur_large_cycles[0],1);
            atomic_fetch_add(&num_edges[0],1);
            if (verbosity >2){
                printf("Thread %3d, removed %8d(th) edge (%18ld,%18ld,%8d)\n",thread_id,num_edges[0], graph->edges[min_shared_edge_index].source,graph->edges[min_shared_edge_index].target,graph->edges[min_shared_edge_index].weight);
            }
        } else if (!visited[target_index] && edge->target > start && removed_edges[edge_index] == false ) {
            if (edge->weight+shared_weights[edge_index]<min_shared_weight) {
                  min_shared_weight=edge->weight+shared_weights[edge_index];
                  min_shared_edge_index=edge_index;
            }
            mark_mix_removed_edges(graph, start, edge->target, length + 1, path, visited, num_edges,table,min_shared_weight,min_shared_edge_index,cur_large_cycles);
        }
    }

    visited[current] = false;
}

int main(int argc, char * argv[]) {
    // run it like this ./ompdfs graphname [maxcyclelen verbersity mincyclelen numoflargecycles largestcyclelen]
    const char* input_filename = argv[1];
    if (argc>2) {
         MAX_CYCLE_LENGTH=atoi(argv[2]);
         if (argc >3) {
            verbosity=atoi(argv[3]);
         }
         if (argc >4) {
            MIN_CYCLE_LENGTH=atoi(argv[4]);
         }
         if (argc >5) {
            ALL_CYCLE_NUM=atoi(argv[5]);
         }
         if (argc >6) {
            MAX_SEARCH_LEN=atoi(argv[6]);
         }
    } 

    struct timeval start, end;
    long seconds, useconds;
    double elapsed;

    time_t now = time(NULL);
    struct tm *t = localtime(&now);

    // Create a filename with the timestamp
    char * output_filename="tmp-omp-removed-edges.csv";
    char loss_filename[100];
    strftime(loss_filename, sizeof(loss_filename), "edge-loss-%Y%m%d-%H%M%S.txt", t);
    snprintf(loss_filename + strlen(loss_filename), sizeof(loss_filename) - strlen(loss_filename), "-Min%d-Max%d.txt", MIN_CYCLE_LENGTH, MAX_CYCLE_LENGTH);
    Graph graph;
    printf("-------------------------------------------------------------\n");
    printf("OMP Subroutine started\n");
    printf("read file name=%s, writefile name=%s\n",input_filename,output_filename);

    gettimeofday(&start, NULL);

    initialize_graph(&graph);
    HashTable* table =create_table();
    read_graph_from_csv(input_filename, &graph,table);

    gettimeofday(&end, NULL);
    seconds  = end.tv_sec  - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;
    elapsed = seconds + useconds/1000000.0;
    printf("Initial graph and read data taken: %f seconds\n\n", elapsed);

    gettimeofday(&start, NULL);

    // Find cycles and update weights
    long int number_cycles=0;
    long int cur_large_cycles=0;
    #pragma omp parallel for reduction(+:number_cycles) schedule(dynamic)
    for (int i = 0; i < graph.num_nodes; i++) {
        bool visited[MAX_NODES] = {false};
        int path[MAX_CYCLE_LENGTH];
        int num_threads=omp_get_num_threads();
        int thread_id=omp_get_thread_num();

        find_fix_cycles(&graph, i, i, 1, path, visited, &number_cycles,table,INT_MAX);
    }

    printf("Find %15ld Cycles in the first search\n",number_cycles);
    if (number_cycles<ALL_CYCLE_NUM) {
	cur_large_cycles=number_cycles;
        #pragma omp parallel for reduction(+:cur_large_cycles) schedule(dynamic)
        for (int i = 0; i < graph.num_nodes; i++) {
            bool visited[MAX_NODES] = {false};
            int path[MAX_SEARCH_LEN];
            int num_threads=omp_get_num_threads();
            int thread_id=omp_get_thread_num();
            find_mix_cycles(&graph, i, i, 1, path, visited, &number_cycles,table,INT_MAX,&cur_large_cycles);
            if (verbosity >0 ){
                printf("Thread %4d of %4d, find %15ld(th) cycle's (%5d=<length<=%5d), %d large cycles\n",thread_id,num_threads,number_cycles, MIN_CYCLE_LENGTH,MAX_CYCLE_LENGTH,cur_large_cycles);
            }
	}
        printf("Find %15ld Cycles in the second search\n",cur_large_cycles);
    }



    gettimeofday(&end, NULL);
    seconds  = end.tv_sec  - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;
    elapsed = seconds + useconds/1000000.0;

    printf("Find %15ld Cycles, Time taken: %f seconds\n\n",(number_cycles>cur_large_cycles)? number_cycles:cur_large_cycles, elapsed);

    // Mark removed edges
    gettimeofday(&start, NULL);
    int number_edges=0;
    cur_large_cycles=0;
    #pragma omp parallel for reduction(+:number_edges) schedule(dynamic)
    for (int i = 0; i < graph.num_nodes; i++) {
        bool visited[MAX_NODES] = {false};
        int path[MAX_CYCLE_LENGTH];
        mark_fix_removed_edges(&graph, i, i, 1, path, visited,&number_edges,table,INT_MAX,0);
    }
    printf("Mark %d removed edges in the first search\n",number_edges);
    if (number_cycles<ALL_CYCLE_NUM) {
	cur_large_cycles=number_cycles;
        #pragma omp parallel for reduction(+:number_edges,cur_large_cycles) schedule(dynamic)
        for (int i = 0; i < graph.num_nodes; i++) {
            bool visited[MAX_NODES] = {false};
            int path[MAX_SEARCH_LEN];
            int num_threads=omp_get_num_threads();
            int thread_id=omp_get_thread_num();
            mark_mix_removed_edges(&graph, i, i, 1, path, visited,&number_edges,table,INT_MAX,0,&cur_large_cycles);
	}
        printf("Mark %d removed edges in the second search\n",number_edges);
    }
    gettimeofday(&end, NULL);
    seconds  = end.tv_sec  - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;
    elapsed = seconds + useconds/1000000.0;
    printf("Mark %d removed edges, Time taken: %f seconds\n\n",number_edges, elapsed);
    // Output removed edges to a file
    FILE* output_file = fopen(output_filename, "w");
    if (output_file == NULL) {
        printf("Error: Cannot open file %s for writing\n", output_filename);
        exit(1);
    }
    int removed_edge_num=0;
    int removed_edge_weight=0;
    for (int i = 0; i < graph.num_edges; i++) {
        if (removed_edges[i] == true) {
            fprintf(output_file, "%ld,%ld,%d\n", graph.edges[i].source, graph.edges[i].target,graph.edges[i].weight);
            removed_edge_num+=1;
            removed_edge_weight+=graph.edges[i].weight;
        }
    }
    fclose(output_file);
    printf("OMP subprogram Removed %d edges with weight sum %d\n", removed_edge_num,removed_edge_weight);
    printf("-------------------------------------------------------------\n");

    // Free dynamically allocated memory
    for (int i = 0; i < graph.num_nodes; i++) {
        free(graph.out_edge[i].edges);
        //free(graph.in_edge[i].edges);
    }
    free_table(table);

    return 0;
}


