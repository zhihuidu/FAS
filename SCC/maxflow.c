#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>
#include <string.h>

#include <stdbool.h>
#include <omp.h>
#include <stdatomic.h>
#include <sys/time.h>
#include <time.h>


int    verbosity=0;
time_t start_time,end_time;
int time_threshold=15;

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>
#include <string.h>

// Structure to represent an adjacency list node
struct AdjListNode {
    int to, capacity, flow, saved_flow;
    struct AdjListNode* next;
};

// Structure to represent an adjacency list
struct AdjList {
    struct AdjListNode* head;
};

// Structure to represent a graph
struct Graph {
    int V;
    struct AdjList* array;
};

// Function to create a new adjacency list node
struct AdjListNode* newAdjListNode(int to, int capacity) {
    struct AdjListNode* newNode = (struct AdjListNode*)malloc(sizeof(struct AdjListNode));
    newNode->to = to;
    newNode->capacity = capacity;
    newNode->flow = 0;
    newNode->saved_flow = 0;  // New field to save flow
    newNode->next = NULL;
    return newNode;
}

// Function to create a graph with V vertices
struct Graph* createGraph(int V) {
    struct Graph* graph = (struct Graph*)malloc(sizeof(struct Graph));
    graph->V = V;

    // Create an array of adjacency lists
    graph->array = (struct AdjList*)malloc(V * sizeof(struct AdjList));

    // Initialize each adjacency list as empty by making head NULL
    for (int i = 0; i < V; ++i) {
        graph->array[i].head = NULL;
    }

    return graph;
}


// Function to add an edge to the graph, avoiding duplicate reverse edges
void addEdge(struct Graph* graph, int from, int to, int capacity) {
    // Add the forward edge from `from` to `to`
    struct AdjListNode* newNode = newAdjListNode(to, capacity);
    newNode->next = graph->array[from].head;
    graph->array[from].head = newNode;

    // Check if the reverse edge (from `to` to `from`) already exists
    struct AdjListNode* reverseNode = graph->array[to].head;
    bool reverseExists = false;
    while (reverseNode != NULL) {
        if (reverseNode->to == from) {
            reverseExists = true;
            break;
        }
        reverseNode = reverseNode->next;
    }

    // Only add the reverse edge with 0 capacity if it doesn't already exist
    if (!reverseExists) {
        newNode = newAdjListNode(from, 0); // Reverse edge with 0 capacity
        newNode->next = graph->array[to].head;
        graph->array[to].head = newNode;
    }
}



// Utility function to perform BFS and find if there is a path from source to sink
bool bfs(struct Graph* graph, int s, int t, int parent[]) {
    bool visited[graph->V];
    memset(visited, 0, sizeof(visited));

    int queue[graph->V], front = 0, rear = 0;
    queue[rear++] = s;
    visited[s] = true;
    parent[s] = -1;

    while (front != rear) {
        int u = queue[front++];
        
        struct AdjListNode* node = graph->array[u].head;
        while (node != NULL) {
            int v = node->to;
            if (!visited[v] && node->capacity > node->flow) {
                queue[rear++] = v;
                parent[v] = u;
                visited[v] = true;
            }
            node = node->next;
        }
    }

    return visited[t];
}

// Function to update the flow in the residual graph
void updateFlow(struct Graph* graph, int s, int t, int parent[], int path_flow) {
    for (int v = t; v != s; v = parent[v]) {
        int u = parent[v];
        
        struct AdjListNode* node = graph->array[u].head;
        while (node != NULL) {
            if (node->to == v) {
                node->flow += path_flow;
                break;
            }
            node = node->next;
        }

        node = graph->array[v].head;
        while (node != NULL) {
            if (node->to == u) {
                node->flow -= path_flow;
                break;
            }
            node = node->next;
        }
    }
}

// Implementation of the Edmonds-Karp algorithm
int edmondsKarp(struct Graph* graph, int s, int t) {
    int max_flow = 0;
    int parent[graph->V];

    while (bfs(graph, s, t, parent)) {
        int path_flow = INT_MAX;

        for (int v = t; v != s; v = parent[v]) {
            int u = parent[v];
            
            struct AdjListNode* node = graph->array[u].head;
            while (node != NULL) {
                if (node->to == v) {
                    path_flow = (path_flow < node->capacity - node->flow) ? path_flow : node->capacity - node->flow;
                    break;
                }
                node = node->next;
            }
        }

        updateFlow(graph, s, t, parent, path_flow);
        max_flow += path_flow;
    }

    return max_flow;
}

// Function to reset all flows to 0 in the graph before running the second max flow
void resetFlows(struct Graph* graph) {
    for (int u = 0; u < graph->V; u++) {
        struct AdjListNode* node = graph->array[u].head;
        while (node != NULL) {
            node->flow = 0;
            node = node->next;
        }
    }
}

// Function to save the flow for future restoration
void saveFlows(struct Graph* graph) {
    for (int u = 0; u < graph->V; u++) {
        struct AdjListNode* node = graph->array[u].head;
        while (node != NULL) {
            node->saved_flow = node->flow;  // Save the current flow
            node = node->next;
        }
    }
}

// Function to restore the flow after computing the second max flow
void restoreFlows(struct Graph* graph) {
    for (int u = 0; u < graph->V; u++) {
        struct AdjListNode* node = graph->array[u].head;
        while (node != NULL) {
            node->flow = node->saved_flow;  // Restore the saved flow
            node = node->next;
        }
    }
}

// Function to find the minimum cut using the residual graph
void minCut(struct Graph* graph, int s, int t) {
    bool visited[graph->V];
    memset(visited, 0, sizeof(visited));

    int queue[graph->V], front = 0, rear = 0;
    queue[rear++] = s;
    visited[s] = true;

    while (front != rear) {
        int u = queue[front++];

        struct AdjListNode* node = graph->array[u].head;
        while (node != NULL) {
            int v = node->to;
            if (node->capacity > node->flow && !visited[v]) {
                queue[rear++] = v;
                visited[v] = true;
            }
            node = node->next;
        }
    }

    printf("The minimum cut edges are:\n");
    for (int u = 0; u < graph->V; u++) {
        struct AdjListNode* node = graph->array[u].head;
        while (node != NULL) {
            int v = node->to;
            if (visited[u] && !visited[v] && node->capacity > 0) {
                printf("%d - %d (Capacity: %d)\n", u, v, node->capacity);
            }
            node = node->next;
        }
    }
}

// Function to read graph data from a CSV file (the same as in the previous example)
void readGraphFromCSV(const char* filename, struct Graph* graph) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Unable to open the file %s\n", filename);
        return;
    }

    int from, to, capacity;
    while (fscanf(file, "%d,%d,%d", &from, &to, &capacity) == 3) {
        addEdge(graph, from, to, capacity);
    }

    fclose(file);
}

int main() {
    int V = 6;  // Number of vertices
    struct Graph* graph = createGraph(V);

    // Read graph from CSV file
    readGraphFromCSV("graph_data.csv", graph);

    // Run first max flow from source to destination
    int source = 0, destination = 5;
    int max_flow_1 = edmondsKarp(graph, source, destination);
    printf("Max flow from source to destination: %d\n", max_flow_1);

    // Save flows after first max flow
    saveFlows(graph);

    // Reset flows
    resetFlows(graph);

    // Run second max flow from destination to source
    int max_flow_2 = edmondsKarp(graph, destination, source);
    printf("Max flow from destination to source: %d\n", max_flow_2);

    // Choose the smaller flow and do the min cut
    if (max_flow_1 < max_flow_2) {
        printf("Min cut based on flow from source to destination:\n");
        restoreFlows(graph);  // Restore the flow to what it was after max_flow_1
        minCut(graph, source, destination);
    } else {
        printf("Min cut based on flow from destination to source:\n");
        minCut(graph, destination, source);
    }

    return 0;
}



// Structure to represent an adjacency list node
struct AdjListNode {
    int to, capacity, flow, saved_flow;
    struct AdjListNode* next;
};

// Structure to represent an adjacency list
struct AdjList {
    struct AdjListNode* head;
};

// Structure to represent a graph
struct Graph {
    int V;
    struct AdjList* array;
};

// Function to create a new adjacency list node
struct AdjListNode* newAdjListNode(int to, int capacity) {
    struct AdjListNode* newNode = (struct AdjListNode*)malloc(sizeof(struct AdjListNode));
    newNode->to = to;
    newNode->capacity = capacity;
    newNode->flow = 0;
    newNode->saved_flow = 0;  // New field to save flow
    newNode->next = NULL;
    return newNode;
}

// Function to create a graph with V vertices
struct Graph* createGraph(int V) {
    struct Graph* graph = (struct Graph*)malloc(sizeof(struct Graph));
    graph->V = V;

    // Create an array of adjacency lists
    graph->array = (struct AdjList*)malloc(V * sizeof(struct AdjList));

    // Initialize each adjacency list as empty by making head NULL
    for (int i = 0; i < V; ++i) {
        graph->array[i].head = NULL;
    }

    return graph;
}

// Function to add an edge to the graph
void addEdge(struct Graph* graph, int from, int to, int capacity) {
    // Add an edge from `from` to `to`
    struct AdjListNode* newNode = newAdjListNode(to, capacity);
    newNode->next = graph->array[from].head;
    graph->array[from].head = newNode;

    // Also add the reverse edge with 0 capacity (for residual graph)
    newNode = newAdjListNode(from, 0);
    newNode->next = graph->array[to].head;
    graph->array[to].head = newNode;
}


// A utility function to perform BFS and find if there is a path from source to sink
bool bfs(struct Graph* graph, int s, int t, int parent[]) {
    bool visited[graph->V];
    memset(visited, 0, sizeof(visited));

    int queue[graph->V], front = 0, rear = 0;
    queue[rear++] = s;
    visited[s] = true;
    parent[s] = -1;

    while (front != rear) {
        int u = queue[front++];
        
        struct AdjListNode* node = graph->array[u].head;
        while (node != NULL) {
            int v = node->to;
            if (!visited[v] && node->capacity > node->flow) {
                queue[rear++] = v;
                parent[v] = u;
                visited[v] = true;
            }
            node = node->next;
        }
    }

    return visited[t];
}

// Function to update the flow in the residual graph
void updateFlow(struct Graph* graph, int s, int t, int parent[], int path_flow) {
    for (int v = t; v != s; v = parent[v]) {
        int u = parent[v];
        
        struct AdjListNode* node = graph->array[u].head;
        while (node != NULL) {
            if (node->to == v) {
                node->flow += path_flow;
                break;
            }
            node = node->next;
        }

        node = graph->array[v].head;
        while (node != NULL) {
            if (node->to == u) {
                node->flow -= path_flow;
                break;
            }
            node = node->next;
        }
    }
}

// Implementation of the Edmonds-Karp algorithm
int edmondsKarp(struct Graph* graph, int s, int t) {
    int max_flow = 0;
    int parent[graph->V];

    while (bfs(graph, s, t, parent)) {
        int path_flow = INT_MAX;

        for (int v = t; v != s; v = parent[v]) {
            int u = parent[v];
            
            struct AdjListNode* node = graph->array[u].head;
            while (node != NULL) {
                if (node->to == v) {
                    path_flow = (path_flow < node->capacity - node->flow) ? path_flow : node->capacity - node->flow;
                    break;
                }
                node = node->next;
            }
        }

        updateFlow(graph, s, t, parent, path_flow);
        max_flow += path_flow;
    }

    return max_flow;
}



// Function to reset all flows to 0 in the graph before running the second max flow
void resetFlows(struct Graph* graph) {
    for (int u = 0; u < graph->V; u++) {
        struct AdjListNode* node = graph->array[u].head;
        while (node != NULL) {
            node->flow = 0;
            node = node->next;
        }
    }
}

// Function to save the flow for future restoration
void saveFlows(struct Graph* graph) {
    for (int u = 0; u < graph->V; u++) {
        struct AdjListNode* node = graph->array[u].head;
        while (node != NULL) {
            node->saved_flow = node->flow;  // Save the current flow
            node = node->next;
        }
    }
}

// Function to restore the flow after computing the second max flow
void restoreFlows(struct Graph* graph) {
    for (int u = 0; u < graph->V; u++) {
        struct AdjListNode* node = graph->array[u].head;
        while (node != NULL) {
            node->flow = node->saved_flow;  // Restore the saved flow
            node = node->next;
        }
    }
}






// Function to find the minimum cut using the residual graph
void minCut(struct Graph* graph, int s, int t,int maxflow,char * output_filename) {
    bool visited[graph->V];
    memset(visited, 0, sizeof(visited));

    int queue[graph->V], front = 0, rear = 0;
    queue[rear++] = s;
    visited[s] = true;

    while (front != rear) {
        int u = queue[front++];

        struct AdjListNode* node = graph->array[u].head;
        while (node != NULL) {
            int v = node->to;
            if (node->capacity > node->flow && !visited[v]) {
                queue[rear++] = v;
                visited[v] = true;
            }
            node = node->next;
        }
    }

    // Output removed edges to a file
    FILE* output_file = fopen(output_filename, "w");
    if (output_file == NULL) {
        printf("Error: Cannot open file %s for writing\n", output_filename);
        exit(1);
    }
    fprintf(output_file, "%d\n", maxflow);
    //printf("The minimum cut edges are:\n");
    for (int u = 0; u < graph->V; u++) {
        struct AdjListNode* node = graph->array[u].head;
        while (node != NULL) {
            int v = node->to;
            if (visited[u] && !visited[v] && node->capacity > 0) {
                //printf("%d - %d (Capacity: %d)\n", u, v, node->capacity);
                fprintf(output_file, "%d,%d,%d\n", u, v,node->capacity);
            }
            node = node->next;
        }
    }
    fclose(output_file);
}

// Function to read graph data from a CSV file
void readGraphFromCSV(const char* filename, struct Graph* graph) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Unable to open the file %s\n", filename);
        return;
    }

    int from, to, capacity;
    while (fscanf(file, "%d,%d,%d\n", &from, &to, &capacity) !=EOF) {
        addEdge(graph, from, to, capacity);
    }

    fclose(file);
}




int main(int argc, char * argv[]) {
    // run it like this ./ompdfs graphname [maxcyclelen verbersity mincyclelen numoflargecycles largestcyclelen]
    const char* input_filename = argv[1];
    int V = 6;  // Number of vertices (adjust based on your input file)
    char * endptr;
    int source_v,dest_v;
  
    if (argc>2) {
         V=atoi(argv[2]);
         if (argc >3) {
            verbosity=atoi(argv[3]);
         }
         if (argc >4) {
            source_v=atoi(argv[4]);
         }
         if (argc >5) {
            dest_v=atoi(argv[5]);
         }
         if (argc >6) {
            time_threshold=atoi(argv[6]);
         }
    } 

    struct timeval start, end;
    long seconds, useconds;
    double elapsed;

    time_t now = time(NULL);
    struct tm *t = localtime(&now);

    time(&start_time);
    // Create a filename with the timestamp
    char * output_filename="tmp-maxflow-removed-edges.csv";


    printf("-------------------------------------------------------------\n");
    printf("MaxFlow Subroutine started");
    printf("read file name=%s, writefile name=%s\n",input_filename,output_filename);

    struct Graph* graph = createGraph(V);

    // Read graph from CSV file
    readGraphFromCSV(input_filename, graph);


    time(&end_time);
    printf("read file takes  %d seconds\n", end_time-start_time);
    time(&start_time);
    // Run Edmonds-Karp algorithm
    int maxflow1= edmondsKarp(graph, source_v, dest_v);
    printf("The maximum flow 1 is %d\n", maxflow1);






    // Save flows after first max flow
    saveFlows(graph);

    // Reset flows
    resetFlows(graph);



    int maxflow2= edmondsKarp(graph, dest_v,source_v);
    printf("The maximum flow 2 is %d\n", maxflow2);
    if (maxflow2<maxflow1) {
        minCut(graph, dest_v,source_v, maxflow2, output_filename);
    } else {
        printf("Min cut based on flow from source to destination:\n");
        restoreFlows(graph);  
        minCut(graph, source_v, dest_v,maxflow1, output_filename);
    }
    time(&end_time);
    printf("The maximum flow takes  %d second\n", end_time-start_time);


    printf("-------------------------------------------------------------\n");
    return 0;
}


