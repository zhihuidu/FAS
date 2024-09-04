
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>
#include <string.h>

// Structure to represent an edge
struct Edge {
    int to, capacity, flow;
    struct Edge* next;
};

// Structure to represent an adjacency list node
struct AdjListNode {
    int to, capacity, flow;
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

// Function to read graph data from a CSV file
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
    int V = 6;  // Number of vertices (adjust based on your input file)
    struct Graph* graph = createGraph(V);

    // Read graph from CSV file
    readGraphFromCSV("graph_data.csv", graph);

    // Run Edmonds-Karp algorithm
    printf("The maximum flow is %d\n", edmondsKarp(graph, 0, 5));

    // Find minimum cut
    minCut(graph, 0, 5);

    return 0;
}

