#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

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

