import networkx as nx

# Union-Find data structure to manage components
class UnionFind:
    def __init__(self):
        self.parent = {}
    
    def find(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]
    
    def union(self, node1, node2):
        root1 = self.find(node1)
        root2 = self.find(node2)
        if root1 != root2:
            self.parent[root2] = root1
    
    def add(self, node):
        if node not in self.parent:
            self.parent[node] = node

def find_cycles_and_components(graph):
    # Find all simple cycles in the graph
    cycles = list(nx.simple_cycles(graph))
    
    uf = UnionFind()
    
    # Add nodes and edges to Union-Find structure
    for cycle in cycles:
        for node in cycle:
            uf.add(node)
        for i in range(len(cycle)):
            uf.union(cycle[i], cycle[(i + 1) % len(cycle)])
    
    # Group cycles by their components
    components = {}
    for cycle in cycles:
        root = uf.find(cycle[0])
        if root not in components:
            components[root] = []
        components[root].append(cycle)
    
    return components

# Example usage
if __name__ == "__main__":
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add edges to the graph
    edges = [
        (1, 2), (2, 3), (3, 1),  # First cycle
        (4, 5), (5, 6), (6, 4),  # Second cycle
        (7, 8), (8, 7)           # Third cycle
    ]
    G.add_edges_from(edges)
    
    # Find cycles and their components
    components = find_cycles_and_components(G)
    
    print(components)
    # Print the components and their cycles
    for component, cycles in components.items():
        print(f"Component rooted at {component}:")
        for cycle in cycles:
            print(f"  Cycle: {cycle}")

