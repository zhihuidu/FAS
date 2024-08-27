def calculate_updated_weight(nodes, edge_weights, out_adj, edge_flag, updated_weights):
    def dfs_iterative(start_node):
        stack = [(start_node, start_node, [])]  # (first, current_node, path_stack)

        while stack:
            first, node, path_stack = stack.pop()
            path_stack.append(node)

            for neighbor, weight in out_adj[node]:
                if neighbor < first:
                    continue

                if neighbor == first:
                    if edge_flag[(node, neighbor)] == 0:
                        continue

                    cycle = path_stack[:] + [neighbor]
                    cycle_edges = []
                    skip = False

                    for i in range(len(cycle) - 1):
                        if edge_flag[(cycle[i], cycle[i + 1])] == 1:
                            cycle_edges.append((cycle[i], cycle[i + 1]))
                        else:
                            skip = True
                            break

                    if not skip:
                        cycle_weights = []
                        for u, v in cycle_edges:
                            cycle_weights.append((u, v, edge_weights[(u, v)]))
                        min_edge = min(cycle_weights, key=lambda x: x[2])
                        for u, v, w in cycle_weights:
                            updated_weights[(u, v)] -= min_edge[2]

                else:
                    stack.append((first, neighbor, path_stack[:]))

            path_stack.pop()

    for node in nodes:
        print(f"Update weight start from node {node} and we have {len(nodes)} nodes now")
        print(f"Enter DFS")
        dfs_iterative(node)

# Example usage of the function
# Assuming `nodes`, `edge_weights`, `out_adj`, `edge_flag`, and `updated_weights` are defined appropriately
# calculate_updated_weight(nodes, edge_weights, out_adj, edge_flag, updated_weights)

