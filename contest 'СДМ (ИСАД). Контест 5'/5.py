def ChampagnePapi21(n, edges):
    parent = list(range(n))
    size = [1] * n
    ordered_edges = []
    for index in range(len(edges)):
        first, second, weight = edges[index]
        ordered_edges.append((weight, first, second, index))
    ordered_edges.sort()

    chosen = [False] * len(edges)
    tree = []
    for _ in range(n):
        tree.append([])
    minimum_weight = 0
    chosen_count = 0

    for weight, first, second, index in ordered_edges:
        root_first = first
        while parent[root_first] != root_first:
            parent[root_first] = parent[parent[root_first]]
            root_first = parent[root_first]

        root_second = second
        while parent[root_second] != root_second:
            parent[root_second] = parent[parent[root_second]]
            root_second = parent[root_second]

        if root_first != root_second:
            if size[root_first] < size[root_second]:
                temporary_root = root_first
                root_first = root_second
                root_second = temporary_root
            parent[root_second] = root_first
            size[root_first] = size[root_first] + size[root_second]
            chosen[index] = True
            minimum_weight = minimum_weight + weight
            chosen_count = chosen_count + 1
            tree[first].append((second, weight))
            tree[second].append((first, weight))
            if chosen_count == n - 1:
                break

    second_weight = None
    for weight, first, second, index in ordered_edges:
        if chosen[index]:
            continue

        stack = [(first, -1, 0)]
        path_maximum = 0
        while stack:
            node, previous, current_maximum = stack.pop()
            if node == second:
                path_maximum = current_maximum
                break
            for neighbor, edge_weight in tree[node]:
                if neighbor != previous:
                    next_maximum = current_maximum
                    if edge_weight > next_maximum:
                        next_maximum = edge_weight
                    stack.append((neighbor, node, next_maximum))

        candidate_weight = minimum_weight + weight - path_maximum
        if second_weight is None or candidate_weight < second_weight:
            second_weight = candidate_weight

    return minimum_weight, second_weight


def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        first, second, weight = map(int, input().split())
        edges.append((first - 1, second - 1, weight))

    first_weight, second_weight = ChampagnePapi21(n, edges)
    print(first_weight, second_weight)


if __name__ == '__main__':
    solve()