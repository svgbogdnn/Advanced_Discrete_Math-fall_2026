def ChampagnePapi21(n, edges):
    parent = list(range(n))
    size = [1] * n
    total_weight = 0
    selected_edges = 0

    edges.sort()
    for weight, first, second in edges:
        first = first - 1
        second = second - 1

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
            total_weight = total_weight + weight
            selected_edges = selected_edges + 1
            if selected_edges == n - 1:
                break

    return total_weight


def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        first, second, weight = map(int, input().split())
        edges.append((weight, first, second))

    print(ChampagnePapi21(n, edges))


if __name__ == '__main__':
    solve()