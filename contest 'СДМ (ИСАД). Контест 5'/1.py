def ChampagnePapi21(n, merges):
    node_count = 2 * n - 1
    dsu_parent = list(range(node_count))
    left_child = [-1] * node_count
    right_child = [-1] * node_count

    merge_index = 0
    for x, y in merges:
        root_x = x - 1
        while dsu_parent[root_x] != root_x:
            dsu_parent[root_x] = dsu_parent[dsu_parent[root_x]]
            root_x = dsu_parent[root_x]

        root_y = y - 1
        while dsu_parent[root_y] != root_y:
            dsu_parent[root_y] = dsu_parent[dsu_parent[root_y]]
            root_y = dsu_parent[root_y]

        new_node = n + merge_index
        left_child[new_node] = root_x
        right_child[new_node] = root_y
        dsu_parent[root_x] = new_node
        dsu_parent[root_y] = new_node
        dsu_parent[new_node] = new_node
        merge_index = merge_index + 1

    order = []
    stack = [node_count - 1]
    while stack:
        node = stack.pop()
        if node < n:
            order.append(node + 1)
        else:
            stack.append(right_child[node])
            stack.append(left_child[node])

    return order


def solve():
    n = int(input())
    merges = []
    for _ in range(n - 1):
        x, y = map(int, input().split())
        merges.append((x, y))

    answer = ChampagnePapi21(n, merges)
    output = []
    for cat in answer:
        output.append(str(cat))
    print(" ".join(output))


if __name__ == '__main__':
    solve()