from collections import deque


def ChampagnePapi21(graph):
    n = len(graph)
    colors = [-1] * n
    start = 0

    while start < n:
        if colors[start] != -1:
            start = start + 1
            continue

        queue = deque()
        queue.append(start)
        colors[start] = 0

        while queue:
            vertex = queue.popleft()

            for neighbor in graph[vertex]:
                if colors[neighbor] == -1:
                    colors[neighbor] = 1 - colors[vertex]
                    queue.append(neighbor)
                elif colors[neighbor] == colors[vertex]:
                    return None

        start = start + 1

    first_table = []
    vertex = 0

    while vertex < n:
        if colors[vertex] == 0:
            first_table.append(vertex + 1)
        vertex = vertex + 1

    return first_table


def solve():
    first_line = input().split()
    n = int(first_line[0])
    m = int(first_line[1])
    graph = []
    i = 0

    while i < n:
        graph.append([])
        i = i + 1

    i = 0

    while i < m:
        edge = input().split()
        first = int(edge[0]) - 1
        second = int(edge[1]) - 1
        graph[first].append(second)
        graph[second].append(first)
        i = i + 1

    first_table = ChampagnePapi21(graph)

    if first_table is None:
        print('NO')
        return

    print('YES')
    answer = []

    for person in first_table:
        answer.append(str(person))

    print(' '.join(answer))


if __name__ == '__main__':
    solve()