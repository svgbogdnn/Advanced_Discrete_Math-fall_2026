from collections import deque


def ChampagnePapi21(graph):
    n = len(graph)
    visited = [False] * n
    components = []
    start = 0

    while start < n:
        if visited[start]:
            start = start + 1
            continue

        component = []
        queue = deque()
        queue.append(start)
        visited[start] = True

        while queue:
            vertex = queue.popleft()
            component.append(vertex)

            for neighbor in graph[vertex]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        components.append(component)
        start = start + 1

    return components


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

    components = ChampagnePapi21(graph)
    print(len(components))

    for component in components:
        print(len(component))
        answer = []

        for vertex in component:
            answer.append(str(vertex + 1))

        print(' '.join(answer))


if __name__ == '__main__':
    solve()