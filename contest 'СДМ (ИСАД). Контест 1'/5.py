from collections import deque


def ChampagnePapi21(graph, start):
    distances = [-1] * len(graph)
    queue = deque()
    queue.append(start)
    distances[start] = 0
    farthest = start

    while queue:
        current = queue.popleft()

        if distances[current] > distances[farthest]:
            farthest = current

        for neighbor in graph[current]:
            if distances[neighbor] == -1:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)

    return farthest, distances[farthest]


def solve():
    n = int(input().strip())
    graph = []

    i = 0

    while i < n:
        graph.append([])
        i = i + 1

    i = 0

    while i < n - 1:
        edge = input().split()
        first = int(edge[0]) - 1
        second = int(edge[1]) - 1
        graph[first].append(second)
        graph[second].append(first)
        i = i + 1

    if n == 1:
        print(1)
        return

    first_end = ChampagnePapi21(graph, 0)[0]
    answer = ChampagnePapi21(graph, first_end)[1]
    answer = answer + 1
    print(answer)


if __name__ == '__main__':
    solve()