from collections import deque


def ChampagnePapi21(matrix, start, target):
    n = len(matrix)
    distance = [-1] * n
    previous = [-1] * n
    queue = deque()
    queue.append(start)
    distance[start] = 0

    while queue:
        vertex = queue.popleft()

        if vertex == target:
            break

        neighbor = 0

        while neighbor < n:
            if matrix[vertex][neighbor] == 1:
                if distance[neighbor] == -1:
                    distance[neighbor] = distance[vertex] + 1
                    previous[neighbor] = vertex
                    queue.append(neighbor)
            neighbor = neighbor + 1

    if distance[target] == -1:
        return None

    path = []
    current = target

    while current != -1:
        path.append(current)

        if current == start:
            break

        current = previous[current]

    path.reverse()
    return path


def solve():
    n = int(input().strip())
    matrix = []
    i = 0

    while i < n:
        row = input().split()
        values = []

        for value in row:
            values.append(int(value))

        matrix.append(values)
        i = i + 1

    last_line = input().split()
    start = int(last_line[0]) - 1
    target = int(last_line[1]) - 1
    path = ChampagnePapi21(matrix, start, target)

    if path is None:
        print(-1)
        return

    print(len(path) - 1)

    if len(path) == 1:
        return

    answer = []

    for vertex in path:
        answer.append(str(vertex + 1))

    print(' '.join(answer))


if __name__ == '__main__':
    solve()