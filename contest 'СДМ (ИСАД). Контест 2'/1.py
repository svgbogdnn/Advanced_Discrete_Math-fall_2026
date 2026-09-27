def ChampagnePapi21(matrix):
    n = len(matrix)
    color = [0] * n
    parent = [-1] * n
    start = 0

    while start < n:
        if color[start] != 0:
            start = start + 1
            continue

        color[start] = 1
        stack_vertices = [start]
        next_neighbors = [0]

        while stack_vertices:
            vertex = stack_vertices[-1]
            neighbor = next_neighbors[-1]

            while neighbor < n and matrix[vertex][neighbor] == 0:
                neighbor = neighbor + 1

            if neighbor == n:
                color[vertex] = 2
                stack_vertices.pop()
                next_neighbors.pop()
                continue

            next_neighbors[-1] = neighbor + 1

            if neighbor == vertex:
                cycle = [vertex]
                return cycle

            if neighbor == parent[vertex]:
                continue

            if color[neighbor] == 0:
                parent[neighbor] = vertex
                color[neighbor] = 1
                stack_vertices.append(neighbor)
                next_neighbors.append(0)
            elif color[neighbor] == 1:
                cycle = [neighbor]
                current = vertex

                while current != neighbor:
                    cycle.append(current)
                    current = parent[current]

                return cycle

        start = start + 1

    cycle = []
    return cycle


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

    cycle = ChampagnePapi21(matrix)

    if not cycle:
        print('NO')
        return

    print('YES')
    print(len(cycle))
    answer = []

    for vertex in cycle:
        answer.append(str(vertex + 1))

    print(' '.join(answer))


if __name__ == '__main__':
    solve()