from collections import deque


def ChampagnePapi21(n, start, target):
    moves = [
        (-2, -1),
        (-2, 1),
        (-1, -2),
        (-1, 2),
        (1, -2),
        (1, 2),
        (2, -1),
        (2, 1),
    ]

    visited = []
    previous = []
    i = 0

    while i < n:
        visited.append([False] * n)
        previous.append([None] * n)
        i = i + 1

    queue = deque()
    queue.append(start)
    visited[start[0]][start[1]] = True

    while queue:
        current = queue.popleft()

        if current == target:
            break

        row = current[0]
        column = current[1]

        for move in moves:
            next_row = row + move[0]
            next_column = column + move[1]

            inside_board = 0 <= next_row < n and 0 <= next_column < n

            if inside_board and not visited[next_row][next_column]:
                visited[next_row][next_column] = True
                previous[next_row][next_column] = current
                queue.append((next_row, next_column))

    path = []
    current = target

    while True:
        path.append(current)

        if current == start:
            break

        current = previous[current[0]][current[1]]

    path.reverse()
    return path


def solve():
    numbers = input().split()

    while len(numbers) < 5:
        numbers.extend(input().split())

    values = []

    for number in numbers[:5]:
        values.append(int(number))

    n = values[0]
    start = (values[1] - 1, values[2] - 1)
    target = (values[3] - 1, values[4] - 1)
    path = ChampagnePapi21(n, start, target)

    print(len(path) - 1)

    for row, column in path:
        print(row + 1, column + 1)


if __name__ == '__main__':
    solve()