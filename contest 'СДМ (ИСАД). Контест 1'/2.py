def ChampagnePapi21(table):
    n = len(table)
    m = len(table[0])
    distance_limit = n + m
    distances = []

    for row in table:
        current_row = []

        for value in row:
            if value == 1:
                current_row.append(0)
            else:
                current_row.append(distance_limit)

        distances.append(current_row)

    i = 0

    while i < n:
        j = 0

        while j < m:
            if distances[i][j] != 0:
                best_distance = distance_limit

                if i > 0:
                    candidate = distances[i - 1][j] + 1

                    if candidate < best_distance:
                        best_distance = candidate

                if j > 0:
                    candidate = distances[i][j - 1] + 1

                    if candidate < best_distance:
                        best_distance = candidate

                distances[i][j] = best_distance

            j = j + 1

        i = i + 1

    i = n - 1

    while i >= 0:
        j = m - 1

        while j >= 0:
            if i < n - 1:
                candidate = distances[i + 1][j] + 1

                if candidate < distances[i][j]:
                    distances[i][j] = candidate

            if j < m - 1:
                candidate = distances[i][j + 1] + 1

                if candidate < distances[i][j]:
                    distances[i][j] = candidate

            j = j - 1

        i = i - 1

    return distances


def solve():
    first_line = input().split()
    n = int(first_line[0])
    m = int(first_line[1])
    table = []

    i = 0

    while i < n:
        row = input().split()
        current_row = []

        for value in row:
            current_row.append(int(value))

        table.append(current_row)
        i = i + 1

    answer = ChampagnePapi21(table)

    for row in answer:
        values = []

        for value in row:
            values.append(str(value))

        print(' '.join(values))


if __name__ == '__main__':
    solve()