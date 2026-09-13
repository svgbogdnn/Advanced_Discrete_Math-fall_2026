def ChampagnePapi21(matrix):
    roads = 0
    n = len(matrix)
    i = 0

    while i < n:
        j = i + 1

        while j < n:
            if matrix[i][j] == 1:
                roads = roads + 1
            j = j + 1

        i = i + 1

    return roads


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

    answer = ChampagnePapi21(matrix)
    print(answer)


if __name__ == '__main__':
    solve()
