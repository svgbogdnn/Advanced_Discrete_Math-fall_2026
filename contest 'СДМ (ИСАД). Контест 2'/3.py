def ChampagnePapi21(keys):
    n = len(keys)
    state = [0] * n
    cycles = 0
    start = 0

    while start < n:
        if state[start] != 0:
            start = start + 1
            continue

        marker = start + 1
        current = start

        while state[current] == 0:
            state[current] = marker
            current = keys[current]

        if state[current] == marker:
            cycles = cycles + 1

        current = start

        while state[current] == marker:
            next_vertex = keys[current]
            state[current] = -1
            current = next_vertex

        start = start + 1

    return cycles


def solve():
    n = int(input().strip())
    keys = []

    while len(keys) < n:
        values = input().split()

        for value in values:
            keys.append(int(value) - 1)

    answer = ChampagnePapi21(keys)
    print(answer)


if __name__ == '__main__':
    solve()