def ChampagnePapi21(keys):
    n = len(keys)
    incoming = [0] * n

    for destination in keys:
        incoming[destination] = incoming[destination] + 1

    queue = []
    for box in range(n):
        if incoming[box] == 0:
            queue.append(box)

    queue_index = 0
    while queue_index < len(queue):
        box = queue[queue_index]
        next_box = keys[box]
        incoming[next_box] = incoming[next_box] - 1
        if incoming[next_box] == 0:
            queue.append(next_box)
        queue_index = queue_index + 1

    visited = [False] * n
    cycle_count = 0
    for box in range(n):
        if incoming[box] > 0 and not visited[box]:
            cycle_count = cycle_count + 1
            current = box
            while not visited[current]:
                visited[current] = True
                current = keys[current]

    return cycle_count


def solve():
    n = int(input())
    keys = []
    for _ in range(n):
        box = int(input()) - 1
        keys.append(box)

    print(ChampagnePapi21(keys))


if __name__ == '__main__':
    solve()