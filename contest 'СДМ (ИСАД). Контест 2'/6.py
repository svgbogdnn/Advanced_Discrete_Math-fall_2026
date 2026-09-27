from collections import deque


def ChampagnePapi21(trains):
    n = len(trains)
    graph = []
    indegree = [0] * n
    i = 0

    while i < n:
        graph.append([])
        i = i + 1

    first = 0

    while first < n:
        second = first + 1

        while second < n:
            left = trains[first][0]

            if trains[second][0] > left:
                left = trains[second][0]

            right = trains[first][1]

            if trains[second][1] < right:
                right = trains[second][1]

            if left <= right:
                first_time = trains[first][2]
                first_time = first_time + (
                    left - trains[first][0]
                ) * trains[first][3]
                second_time = trains[second][2]
                second_time = second_time + (
                    left - trains[second][0]
                ) * trains[second][3]

                if first_time < second_time:
                    graph[first].append(second)
                    indegree[second] = indegree[second] + 1
                elif second_time < first_time:
                    graph[second].append(first)
                    indegree[first] = indegree[first] + 1

            second = second + 1

        first = first + 1

    queue = deque()
    vertex = 0

    while vertex < n:
        if indegree[vertex] == 0:
            queue.append(vertex)
        vertex = vertex + 1

    order = []

    while queue:
        vertex = queue.popleft()
        order.append(vertex)

        for neighbor in graph[vertex]:
            indegree[neighbor] = indegree[neighbor] - 1

            if indegree[neighbor] == 0:
                queue.append(neighbor)

    return order


def solve():
    n = int(input().strip())
    trains = []
    i = 0

    while i < n:
        values = input().split()
        train = []

        for value in values:
            train.append(int(value))

        trains.append(train)
        i = i + 1

    order = ChampagnePapi21(trains)
    answer = []

    for train in order:
        answer.append(str(train + 1))

    print(' '.join(answer))


if __name__ == '__main__':
    solve()