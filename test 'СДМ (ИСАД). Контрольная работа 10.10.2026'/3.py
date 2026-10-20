from collections import deque

def table():
    n, m = map(int, input().split())
    queue = deque()
    directions = [ (-1,0), (1, 0), (0, -1), (0, 1) ]
    distance = []

    for i in range(n):
        distance.append([-1] * m)

    for i in range(n):
        row = input().split()

        for j in range(m):
            if row[j] == '1':
                distance[i][j] = 0
                queue.append((i, j))

    while queue:
        x, y = queue.popleft()

        for dx, dy in directions:
            nx = x + dx
            ny = y + dy

            if (0 <= nx < n) and (0 <= ny < m):
                if distance[nx][ny] == -1:
                    distance[nx][ny] = distance[x][y] + 1
                    queue.append((nx, ny))

    for row in distance:
        print(row)

if __name__ == '__main__':
    table()