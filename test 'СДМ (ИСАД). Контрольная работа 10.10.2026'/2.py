def depth(n, graph):
    stack = [1]
    visited = [False] * (n+1)
    visited[1] = True

    while stack:
        v = stack.pop()

        for w in range(1, n+1):
            if graph[v][w] == 1 and not visited[w]:
                visited[w] = True
                stack.append(w)

    res = []

    for i in range(1, n+1):
        if visited[i]:
            res.append(i)

    return res

def solve():
    n, m = map(int, input().split())
    graph = []

    for i in range(n+1):
        graph.append(bytearray(n+1))

    for j in range(m):
        a, b = map(int, input().split())
        graph[a][b] = 1
        graph[b][a] = 1

    res = depth(n, graph)

    print(len(res))
    print(*(res))

if __name__ == '__main__':
    solve()




















