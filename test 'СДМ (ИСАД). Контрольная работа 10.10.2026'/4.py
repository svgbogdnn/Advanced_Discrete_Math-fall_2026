from collections import deque

def chains(graph, start):
    n = len(graph) - 1

    distance = [-1] * (n+1)
    distance[start] = 0
    queue = deque([start])
    furthest_thing = start

    while queue:
        
        v = queue.popleft()

        for w in graph[v]:
            
            if distance[w] == -1:
                distance[w] = distance[v] + 1
                queue.append(w)

            if distance[w] > distance[furthest_thing]:
                furthest_thing = w

    return furthest_thing, distance[furthest_thing]

def solve():
    n = int(input())

    graph = []    
    for i in range(n+1):
        graph.append([])


    for i in range(n-1):
        a, b = map(int, input().split())
        graph[b].append(a)
        graph[a].append(b)

    start = chains(graph, 1)[0]
    distance = chains(graph, start)[1] + 1

    print(distance)

if __name__ == '__main__':
    solve()