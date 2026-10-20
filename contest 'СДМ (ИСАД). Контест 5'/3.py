def ChampagnePapi21(n, operations):
    parent = list(range(n))
    size = [1] * n
    answers = []

    for kind, first, second in reversed(operations):
        root_first = first
        while parent[root_first] != root_first:
            parent[root_first] = parent[parent[root_first]]
            root_first = parent[root_first]

        root_second = second
        while parent[root_second] != root_second:
            parent[root_second] = parent[parent[root_second]]
            root_second = parent[root_second]

        if kind == "cut":
            if root_first != root_second:
                if size[root_first] < size[root_second]:
                    temporary_root = root_first
                    root_first = root_second
                    root_second = temporary_root
                parent[root_second] = root_first
                size[root_first] = size[root_first] + size[root_second]
        elif root_first == root_second:
            answers.append("YES")
        else:
            answers.append("NO")

    answers.reverse()
    return answers


def solve():
    n, m, k = map(int, input().split())
    for _ in range(m):
        input()

    operations = []
    for _ in range(k):
        kind, first, second = input().split()
        operations.append((kind, int(first) - 1, int(second) - 1))

    answers = ChampagnePapi21(n, operations)
    for answer in answers:
        print(answer)


if __name__ == '__main__':
    solve()