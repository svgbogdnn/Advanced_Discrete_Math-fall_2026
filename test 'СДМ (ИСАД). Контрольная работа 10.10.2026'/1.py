def solve():
    n = int(input())

    for i in range (n):
        row = input().split()

        for j in range (n):
            if row[j] == '1':
                print(i+1, j+1)

if __name__ == '__main__':
    solve()