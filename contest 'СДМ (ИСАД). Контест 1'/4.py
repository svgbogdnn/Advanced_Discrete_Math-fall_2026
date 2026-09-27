from collections import deque


def ChampagnePapi21(first_position, second_position):
    moves = [
        (-2, -1),
        (-2, 1),
        (-1, -2),
        (-1, 2),
        (1, -2),
        (1, 2),
        (2, -1),
        (2, 1),
    ]
    queue = deque()
    visited = set()

    queue.append((first_position, second_position, 0))
    visited.add((first_position, second_position))

    while queue:
        first, second, steps = queue.popleft()

        if first == second:
            return steps

        for first_move in moves:
            next_first_row = first[0] + first_move[0]
            next_first_column = first[1] + first_move[1]
            first_is_inside = (
                0 <= next_first_row < 8
                and 0 <= next_first_column < 8
            )

            if not first_is_inside:
                continue

            next_first = (next_first_row, next_first_column)

            for second_move in moves:
                next_second_row = second[0] + second_move[0]
                next_second_column = second[1] + second_move[1]
                second_is_inside = (
                    0 <= next_second_row < 8
                    and 0 <= next_second_column < 8
                )

                if not second_is_inside:
                    continue

                next_second = (next_second_row, next_second_column)
                state = (next_first, next_second)

                if state not in visited:
                    visited.add(state)
                    queue.append((next_first, next_second, steps + 1))

    return -1


def solve():
    positions = input().split()

    while len(positions) < 2:
        positions.extend(input().split())

    first_position = (
        int(positions[0][1]) - 1,
        ord(positions[0][0]) - ord('a'),
    )
    second_position = (
        int(positions[1][1]) - 1,
        ord(positions[1][0]) - ord('a'),
    )
    answer = ChampagnePapi21(first_position, second_position)
    print(answer)


if __name__ == '__main__':
    solve()