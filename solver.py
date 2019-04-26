import heapq

import numpy as np

class Board:

    def __init__(self, board: np.ndarray, parent=None):
        self.board = board
        self.parent = parent
        self.children = []

        self.calc_score()

        self.idx = self.board.tostring()

    def __lt__(self, other):
        """For heapq sorting when the priority is equals.
        """

        return self.score > other.score

    def calc_score(self):
        """Calculate the score of the board.

        This is the heuristic function.
        """

        score = 0
        row, col = self.board.shape

        for r in range(row):
            for c in range(col):
                value = self.board[r, c]

                # If it's a nan or zero, skip.
                if np.isnan(value) or value == 0:
                    continue

                # Increase the value. Move 2 forward.
                value += 3

                value_row = value // col
                value_col = value % col

                score += abs(value_row - r) + abs(value_col - c)

        self.score = score

    def generate_children(self):
        zero_loc = np.argwhere(self.board==0)[0]

        moves = ((1, 0), (0, 1), (-1, 0), (0, -1))

        for variation in moves:
            board_copy = np.copy(self.board)

            move_to = zero_loc + variation

            # If new location is out of bound.
            if np.any(move_to<0) or move_to[0] >= self.board.shape[0] or move_to[1] >= self.board.shape[1]:
                continue

            # If new location is blocked field (nan).
            if np.isnan(board_copy[tuple(move_to)]):
                continue

            board_copy[tuple(zero_loc)] = board_copy[tuple(move_to)]
            board_copy[tuple(move_to)] = 0

            self.children.append(Board(board_copy, self))

    def get_best_child(self):
        best = None

        for child in self.children:
            if (best is None or child.score < best.score) and not self.is_parent(child):
                best = child

        return best

    def pprint(self):
        print('Score: ' + str(self.score))
        print(self.board)

    def is_parent(self, child):
        if self.parent is None:
            return False

        return (self.parent.board == child.board).all()

class PuzzleSolver:

    def __init__(self, board: np.ndarray):
        self.prio_queue = []
        self.costs = {}
        self.path = {}

        self.initial_board = Board(board)

        heapq.heappush(self.prio_queue, (0, self.initial_board))
        self.costs[self.initial_board.idx] = 0

    def solve(self):
        """Solve the game.

        Returns:
            list: Reconstructed moves.
        """

        while self.prio_queue:
            _, board = heapq.heappop(self.prio_queue)

            #board.pprint()

            if board.score == 0:
                return self.reconstruct_solution(board)

            board.generate_children()

            for child in board.children:
                cost = board.score + 1

                if child.idx not in self.costs or cost < self.costs[child.idx]:
                    self.costs[child.idx] = cost
                    priority = cost + child.score
                    heapq.heappush(self.prio_queue, (priority, child))

        return []

    def reconstruct_solution(self, final: Board):
        """Rebuild the solution.

        Args:
            final (Board): Solved game.

        Returns:
            list: Moves for solving the game.
        """

        solution = []
        solution.append(final)

        board = final
        while board.parent is not None:
            solution.append(board.parent)
            board = board.parent

        solution.reverse()

        return solution

def main():
    """
    ONLY FOR TESTING!
    """

    moves_10 = np.array([np.nan, np.nan, np.nan, 0, 1, 2, 8, 4, 5, 6, 3, 12, 9, 10, 7, 16, 13, 14, 11, 15])
    custom = moves_10

    custom = custom.reshape(5, 4)

    solver = PuzzleSolver(custom)
    moves = solver.solve()

    for move in moves:
        move.pprint()

    print(len(moves))
    print('DONE!!')


if __name__ == '__main__':
    main()
