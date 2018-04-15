import json


class Puzzle(object):
    def __init__(self, puzzle_number, group='test'):
        try:
            f = open('{}_puzzles/{}.json'.format(group, puzzle_number), 'r')
        except IOError:
            raise ValueError("""
                This puzzle does not exist.

                For test puzzles pick a number in the range [1, 3825]
                For control puzzles pick a number in the range [1, 3918]

                """)

        json_data = json.load(f)
        json_data.update(json_data['details'])
        json_data['_solution'] = json_data['solution']

        del json_data['solution']
        del json_data['details']

        self.__dict__.update(json_data)
        f.close()

        del json_data

    def is_correct_solution(self, solution):
        if not len(solution[0]) == self.columns:
            return False
        if not len(solution) == self.rows:
            return False
        if solution.min() == -1:
            return False

        max_val = max([max(row) for row in solution] + [1])

        sol_attempt = ''
        for row in solution:
            sol_attempt += hex(int(''.join(str(v) for v in row), max_val + 1))

        return sol_attempt == self._solution


if __name__ == "__main__":
    # Import modules to showcase NonogramSolver
    import os
    from time import time
    import numpy as np
    import NonogramSolver

    # Find how many (possible) puzzle files there are in the test_puzzle directory. It does not check if the files
    # are valid or correctly named.
    directory = os.path.join(os.getcwd(), 'test_puzzles')
    nr_puzzle_files = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

    # Randomly pick 10 puzzles and solve them
    for i in range(10):
        start_time = time()

        puzzle_nr = np.random.randint(nr_puzzle_files) + 1
        puzzle = Puzzle(puzzle_nr)
        solution = NonogramSolver.solve_pixelo(puzzle.row_clues, puzzle.col_clues, max_search_time=60)

        message = 'Puzzle nr: {} Time Taken: {:.3f} seconds.'.format(puzzle_nr, time() - start_time)
        if puzzle.unique:
            message += " Only 1 possible solution."
        else:
            message += " Multiple possible solutions."
        if puzzle.is_correct_solution(solution):
            message += " The solution was confirmed!"
        else:
            if NonogramSolver.check_solution(puzzle.row_clues, puzzle.col_clues, solution):
                message += " The solution was not accepted but it was still conform with the given clues."
            else:
                message += " No solution was found"

        print(message)

        NonogramSolver.visualize_field(solution, 2000)
