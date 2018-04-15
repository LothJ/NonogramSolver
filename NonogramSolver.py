import itertools
import numpy as np
import cv2
from time import time


# The function scan_line tries to fit the clues in the line as it currently is. Where there is definitely color or
# no color, that is filled in the line. When encountering impossibilities, None is returned. Otherwise an indication
# of the uncertainty of the line is returned in the form of the percentage of unknown fields in the line.
def _scan_line(old, clues):
    # if no clues, the line should be all 0
    if not clues:
        # if color is found, it is invalid
        if True in (old >= 1):
            return None
        old[:] = 0
        return 0

    # Find the lowest starting points of all clues while making sure the chunk is not blocked by 0
    place_first = [0] * len(clues)
    place = 0
    for idx, curr_clue in enumerate(clues):
        while True:
            # if the chunks do not fit the line, it is invalid
            if place + curr_clue > len(old):
                return None
            if 0 not in old[place:place + curr_clue]:
                place_first[idx] = place
                place += curr_clue + 1
                break
            place += 1

    # retrace to move the locations when needed with possible back tracking to make sure the line is valid
    stop = len(old)
    idx = len(clues)
    while idx > 0:
        idx -= 1
        # all possible invalid situations
        while 1 in old[place_first[idx] + clues[idx]:stop] or \
                0 in old[place_first[idx]:place_first[idx] + clues[idx]] or \
                (place_first[idx] > 0 and old[place_first[idx] - 1] >= 1) or \
                (stop < len(old) and place_first[idx] + clues[idx] >= stop) or \
                (stop == len(old) and place_first[idx] + clues[idx] > stop):
            # when there is no room to correct, move back a clue to reposition that one
            if (stop < len(old) and place_first[idx] + clues[idx] >= stop) or \
                    (stop == len(old) and place_first[idx] + clues[idx] > stop):
                idx += 2
                # if the last chunk has no room, the line is invalid
                if idx > len(clues):
                    return None
                place_first[idx - 1] += 1
                break
            place_first[idx] += 1
        if idx == len(clues):
            stop = len(old)
        else:
            stop = place_first[idx]
    # if color is left that is not a part of the clues, the line is invalid
    if True in (old[0:stop] >= 1):
        return None

    # Find the highest ending points of all clues while making sure the chunk is not blocked by 0
    place_last = [0] * len(clues)
    place = len(old)
    for idx, curr_clue in reversed(list(enumerate(clues))):
        while True:
            # if the chunks do not fit the line, it is invalid
            if place - curr_clue < 0:
                return None
            if 0 not in old[place - curr_clue:place]:
                place_last[idx] = place
                place -= curr_clue + 1
                break
            place -= 1

    # retrace to move the locations when needed with possible back tracking to make sure the line is valid
    start = 0
    idx = -1
    while idx < len(clues) - 1:
        idx += 1
        # all possible invalid situations
        while 1 in old[start:place_last[idx] - clues[idx]] or \
                0 in old[place_last[idx] - clues[idx]:place_last[idx]] or \
                (place_last[idx] < len(old) and old[place_last[idx]] >= 1) or \
                (start > 0 and place_last[idx] - clues[idx] <= start) or \
                (start == 0 and place_last[idx] - clues[idx] < start):
            # when there is no room to correct, move back a clue to reposition that one
            if (start > 0 and place_last[idx] - clues[idx] <= start) or \
                    (start == 0 and place_last[idx] - clues[idx] < start):
                idx -= 2
                # if the first chunk has no room, the line is invalid
                if idx < -1:
                    return None
                place_last[idx + 1] -= 1
                break
            place_last[idx] -= 1
        if idx == -1:
            start = 0
        else:
            start = place_last[idx]
    # if color is left that is not a part of the clues, the line is invalid
    if True in (old[start:len(old)] >= 1):
        return None

    distances = np.subtract(np.subtract(place_last, place_first), clues)
    for i in range(len(clues)):
        # fill in the overlap of a single clue between first and last where there is definately color
        if clues[i] > distances[i]:
            old[place_last[i] - clues[i]:place_first[i] + clues[i]] = 1

        # If there is already color in the space of the chunk, one of the clues has to represent that place. Here all
        # possible clues to represent that field are checked to see where there is definately color across all clues.
        # Those fields are then colored in.
        if (not np.all(old[place_first[i]:place_last[i]] == 1)) and clues[i] > 1:
            # do the check for all colored fields found but only if the chunk is not already locked by the clue
            for place_idx in np.where(old[place_first[i]:place_first[i] + clues[i]] == 1)[0]:
                place_point = place_idx + place_first[i]
                # find the range of clues that can represent the field place_point
                for j in range(len(clues)):
                    if place_last[j] > place_point:
                        other_start = j
                        break
                for j in reversed(range(len(clues))):
                    if place_first[j] <= place_point:
                        other_end = j
                        break
                # if more than one clues are found, find the first and last possible position and record the overlap
                if not other_start == other_end:
                    min_clue = min(clues[other_start:other_end+1])
                    overlap = np.ones((min_clue * 2) - 1, dtype=int)
                    # set places that fall outside the line to 0
                    if place_point - (min_clue - 1) < 0:
                        overlap[:(min_clue - 1) - place_point] = 0
                    if place_point + (min_clue - 1) >= len(old):
                        overlap[(len(old) - place_point) + (min_clue - 1):] = 0
                    # set the original colored field to 0
                    overlap[min_clue - 1] = 0

                    for j in range(other_start, other_end + 1):
                        if place_point + clues[j] < len(old):
                            overlap_end = place_point + clues[j]
                        else:
                            overlap_end = len(old)
                        # find where the clue could be relative to place_point
                        while overlap_end > place_point + 1 and \
                                ((overlap_end < len(old) and old[overlap_end] == 1) or
                                    (overlap_end - clues[j] > 0 and old[(overlap_end - clues[j]) - 1] == 1) or
                                    0 in old[overlap_end - clues[j]:overlap_end]):
                            overlap_end -= 1
                        # make sure the found place_last does not conflict with overlap_end
                        overlap_end = min(overlap_end, place_last[j])

                        if (place_point + 1) - clues[j] > 0:
                            overlap_start = (place_point + 1) - clues[j]
                        else:
                            overlap_start = 0
                        # find where the clue could be relative to place_point
                        while overlap_start < place_point and \
                                ((overlap_start + clues[j] < len(old) and old[overlap_start + clues[j]] == 1) or
                                    (overlap_start > 0 and old[overlap_start - 1] == 1) or
                                    0 in old[overlap_start:overlap_start + clues[j]]):
                            overlap_start += 1
                        # make sure the found place_first does not conflict with overlap_start
                        overlap_start = max(overlap_start, place_first[j])

                        # find unsure fields in the overlap and set them to 0
                        for k in np.where(overlap == 1)[0]:
                            k_place = (k - (min_clue - 1)) + place_point
                            if k_place >= overlap_start + min_clue or k_place < overlap_end - min_clue:
                                overlap[k] = 0
                        # if overlap is all 0 there is no reason to continue
                        if 1 not in overlap:
                            break
                    # fill in the fields where overlap is certain
                    for k in np.where(overlap == 1)[0]:
                        k_place = (k - (min_clue - 1)) + place_point
                        old[k_place] = 1

        # If multiple clues can represent one completed chunk, the chunk can be surrounded with 0 if all possible clues
        # are the same size.
        if place_first[i] + clues[i] in place_last[:i]:
            clue_range = place_last.index(place_first[i] + clues[i])
            if np.all(old[place_first[i]:place_first[i] + clues[i]] == 1) and \
                    all(j == clues[i] for j in clues[clue_range:i]):
                if place_first[i] > 0:
                    old[place_first[i] - 1] = 0
                if place_first[i] + clues[i] < len(old):
                    old[place_first[i] + clues[i]] = 0

    # fill in the overlap of between spaces where there is definately no color
    old[:place_first[0]] = 0
    for i in range(1, len(clues)):
        if place_first[i] > place_last[i - 1]:
            old[place_last[i - 1]:place_first[i]] = 0
    old[place_last[-1]:] = 0

    # fill in gaps too small to fit any clues
    size_gap = -1
    previous = 0
    for i, value in enumerate(old):
        if value > 0:
            size_gap = -1
        if value == -1 and previous == 0:
            size_gap = 1
        elif value == -1 and size_gap > 0:
            size_gap += 1
        if value == 0 and size_gap > 0:
            if size_gap < min(clues):
                old[i - size_gap:i] = 0
            size_gap = -1
        previous = value

    # Return the percentage of uncertain fields in the line as an indication of priority
    if -1 in old:
        return sum(old == -1) / len(old)
    return 0


# Add a position to put in the stack in order of least uncertainty to most uncertain
def _add2stack(stack, place, distance):
    for i in range(len(stack)):
        if distance < stack[i][1]:
            stack.insert(i, (place, distance))
            return
    stack.append((place, distance))


# Find the combination of x_stack and y_stack with the lowest uncertainty that has -1 at the crossing field
def _pop_stacks(x_stack, y_stack, field):
    combinations = list(itertools.product(range(len(x_stack)), range(len(y_stack))))
    for i in range(len(x_stack) + len(y_stack) - 1):
        for item in combinations:
            if sum(item) == i:
                if field[y_stack[item[1]][0], x_stack[item[0]][0]] == -1:
                    return x_stack[item[0]][0], y_stack[item[1]][0]


# The recursion function tries to solve as much as possible with scan_line(). After that it takes an informed guess at
# the least uncertain field and makes a copy of the field to let the next recursion instance do the same with the copy.
def _recursion(field, x_stack, y_stack, row_clues, col_clues, count, start_time, max_search_time, visual):
    # If visual is true, show the field as it is solved
    if visual:
        visualize_field(field, 1)

    # Check if the maximum search time is exceeded.
    if start_time is not None:
        if time() - start_time > max_search_time:
            return None

    # while the field is changed, update the stacks and in the process run scan_line
    while True:
        old_field = np.copy(field)
        update = _update_stacks(x_stack, y_stack, field, row_clues, col_clues)
        if not update:
            return None
        (x_stack, y_stack) = update
        # also show the progress here if visual
        if visual:
            visualize_field(field, 1)
        if np.array_equal(field, old_field):
            break

    # When no uncertain places are in the field a solution is found. The puzz solution check is not reliable so the
    # check is done by my own function
    if -1 not in field:
        # if puzz.is_correct_solution(field):
        if check_solution(row_clues, col_clues, field):
            return field
        return None

    # A new place to make a guess is picked and the first color to try is chosen
    new_x, new_y = _pop_stacks(x_stack, y_stack, field)
    color = _probable_color(field, row_clues, col_clues, new_x, new_y)
    result = None
    counter = 0
    while result is None and counter < 2:
        field[new_y, new_x] = color
        result = _recursion(field.copy(), x_stack[:], y_stack[:], row_clues,
                            col_clues, count+1, start_time, max_search_time, visual)
        color = (color + 1) % 2
        counter += 1
    return result


# The first color to try is chosen based on the percentage of color indicated by the clues
def _probable_color(field, row_clues, col_clues, new_x, new_y):
    line_x = field[:, new_x]
    line_y = field[new_y, :]
    black = (line_x == 1).sum() + (line_y == 1).sum()
    white = (line_x == 0).sum() + (line_y == 0).sum()
    clues_x = col_clues[new_x]
    clues_y = row_clues[new_y]
    if black + white == 0:
        if sum(clues_x) + sum(clues_y) / (len(line_x) + len(line_y)) < 0.5:
            return 0
        return 1
    if black / (black + white) < sum(clues_x) + sum(clues_y) / (len(line_x) + len(line_y)):
        return 1
    return 0


# Both the stacks are refreshed by going through all lines with scan_line() and updating the order of the stacks.
def _update_stacks(x_stack, y_stack, field, row_clues, col_clues):
    new_x_stack = []
    new_y_stack = []
    for x in x_stack:
        dist_x = _scan_line(field[:, x[0]], col_clues[x[0]])
        if dist_x is None:
            return False
        if dist_x > 0:
            _add2stack(new_x_stack, x[0], dist_x)

    for y in y_stack:
        dist_y = _scan_line(field[y[0], :], row_clues[y[0]])
        if dist_y is None:
            return False
        if dist_y > 0:
            _add2stack(new_y_stack, y[0], dist_y)

    return new_x_stack, new_y_stack


# Similar to the function update_stacks(), this function goes through all the lines to create the stacks.
def _create_stacks(x_stack, y_stack, field, row_clues, col_clues):
    for x in range(len(col_clues)):
        dist_x = _scan_line(field[:, x], col_clues[x])
        if dist_x is None:
            return False
        if dist_x > 0:
            _add2stack(x_stack, x, dist_x)

    for y in range(len(row_clues)):
        dist_y = _scan_line(field[y, :], row_clues[y])
        if dist_y is None:
            return False
        if dist_y > 0:
            _add2stack(y_stack, y, dist_y)
    return True


# Translate the field to white, grey and black representation and show it in the window
def visualize_field(field, waittime):
    img = field.astype(float)
    img[img == -1] = 0.5
    img[img == 1] = -1
    img[img == 0] = 1
    img[img == -1] = 0
    cv2.imshow('image', img)
    cv2.waitKey(waittime)


# Quick and dirty function that runs through a line and records clues that would represent that line.
def _create_clues_from_line(line):
    chunk = []
    previous = 0
    for pixel in line:
        if pixel == 1:
            if previous == 1:
                chunk[-1] += 1
            else:
                chunk.append(1)
        previous = pixel
    return chunk


# To check a solution, get the clues that would represent the given solution and compare that to the actual clues.
# If no difference is found, the given solution is a valid one but not necessary the right one if multiple solutions
# are possible.
def check_solution(row_clues, col_clues, solution):
    if -1 in solution:
        return False

    for i in range(len(row_clues)):
        if row_clues[i] != _create_clues_from_line(solution[i, :]):
            return False

    for i in range(len(col_clues)):
        if col_clues[i] != _create_clues_from_line(solution[:, i]):
            return False

    return True


# This function handles the process of solving a pixelo puzzel.
def solve_pixelo(row_clues, col_clues, max_search_time=0, visual=True):
    start_time = None
    if max_search_time > 0:
        start_time = time()

    nr_rows = len(row_clues)
    nr_cols = len(col_clues)

    # start the field with the appropriate size
    field = np.zeros((nr_rows, nr_cols), dtype=np.int8) - 1
    # if visual, create the image to display the process of solving and the result with the appropriate dimensions
    if visual:
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        if nr_rows > nr_cols:
            cv2.resizeWindow('image', 600 * nr_cols // nr_rows, 600)
        else:
            cv2.resizeWindow('image', 600, 600 * nr_rows // nr_cols)

    x_stack = []
    y_stack = []
    solution = None
    if _create_stacks(x_stack, y_stack, field, row_clues, col_clues):
        solution = _recursion(field, x_stack, y_stack, row_clues, col_clues, 0, start_time, max_search_time, visual)
    # if np solution is found, checking None would cause an error so the solution is the original field

    if solution is None:
        solution = field

    return solution


if __name__ == "__main__":
    start_time = time()

    row_clues = [[2], [2, 1], [1, 1], [3], [1, 1], [1, 1], [2], [1, 1], [1, 2], [2]]
    col_clues = [[2, 1], [2, 1, 3], [7], [1, 3], [2, 1]]
    solution = solve_pixelo(row_clues, col_clues, max_search_time=60)

    if check_solution(row_clues, col_clues, solution):
        print("Puzzel solved after {:.3f} seconds".format(time() - start_time))

    else:
        print("No solution was found after {:.3f} seconds".format(time() - start_time))

    visualize_field(solution, 0)
