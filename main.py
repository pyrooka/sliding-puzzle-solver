import time
import argparse
import threading
import webbrowser

import cv2
import numpy as np
import skimage.measure as skm

import solver


ROWS, COLS = 4, 4

def resize_to_smaller(image_1: np.ndarray, image_2: np.ndarray) -> tuple:
    """Resize the larger image to the size of the smaller one.

    Args:
        image_1 (np.ndarray): Image one.
        image_2 (np.ndarray): Image two.

    Returns:
        tuple: The resized image and the original one. (image_1, image_2)
    """

    image_1_rows, image_1_cols, _ = image_1.shape
    image_2_rows, image_2_cols, _ = image_2.shape

    if image_1.size == image_2.size:
        # If the size is same, choose by the rows.
        if image_1_rows < image_2_rows:
            return (image_1, cv2.resize(image_2, (image_1_cols, image_1_rows), interpolation=cv2.INTER_AREA))
    elif image_1.size < image_2.size:
        return (image_1, cv2.resize(image_2, (image_1_cols, image_1_rows), interpolation=cv2.INTER_AREA))
    else:
        return (cv2.resize(image_1, (image_2_cols, image_2_rows), interpolation=cv2.INTER_AREA), image_2)

def get_tiles(image: np.ndarray, rows: int, cols: int) -> dict:
    """Divide an image to tiles.

    Args:
        image (np.ndarray): Grayscale base image.
        rows (int): Number of rows for divide.
        cols (int): Number of columns for divide.

    Returns:
        dict: Dictionary with keys of the coordinate (row, col) and value of the numpy array (image).
    """

    height, width, _ = image.shape

    tile_height = height // rows
    tile_width = width // cols

    tiles = {}

    for row in range(rows):
        for col in range(cols):
            row_from = row * tile_height
            row_to = (row + 1) * tile_height
            col_from = col * tile_width
            col_to = (col + 1) * tile_width

            tiles[(row, col)] = image[row_from:row_to, col_from:col_to]

    return tiles

def match_tile(tile_original: np.ndarray, tile_current: np.ndarray) -> float:
    """Match two tiles with the mean structural similarity index (SSIM).

    Args:
        tile_original (np.ndarray): A gray image.
        tile_current (np.ndarray): A gray image.

    Returns:
        float: Similarity index.
    """

    return skm.compare_ssim(tile_original, tile_current, multichannel=True)

def get_positions(original_tiles: dict, current_tiles: dict, rows: int, cols: int) -> np.ndarray:
    """Calculate the positions of the original tiles on the current state.

    Args:
        original_tiles (dict): Tiles of the original image.
        current_tiles (dict): TIles of the current image.
        rows (int): Number of rows in the image.
        cols (int): Number of cols in the image.

    Returns:
        np.ndarray: A matrix with the tile positions represented with its number.
        E.g.:
        [
            [nan,   nan,    nan,    0],
            [1,     2,      3,      4],
            [5,     6,      7,      8],
            [9,     10,     11,     12],
            [11,    12,     13,     14]
        ]
    """

    # Create the result "body".
    base = np.zeros((rows, cols))

    # Iterate over the original tiles.
    for pos_orig, tile_orig in original_tiles.items():
        row_orig, col_orig = pos_orig
        tile_number = (row_orig * cols) + (col_orig + 1)

        best_similarity = None
        best_tile_index = None

        for pos_curr, tile_curr in current_tiles.items():
            similarity = match_tile(tile_orig, tile_curr)

            if best_similarity is None or similarity > best_similarity:
                best_similarity = similarity
                best_tile_index = pos_curr

        base[best_tile_index] = tile_number

    # Create the top row.
    top = np.array([np.nan, np.nan, np.nan, 0])

    # Put them together.
    result = np.vstack((top, base))

    return result

def process_images(original: np.ndarray, current: np.ndarray):
    """Process the images.

    Args:
        original (np.ndarray): Image of the original state.
        current (np.ndarray): Image of the current state.

    Returns:
        [type]: [description]
    """

    # Resize the larger image to the size of the smaller one.
    orig, curr = resize_to_smaller(original, current)

    # Get the tiles for the images.
    orig_tiles = get_tiles(orig, ROWS, COLS)
    curr_tiles = get_tiles(curr, ROWS, COLS)

    # Get the tile positions on the current image based on the original.
    positions = get_positions(orig_tiles, curr_tiles, ROWS, COLS)

    # Create the solver with the current state.
    s = solver.PuzzleSolver(positions)
    moves = s.solve()

    return moves

def print_solution(moves: list):
    """Print the solution in better format.

    Args:
        moves (list): list of tables.
    """

    moves_count = len(moves)

    if moves_count < 1:
        print('Ooops. No solution found.')

    print('Doing {0} moves'.format(moves_count))
    for i in range(1, moves_count):
        prev = moves[i-1]
        curr = moves[i]

        prev_zero = np.argwhere(prev.board==0)[0]
        curr_zero = np.argwhere(curr.board==0)[0]

        if prev_zero[0] < curr_zero[0]:
            print('DOWN')
        elif prev_zero[0] > curr_zero[0]:
            print('UP')
        elif prev_zero[1] < curr_zero[1]:
            print('RIGHT')
        elif prev_zero[1] > curr_zero[1]:
            print('LEFT')
        else:
            print('Sorry WHAT?')
            prev.pprint()
            curr.pprint()

        time.sleep(1)

def main():
    # Parse the CLI arguments.
    parser = argparse.ArgumentParser(description='Puzzle solver.')
    parser.add_argument('original_image', type=str, help='Path to the solved puzzle.')
    parser.add_argument('current_image', type=str, help='Path to the current puzzle image.')
    args = parser.parse_args()

    # Load the images.
    original = cv2.imread(args.original_image)
    current = cv2.imread(args.current_image)

    moves = process_images(original, current)

    print_solution(moves)

if __name__ == '__main__':
    main()