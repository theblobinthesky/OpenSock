import numpy as np
import cv2

def generate_checkerboard(rows, cols, square_size, output_path):
    """
    Generate and save a checkerboard pattern image.

    Args:
        rows (int): Number of inner corners per column.
        cols (int): Number of inner corners per row.
        square_size (int): Size of each square in pixels.
        output_path (str): Path to save the generated PNG image.
    """
    # Compute total image size
    height = (rows + 1) * square_size
    width = (cols + 1) * square_size

    # Create a blank (black) image
    board = np.zeros((height, width), dtype=np.uint8)

    # Fill squares
    for y in range(rows + 1):
        for x in range(cols + 1):
            if (x + y) % 2 == 0:
                top_left = (x * square_size, y * square_size)
                bottom_right = ((x + 1) * square_size, (y + 1) * square_size)
                cv2.rectangle(board, top_left, bottom_right, 255, -1)

    # Save as PNG
    cv2.imwrite(output_path, board)

if __name__ == "__main__":
    # Example: 9x6 inner corners, 50px squares
    generate_checkerboard(rows=6, cols=9, square_size=250, output_path="../data/checkerboard.png")

