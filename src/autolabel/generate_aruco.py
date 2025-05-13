import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_aruco_marker(marker_id: int, dictionary: int, size: int) -> np.ndarray:
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size)
    white_bg = np.ones((size, size), dtype=np.uint8) * 255
    offset = (white_bg.shape[0] - marker.shape[0]) // 2
    white_bg[offset:offset + marker.shape[0], offset:offset + marker.shape[0]] = marker
    return white_bg


def generate_aruco_marker(marker_id: int, dictionary: int, size: int) -> np.ndarray:
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size)
    white_bg = np.ones((size, size), dtype=np.uint8) * 255
    offset = (white_bg.shape[0] - marker.shape[0]) // 2
    white_bg[offset:offset + marker.shape[0], offset:offset + marker.shape[0]] = marker

    return white_bg


def create_a4_with_markers(marker_ids: list[int],
                           shape: tuple[int, int],
                           dictionary: int,
                           marker_size: int,
                           marker_margin: int,
                           dpi: int,
                           angle: float) -> np.ndarray:
    mm_to_pixels = dpi / 25.4
    marker_size = int(marker_size * mm_to_pixels)
    marker_margin = int(marker_margin * mm_to_pixels)
    bg = 255

    # Render the individual markers.
    markers = []
    for marker_id in marker_ids:
        marker = generate_aruco_marker(marker_id, dictionary, marker_size)
        markers.append(marker)

    # Put into shape.
    size = marker_size + marker_margin

    additional_side_margin = int(np.ceil(
        np.sqrt((size * shape[0] - marker_margin) ** 2 + (size * shape[1] - marker_margin) ** 2)
        - (size * max(shape[0], shape[1]) - marker_margin)
        - 2 * marker_margin
    ) / 2)

    grid = np.full(
        (size * shape[0] + marker_margin + 2 * additional_side_margin,
         size * shape[1] + marker_margin + 2 * additional_side_margin),
        bg, np.uint8
    )
    idx = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            y0 = additional_side_margin + marker_margin + i * size
            x0 = additional_side_margin + marker_margin + j * size
            grid[y0:y0 + marker_size,
                 x0:x0 + marker_size] = markers[idx]
            idx += 1

    # Composite into A4 sheet at an angle.
    max_size = max(grid.shape)           # NumPy shape is (h, w)
    sin, cos = np.sin(angle), np.cos(angle)
    x, y = -grid.shape[1] / 2, -grid.shape[0] / 2
    sh = max_size / 2
    H = np.array([
        [cos, -sin, x * cos - sin * y + sh],
        [sin,  cos, x * sin + y * cos + sh]
    ])
    rotated_grid = cv2.warpAffine(grid, H, dsize=(max_size, max_size))

    # Compose final A4 sheet (portrait) using original √2 logic.
    a4_h = int(round(max_size * np.sqrt(2)))  # height (rows)
    a4_w = max_size                            # width (cols)
    a4_sheet = np.full((a4_h, a4_w), 255, np.uint8)
    a4_sheet[:max_size, :max_size] = rotated_grid

    return a4_sheet


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate A4 sheet with ArUco markers")
    parser.add_argument("--dict", type=int, default=cv2.aruco.DICT_6X6_250, help="Dictionary type")
    parser.add_argument("--size", type=int, default=80, help="Marker size in millimeters")
    parser.add_argument("--dpi", type=int, default=300, help="Output resolution in DPI")

    args = parser.parse_args()

    marker_sheets = [
        ("markers1.jpg", [1, 2, 3, 4], (2, 2)),
        ("markers2.jpg", [5, 6, 7, 8], (2, 2)),
        ("markers3.jpg", [9, 10, 11, 12], (2, 2)),
        ("markers4.jpg", [13, 14, 15, 16], (2, 2))
    ]

    for i, (file, marker_ids, shape) in enumerate(marker_sheets):
        sheet = create_a4_with_markers(
            marker_ids, shape,
            dictionary=args.dict,
            marker_size=args.size,
            marker_margin=20,
            dpi=args.dpi,
            angle=0.5 * np.pi * i / len(marker_sheets)
        )

        cv2.imwrite(f"../data/{file}", sheet)

if __name__ == "__main__":
    main()
