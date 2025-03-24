import cv2
import numpy as np

def generate_aruco_marker(marker_id: int, dictionary: int, size: int, border: int) -> np.ndarray:
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size)
    
    # Add white border
    white_bg = np.ones((size + 2 * border, size + 2 * border), dtype=np.uint8) * 255
    marker_size = marker.shape[0]
    offset = (white_bg.shape[0] - marker_size) // 2
    white_bg[offset:offset + marker_size, offset:offset + marker_size] = marker
    
    return white_bg

def create_a4_with_markers(marker1_id: int, marker2_id: int, dictionary: int, marker_size: int, output_path: str, dpi: int) -> np.ndarray:
    # A4 dimensions in mm (210 x 297)
    a4_width_mm, a4_height_mm = 210, 297
    
    # Convert mm to pixels based on DPI
    mm_to_pixels = dpi / 25.4  # 25.4 mm = 1 inch
    a4_width_px = int(a4_width_mm * mm_to_pixels)
    a4_height_px = int(a4_height_mm * mm_to_pixels)
    marker_size_px = int(marker_size * mm_to_pixels)
    
    # Create blank white A4 sheet
    a4_sheet = np.ones((a4_height_px, a4_width_px), dtype=np.uint8) * 255
    
    # Define border as 5% of marker size in pixels
    border_px = int(marker_size_px * 0.05)
    
    # Generate two markers
    marker1 = generate_aruco_marker(marker1_id, dictionary, marker_size_px, border_px)
    marker2 = generate_aruco_marker(marker2_id, dictionary, marker_size_px, border_px)
    
    # Calculate positions with a 10mm margin
    margin_px = int(10 * mm_to_pixels)
    marker1_x, marker1_y = margin_px, margin_px
    marker2_x = a4_width_px - marker2.shape[1] - margin_px
    marker2_y = margin_px
    
    # Place markers on the sheet
    a4_sheet[marker1_y:marker1_y+marker1.shape[0], marker1_x:marker1_x+marker1.shape[1]] = marker1
    a4_sheet[marker2_y:marker2_y+marker2.shape[0], marker2_x:marker2_x+marker2.shape[1]] = marker2
    
    # Set up font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0 * (dpi / 300)
    text_color = 0  # Black
    thickness = max(2, int(1.5 * (dpi / 300)))
    
    # Add text below marker1
    marker1_text_y = marker1_y + marker1.shape[0] + int(5 * mm_to_pixels)
    cv2.putText(a4_sheet, f"ArUco Marker ID: {marker1_id}", (marker1_x, marker1_text_y), font, font_scale, text_color, thickness)
    cv2.putText(a4_sheet, "Dictionary: DICT_6X6_250", (marker1_x, marker1_text_y + int(7 * mm_to_pixels)), font, font_scale, text_color, thickness)
    cv2.putText(a4_sheet, f"Size: {marker_size}mm x {marker_size}mm", (marker1_x, marker1_text_y + int(14 * mm_to_pixels)), font, font_scale, text_color, thickness)
    
    # Add text below marker2
    marker2_text_y = marker2_y + marker2.shape[0] + int(5 * mm_to_pixels)
    cv2.putText(a4_sheet, f"ArUco Marker ID: {marker2_id}", (marker2_x, marker2_text_y), font, font_scale, text_color, thickness)
    cv2.putText(a4_sheet, "Dictionary: DICT_6X6_250", (marker2_x, marker2_text_y + int(7 * mm_to_pixels)), font, font_scale, text_color, thickness)
    cv2.putText(a4_sheet, f"Size: {marker_size}mm x {marker_size}mm", (marker2_x, marker2_text_y + int(14 * mm_to_pixels)), font, font_scale, text_color, thickness)
    
    # Add printing instructions at the bottom of the page
    instructions = [
        "INSTRUCTIONS:",
        "1. Print this page at 100% scale (no scaling)",
        "2. Verify the marker sizes after printing",
        "3. Place on the floor in view of the camera",
        "4. Ensure markers are flat and not bent"
    ]
    instr_y = a4_height_px - int(40 * mm_to_pixels)
    for i, line in enumerate(instructions):
        cv2.putText(a4_sheet, line, (margin_px, instr_y + i * int(7 * mm_to_pixels)), font, font_scale, text_color, thickness)
    
    cv2.imwrite(output_path, a4_sheet)
    print(f"A4 sheet with ArUco markers saved to {output_path}")
    
    return a4_sheet

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate A4 sheet with two ArUco markers")
    parser.add_argument("--id1", type=int, default=5, help="First marker ID (default: 5)")
    parser.add_argument("--id2", type=int, default=10, help="Second marker ID (default: 10)")
    parser.add_argument("--dict", type=int, default=cv2.aruco.DICT_6X6_250, help="Dictionary type (default: DICT_6X6_250)")
    parser.add_argument("--size", type=int, default=80, help="Marker size in millimeters (default: 80)")
    parser.add_argument("--dpi", type=int, default=300, help="Output resolution in DPI (default: 300)")
    parser.add_argument("--output", type=str, default="marker.jpg", help="Output file path (default: marker.jpg)")
    
    args = parser.parse_args()
    
    create_a4_with_markers(
        marker1_id=args.id1,
        marker2_id=args.id2,
        dictionary=args.dict,
        marker_size=args.size,
        output_path=args.output,
        dpi=args.dpi
    )
    
    print(f"Generated A4 sheet with ArUco marker IDs {args.id1} and {args.id2}.")
    print("Print this sheet at 100% scale (no scaling).")

if __name__ == "__main__":
    main()
