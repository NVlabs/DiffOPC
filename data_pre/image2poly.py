import cv2
import numpy as np
from skimage import measure


def get_polygons_from_target(target_path):
    # Read the binary image
    img_path = target_path
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Find contours in the binary image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a list to store the polygon sets
    polygon_sets = []
    polygon_strs = []
    # Iterate over each contour
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # Convert the polygon points to a list of tuples
        polygon = [tuple(point[0]) for point in approx]
        poly_str = [f"{int(point[0][0])} {int(point[0][1])} " for point in approx]
        poly_str = "".join(poly_str)
        # Add the polygon to the polygon sets
        polygon_sets.append(polygon)
        polygon_strs.append(poly_str)

    # Print the polygon sets
    # for polygon_set in polygon_sets:
    # print(polygon_set)

    # print("\n")
    tmp_start = """BEGIN     /* GL1TOGULP CALLED ON FRI MAY 17 11:33:25 2013 */
EQUIV  1  1000  MICRON  +X,+Y
CNAME Temp_Top
LEVEL M1

CELL Temp_Top PRIME
"""

    # print(polygon_strs)
    for poly_str in polygon_strs:
        tmp_start += f"   PGON N M1 {poly_str.strip()}\n"

    tmp_start += "ENDMSG"

    return tmp_start


if __name__ == "__main__":
    save_dir = '/home/local/eda13/gc29434/phd/intern/DiffOPC/benchmark/ICCAD2013_large'
    for i in range(1, 11):
        tid = i + 10
        target_path = f"/home/local/eda13/gc29434/phd/intern/DiffOPC/benchmark/baseline/multi-large/target/t{tid}_0_mask.png"
        tmp = get_polygons_from_target(target_path)
        # print(tmp)
        save_path = f"{save_dir}/M1_test{i}.glp"
        with open(save_path, 'w') as f:
            f.write(tmp)
