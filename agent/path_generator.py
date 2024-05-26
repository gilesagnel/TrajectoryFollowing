import math
import random
import numpy as np
import cv2

def method1(path, max_distance, min_distance):
    dim = random.randint(0, 1)
    new_point = path[-1][:]
    new_point[dim] += random.uniform(min_distance, min(min_distance, max_distance))
    new_point = [round(x, 2) for x in new_point]
    path.append(new_point)
    return path

def limit_point(point, x_range, y_range):
    limited_x = max(x_range[0], min(point[0], x_range[1]))
    limited_y = max(y_range[0], min(point[1], y_range[1]))
    
    return [limited_x, limited_y]

def generate_path(max_distance, min_distance, steps=1):
    limits = [(-25, 25), (0, 22)]  
    path = [[round(random.uniform(p[0], p[1]), 2) for p in limits]]
    distance = 0
    j = steps
    while distance < max_distance or j > 0:
        method1(path, int(max_distance/steps), min_distance)
        path[-1] = limit_point(path[-1], limits[0], limits[1])
        if min_distance > get_distance(path[-1], path[-2]):
            path.pop()
            continue
        distance = get_distance(path[0], path[-1])
        j -= 1
    remove_redundant_point(path)
    return path


def remove_redundant_point(path):
    i = 0
    while i < len(path) - 2:
        if path[i][0] == path[i + 1][0] and path[i][0] == path[i + 2][0]:  
            del path[i + 1]
        elif path[i][1] == path[i + 1][1] and path[i][1] == path[i + 2][1]:  
            del path[i + 1]
        else:
            i += 1

def get_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def get_orientation(start, end):
    x_dist = abs(end[0] - start[0])
    y_dist = abs(end[1] - start[1])
    if x_dist > y_dist:
        if end[0] > start[0]:
            return 0.0
        else:
            return 3.12
    else:
        if end[1] > start[1]:
            return 1.57
        else:
            return -1.57

def generate_floor_plan(path, image_size=(224, 224)):
    image = np.zeros((image_size))
    points = np.array(path[:])
    points[:, 0] += 25
    points[:, 0] *= 4.48
    points[:, 1] *= 10.18 
    points = points.clip(0, [image_size[0]-1, image_size[1]-1]).astype(int)
    floor_path = []

    s = points[0]
    rect_top_left = s - 1  
    rect_bottom_right = s + 1  
    rect_points = np.array([[rect_top_left[0], rect_top_left[1]],
                            [rect_top_left[0], rect_bottom_right[1]],
                            [rect_bottom_right[0], rect_bottom_right[1]],
                            [rect_bottom_right[0], rect_top_left[1]],
                            [rect_top_left[0], rect_top_left[1]]])
    rect_points = rect_points.clip(0, [image_size[0]-1, image_size[1]-1]).astype(int)
    floor_path.extend(rect_points)

    for i in range(len(points) - 1):
        p1, p2 = points[i:i+2]
        num_points = max(abs(p1[0] - p2[0]), abs(p2[1] - p1[1])) + 1
        pp = np.linspace(p1, p2, num_points)
        pp = np.round(pp).astype(int)
        floor_path.extend(pp)
    
    floor_path = np.array(floor_path, dtype=int)
    image[floor_path[:, 0], floor_path[:,1]] = 1

    return np.expand_dims(image, axis=0)

def display_image(image):
    if image is None:
        print("Error: Unable to read the image.")
        return
    image = np.transpose(image, (1, 2, 0))
    image = cv2.resize(image, (800, 600))
    cv2.imshow('Floor plan', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    goal_trajectory = generate_path(4.0, 2.0, 1)
    image = generate_floor_plan(goal_trajectory)
    display_image(image[:] * 255)