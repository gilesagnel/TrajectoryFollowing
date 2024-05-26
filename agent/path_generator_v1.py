import math
import random
import numpy as np
import cv2

def choose_new_point(current_point, end, min_distance, min_step):
    dim = random.randint(0, 1)

    if abs(current_point[dim] - end[dim]) == 0.0:
        return 0.0, current_point

    if round(abs(current_point[dim] - end[dim]),1) <= min_distance:
            new_point = current_point[:]
            new_point[dim] = end[dim]
            return min_distance * 2, new_point

    if current_point[dim] < end[dim]:
        direction = 1
    else:
        direction = -1

    dist_to_end = abs(end[dim] - current_point[dim])
    step_size = round(random.uniform(min_distance, dist_to_end), 2)

    mod = round(step_size % min_step, 2)
    if mod != min_step:
        step_size += mod

    step_size = direction * step_size        
    if 0 == dim:
        new_point = (current_point[0] + step_size, current_point[1])
    else:
        new_point = (current_point[0], current_point[1] + step_size)
    
    new_point = bound_point(new_point)
    new_dist = get_distance(current_point, new_point)
    return new_dist, new_point

def generate_path(start, end, min_distance=0.5, min_step=0.5):
    path = [start]
    current_point = start
    while current_point[0] != end[0] or current_point[1] != end[1]:
        new_dist_moved, new_point = choose_new_point(current_point, end, min_distance, min_step)
        tries = 0 
        while new_dist_moved < min_distance:
            tries += 1
            if tries > 50 and len(path) > 1:
                path = path[:-1]
                current_point = path[-1]
                tries = 0
            new_dist_moved, new_point = choose_new_point(current_point, end, min_distance, min_step)
        new_point = [round(x, 2) for x in new_point]
        path.append(new_point)
        current_point = new_point
    return path


def generate_start_end_point(max_distance=1.0, min_distance=0.3, x_limit=(-25, 25), y_limit=(0, 22)):
    start = [round(random.uniform(x_limit[0], x_limit[1]), 2), round(random.uniform(y_limit[0], y_limit[1]), 2)]
    distance = 0
    end = start[:]
    
    while distance < min_distance or distance > max_distance or distance == 0:
        x = round(random.uniform(start[0] + max_distance, start[0] - max_distance), 2)
        y = round(random.uniform(start[1] + max_distance, start[1] - max_distance), 2)
        end = bound_point([x, y])
        distance = get_distance(start, end)
    
    return start, end

def bound_point(point, x_limit=(-25, 25), y_limit=(0, 22)):
    x, y = point
    x = min(max(x_limit[0], x), x_limit[1])
    y = min(max(y_limit[0], y), y_limit[1])
    return [x, y]

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
    start, end = generate_start_end_point(30)
    goal_trajectory = generate_path(start, end)
    image = generate_floor_plan(goal_trajectory)
    print(np.count_nonzero(image))
    display_image(image[:] * 255)