import math
import random
import numpy as np
import cv2

class PathGenerator():
    def __init__(self, min_dist, max_dist, x_limit, y_limit, n_step):
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.min_step = min_dist / n_step
        self.max_step = max_dist / n_step
        self.start_point = [round(random.uniform(x_limit[0], x_limit[1]), 2), round(random.uniform(y_limit[0], y_limit[1]), 2)]
        self.end_point = self.choose_point_with_radius(self.start_point)
        self.image_size = (224, 224)
        

    def choose_point_with_radius(self, p1):
        angle = random.uniform(0, 2 * math.pi)

        radius = random.uniform(self.min_dist, self.max_dist)

        x = p1[0] + radius * math.cos(angle)
        y = p1[1] + radius * math.sin(angle)

        x = max(min(x, self.x_limit[1]), self.x_limit[0])
        y = max(min(y, self.y_limit[1]), self.y_limit[0])

        return [x, y]
    
    def generate_path(self):
        path = [self.start_point]
        current_point = self.start_point[:]

        while True:
            distance_to_end = self.get_distance(current_point, self.end_point)

            if distance_to_end <= self.max_step:
                path.append(self.end_point)
                break

            step_size = random.uniform(self.min_step, self.max_step)

            angle = random.uniform(0, 2 * math.pi)

            step_x = step_size * math.cos(angle)
            step_y = step_size * math.sin(angle)

            new_point = (current_point[0] + step_x, current_point[1] + step_y)

            new_dist = self.get_distance(new_point, self.end_point)

            if new_dist > distance_to_end:
                continue

            path.append(new_point)
            current_point = new_point

        path = [[round(x, 2), round(y, 2)] for x, y in path]

        return path
    
    @staticmethod
    def get_orientation(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return round(math.atan2(dy, dx), 2)
            
    @staticmethod
    def get_distance(p1, p2):
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def generate_floor_plan(self, path):
        image = np.zeros((self.image_size))
        points = np.array(path[:], dtype=float)
        if self.x_limit[0] < 0:
            points[:, 0] += abs(self.x_limit[0])

        if self.y_limit[0] < 0:
            points[:, 1] += abs(self.y_limit[0])

        points[:, 0] *= round(self.image_size[0] / abs(self.x_limit[1] - self.x_limit[0]), 2)
        points[:, 1] *= round(self.image_size[1] / abs(self.y_limit[1] - self.y_limit[0]), 2)
        points = points.clip(0, [self.image_size[0]-1, self.image_size[1]-1]).astype(int)
        floor_path = []

        s = points[0]
        rect_top_left = s - 1  
        rect_bottom_right = s + 1  
        rect_points = np.array([[rect_top_left[0], rect_top_left[1]],
                                [rect_top_left[0], rect_bottom_right[1]],
                                [rect_bottom_right[0], rect_bottom_right[1]],
                                [rect_bottom_right[0], rect_top_left[1]],
                                [rect_top_left[0], rect_top_left[1]]])
        rect_points = rect_points.clip(0, [self.image_size[0]-1, self.image_size[1]-1]).astype(int)
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
    pg = PathGenerator(8.0, 10.0, [-25, 25], [0, 22], 3)
    goal_trajectory = pg.generate_path()
    image = pg.generate_floor_plan(goal_trajectory)
    display_image(image[:] * 255)