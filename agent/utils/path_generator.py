import math
import random
import numpy as np
import matplotlib.pyplot as plt


class PathGenerator():
    def __init__(self, min_dist, max_dist, x_limit, y_limit, n_step):
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.min_step = min_dist / n_step
        self.max_step = max_dist / n_step
        self.start_point = [round(random.uniform(x_limit[0]+3, x_limit[1]-3), 2), round(random.uniform(y_limit[0]+3, y_limit[1]-3), 2)]
        self.end_point = self.choose_point_with_radius(self.start_point)
        self.image_size = (112, 112)
        self.angle_radius_var = 30
        

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

            target_angle = math.degrees(self.get_orientation(current_point, self.end_point))

            angle = math.radians(random.uniform(target_angle - self.angle_radius_var, target_angle + self.angle_radius_var))

            step_x = step_size * math.cos(angle)
            step_y = step_size * math.sin(angle)

            if 0.5 < random.uniform(0, 1):
                direction = 1
            else:
                direction = -1

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
        image = np.zeros(self.image_size)
        points = np.array(path[:], dtype=float)
        if self.x_limit[0] < 0:
            points[:, 0] += abs(self.x_limit[0])

        if self.y_limit[0] < 0:
            points[:, 1] += abs(self.y_limit[0])

        points[:, 0] *= round(self.image_size[0] / abs(self.x_limit[1] - self.x_limit[0]), 2)
        points[:, 1] *= round(self.image_size[1] / abs(self.y_limit[1] - self.y_limit[0]), 2)
        points = points.clip(0, [self.image_size[0]-1, self.image_size[1]-1]).astype(int)
        floor_path = []

    
        for i in range(len(points) - 1):
            p1, p2 = points[i:i+2]
            num_points = max(abs(p1[0] - p2[0]), abs(p2[1] - p1[1])) + 1
            pp = np.linspace(p1, p2, num_points)
            pp = np.round(pp).astype(int)
            floor_path.extend(pp)
        
        floor_path = np.array(floor_path)
        image[floor_path[:, 0], floor_path[:,1]] = [1]

        return image
    

def rotate_point(point, center, angle):
    angle_rad = np.deg2rad(angle)
    x_new = center[0] + (point[0] - center[0]) * np.cos(angle_rad) - (point[1] - center[1]) * np.sin(angle_rad)
    y_new = center[1] + (point[0] - center[0]) * np.sin(angle_rad) + (point[1] - center[1]) * np.cos(angle_rad)
    return np.array([x_new, y_new])

def add_point_to_floor_plan(image, point, angle):
    color = [1]

    x_limit, y_limit = [-25, 25], [0, 22]

    if x_limit[0] < 0:
        point[0] += abs(x_limit[0])

    if y_limit[0] < 0:
            point[1] += abs(y_limit[0])
            
    point[0] = round(point[0] * image.shape[0] / abs(x_limit[1] - x_limit[0]), 2)
    point[1] = round(point[1] * image.shape[1] / abs(y_limit[1] - y_limit[0]), 2)

    angle = math.degrees(angle)
    rect_angle = round((angle + 360) % 360)
    rect_size = 2  

    rect_center = np.array(point)
    rect_top_left = rotate_point(rect_center + [-rect_size / 2, -rect_size / 2], rect_center, rect_angle)
    rect_bottom_right = rotate_point(rect_center + [rect_size / 2, rect_size / 2], rect_center, rect_angle)


    # Ensure rectangle coordinates are within image bounds
    rect_top_left = np.clip(rect_top_left, [0, 0], [image.shape[0]-1, image.shape[1]-1]).astype(int)
    rect_bottom_right = np.clip(rect_bottom_right, [0, 0], [image.shape[0]-1, image.shape[1]-1]).astype(int)

    # Fill the rectangle with the specified color
    new_image = image[:]

    s1, e1 = sorted([rect_top_left[0], rect_bottom_right[0]])
    s2, e2 = sorted([rect_top_left[1], rect_bottom_right[1]])
    new_image[s1:e1+4, s2:e2+4] = color

    return new_image

def display_image(image):
    if image is None:
        print("Error: Unable to read the image.")
        return
    # image = np.transpose(image, (1, 2, 0))
    # Display the image using matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.title('Floor plan')
    plt.show()


if __name__ == "__main__":
    x_limit = [-25, 25]
    y_limit = [0, 22]
    pg = PathGenerator(25.0, 35.0, x_limit, y_limit, 3)
    goal_trajectory = pg.generate_path()
    image = pg.generate_floor_plan(goal_trajectory)

    point = goal_trajectory[0]
    if x_limit[0] < 0:
        point[0] += abs(x_limit[0])

    if y_limit[0] < 0:
            point[1] += abs(y_limit[0])
    point[0] = round(point[0] * image.shape[0] / abs(x_limit[1] - x_limit[0]), 2)
    point[1] = round(point[1] * image.shape[1] / abs(y_limit[1] - y_limit[0]), 2)
    angle = pg.get_orientation(goal_trajectory[0], goal_trajectory[1])
    angle = math.degrees(angle)
    angle = round((angle + 360) % 360)
    image = add_point_to_floor_plan(image, point, angle)
    display_image(image)
    print("completed")