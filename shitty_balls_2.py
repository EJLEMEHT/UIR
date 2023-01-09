import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image


def normalize(vector):
    return vector / np.linalg.norm(vector)


def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis


def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None


def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(obj['center'], obj['radius'], ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance


width, height = 300, 200
max_depth = 1
camera = np.array([0, 0, 1])
ratio = float(width) / height
screen = (-1, 1 / ratio, 1, -1 / ratio)  # слева, верх, справа, низ

objects = [
    {'center': np.array([-0.7, 0, 0]), 'radius': 0.3, 'reflection': 0.01, 'light': False},
    {'center': np.array([0.0, 0, 0]), 'radius': 0.2, 'reflection': 0.01, 'light': True},
    {'center': np.array([0.7, 0, 0]), 'radius': 0.3, 'reflection': 0.01, 'light': False},
    {'center': np.array([0, -9000, 0]), 'radius': 9000 - 0.4, 'reflection': 0.5, 'light': False}]

image = np.zeros((height, width, 3))
for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
    for j, x in enumerate(np.linspace(screen[0], screen[2], width)):

        # экран в начальной точке

        pixel = np.array([x, y, 0])
        origin = camera
        direction = normalize(pixel - origin)

        color = np.array([1., 1., 1.])
        reflection = 1

        for k in range(max_depth):

            # проверка пересечений

            nearest_object, min_distance = nearest_intersected_object(objects, origin, direction)
            if nearest_object is None:
                break

            intersection = origin + min_distance * direction
            normal_to_surface = normalize(intersection - nearest_object['center'])
            shifted_point = intersection + 1e-5 * normal_to_surface
            intersection_to_light = normalize(light['position'] - shifted_point)

            _, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light)

            intersection_to_light_distance = np.linalg.norm(light['position'] - intersection)
            is_shadowed = min_distance < intersection_to_light_distance

            if is_shadowed:
                color = np.array([0., 0., 0.])

            if nearest_object['light']:
                break

            illumination = np.zeros((3))

            # ambiant

            illumination += nearest_object['ambient']

            # diffuse

            # illumination += nearest_object['ambient'] * deffuse_coef * np.dot(intersection_to_light, normal_to_surface)

            # specular

            intersection_to_camera = normalize(camera - intersection)
            H = normalize(intersection_to_light + intersection_to_camera)
            # illumination += nearest_object['specular'] * light['specular'] * np.dot(normal_to_surface, H) ** (nearest_object['shininess'] / 4)

            # reflection

            color *= reflection * illumination
            reflection *= nearest_object['reflection']
            origin = shifted_point
            direction = reflected(direction, normal_to_surface)
        image[i, j] = np.clip(color, 0, 1)
    print(f'{"%.1f" % (i/height * 100)}% complete')
plt.imsave('shitty_balls.png', image)
print(time.time() - start)

