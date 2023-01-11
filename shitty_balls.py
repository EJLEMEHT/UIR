import numpy as np
import matplotlib.pyplot as plt
import time


def normalize(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    dp = v1.dot(v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.arccos(dp / norms)


def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis


def refracted(vector, norm, n1, n2):
    n = n1 / n2
    cosI = -np.dot(norm, vector)
    sinT2 = n ** 2 * (1 - cosI ** 2)
    cosT = np.sqrt(1 - sinT2)
    # if cosI < 0.25:
    #     return reflected(vector, norm)
    return n * vector + (n * cosI - cosT) * norm


def refracted2(vector, norm, n1, n2):
    n = n2/n1
    return n*(vector - np.dot(norm, vector) * norm) + np.sqrt(1 - n ** 2 * (1 - np.dot(norm, vector) ** 2)) * norm


def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            # returns min distance to intersection
            return min(t1, t2)
    return None


def glass_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        # returns cords of intersection
        if max(t1, t2) > 0:
            return ray_origin + max(t1, t2) * ray_direction
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


def sphere_trace(origin, direction, min_distance, sphere, light, color, reflection):
    intersection = origin + min_distance * direction
    normal_to_surface = normalize(intersection - sphere['center'])
    shifted_point = intersection + 1e-5 * normal_to_surface
    intersection_to_light = normalize(light['position'] - shifted_point)

    _, min_distance = nearest_intersected_object(objects, origin, intersection_to_light)

    intersection_to_light_distance = np.linalg.norm(light['position'] - intersection)
    is_shadowed = min_distance < intersection_to_light_distance

    # if is_shadowed:
    #     break

    illumination = np.zeros(3)

    # ambiant
    illumination += sphere['ambient'] * light['ambient']
    # diffuse
    illumination += sphere['ambient'] * light['ambient'] * deffuse_coef * np.dot(
        intersection_to_light, normal_to_surface)

    # specular

    intersection_to_camera = normalize(camera - intersection)
    H = normalize(intersection_to_light + intersection_to_camera)
    illumination += sphere['specular'] * light['specular'] * np.dot(normal_to_surface, H) ** (
                sphere['shininess'] / 4)

    # reflection
    if sphere['radius'] > 5000 and (round(intersection[0]) % 2 == 0 and round(intersection[2]) % 2 != 0
                                            or round(intersection[0]) % 2 != 0 and round(intersection[2]) % 2 == 0):
        color += reflection * illumination * np.array([0.01, 0.01, 0.01])
    else:
        color += reflection * illumination

    reflection *= sphere['reflection'] * 0.5
    origin = shifted_point
    direction = reflected(direction, normal_to_surface)

    return origin, direction, color, reflection


def glass_trace(origin, direction, min_distance, glass, light, color, reflection):
    intersection = origin + min_distance * direction
    normal_to_surface = normalize(intersection - glass['center'])
    shifted_point = intersection + 1e-5 * normal_to_surface

    # 1st refraction
    origin = intersection - 1e-5 * normal_to_surface
    direction = refracted(direction, normal_to_surface, 1, 1.5)
    # 2nd refraction
    intersection = glass_intersect(glass['center'], glass['radius'], origin, direction)
    normal_to_surface = normalize(intersection - glass['center'])
    origin = intersection + 1e-5 * normal_to_surface
    direction = refracted2(direction, normal_to_surface, 1, 1.5)

    return origin, direction, color, reflection



start = time.time()

width = 300
height = 200
max_depth = 5
deffuse_coef = 7

camera = np.array([0, 0, 1])
ratio = float(width) / height
screen = (-1, 1 / ratio, 1, -1 / ratio)  # screen cords

light = {'position': np.array([1, 3, 4]), 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]) }

objects = [
    {'center': np.array([-0.5, 0, -0.2]), 'radius': 0.3, 'ambient': np.array([0.1, 0., 0]), 'specular': np.array([1, 1, 1]), 'shininess': 200, 'reflection': 0.5, 'type': 'sphere'},
    {'center': np.array([0.5, 2, -5]), 'radius': 2, 'ambient': np.array([0, 0.1, 0.1]), 'specular': np.array([1, 1, 1]), 'shininess': 200, 'reflection': 0.5, 'type': 'sphere'},
    {'center': np.array([0.15, 0.1, 0.6]), 'radius': 0.2, 'ambient': np.array([0., 0, 0.1]), 'specular': np.array([1, 1, 1]), 'shininess': 200, 'reflection': 0.5, 'type': 'glass'},
    {'center': np.array([0.5, 0.3, -0.2]), 'radius': 0.25, 'ambient': np.array([0, 0.1, 0.]), 'specular': np.array([1, 1, 1]), 'shininess': 200, 'reflection': 0.5, 'type': 'sphere'},
    {'center': np.array([0, -9000, 0]), 'radius': 9000 - 0.7, 'ambient': np.array([0.1, 0.1, 0.1]), 'specular': np.array([1, 1, 1]), 'shininess': 200, 'reflection': 0.5, 'type': 'sphere'}
]

image = np.zeros((height, width, 3))
# Проходим по всем пикселям
for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
    for j, x in enumerate(np.linspace(screen[0], screen[2], width)):

        # screen and start vector

        pixel = np.array([x, y, 0])
        origin = camera
        direction = normalize(pixel - origin)

        color = np.zeros(3)
        reflection = 1

        # кол-во переотражений
        for k in range(max_depth):

            # intersection

            nearest_object, min_distance = nearest_intersected_object(objects, origin, direction)
            if nearest_object is None:
                break
            if nearest_object['type'] == 'sphere':
                origin, direction, color, reflection = sphere_trace(origin, direction, min_distance, nearest_object, light, color, reflection)
            if nearest_object['type'] == 'glass':
                origin, direction, color, reflection = glass_trace(origin, direction, min_distance, nearest_object,
                                                                    light, color, reflection)
        image[i, j] = np.clip(color, 0, 1)
    print(f'{"%.1f" % (i/height * 100)}% complete')
plt.imsave('Output.png', image)
print(time.time() - start)
