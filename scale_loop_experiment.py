from PIL import Image, ImageOps
import numpy as np
import cv2
from scipy import spatial, ndimage
import matplotlib.pyplot as plt
import random
NEIGHBOR_REFERENCE = [[-1,-1], [-1,0], [-1,1], [0,-1], [0,1], [1,-1], [1,0], [1,1]]
STD_GAP = 18
class Pixel():
    def __init__(self, x, y):
        self.x = x
        self.y =  y
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

def get_scaling_transformation(estimated_gap,bar_img, width_page):
    #estimated_y_scaling = estimated_gap/STD_GAP
    estimated_x_min_len = np.sqrt(width_page)*(2/3.0)#width_page/50-10
    sol_file = Image.open('images\\sol.png')
    sol_file = ImageOps.grayscale(sol_file)
    matching_array = np.array([[np.max(cv2.matchTemplate(bar_img,
                                         np.asarray(sol_file.resize((int(estimated_x_min_len +(estimated_x_min_len/10.0)*i),
                                                                         int(sol_file.size[1]*(estimated_gap/(STD_GAP+j/3.0-3))))))
                                         , cv2.TM_CCOEFF_NORMED))for j in range(18)]for i in range(20)])
    max_x, max_y = np.where(matching_array == np.max(matching_array))
    width  = estimated_x_min_len + (estimated_x_min_len/10.0)*max_x[0]
    length = sol_file.size[1]*(estimated_gap/(STD_GAP+max_y[0]/3.0-3))
    scale_x = width/sol_file.size[0]
    scale_y  = length/sol_file.size[1]
    return scale_x, scale_y





def test_scale_loop():
    music_file = Image.open('images\\bar3.png')
    music_file = ImageOps.grayscale(music_file)
    music_img = np.asarray(music_file.resize((music_file.size[0] * 2, music_file.size[1] * 2)))

    scale_x, scale_y = get_scaling_transformation(10, music_img, 2640)
    finding_objects_in_bar(music_img, 'images\\note.png', scale_x, scale_y, 0.7)


def connected_components(img):
    pre_coordinates = np.where(img == 0)
    set_coordinates = set()
    components = []
    for i in range(len(pre_coordinates[0])):
        set_coordinates.add((pre_coordinates[0][i], pre_coordinates[1][i]))
    while len(set_coordinates) != 0:
        start_coordinate = [set_coordinates.pop()]
        curent_component = [start_coordinate[0]]
        while len(start_coordinate)!=0:
            neighboring_pixels = []
            for pixel in start_coordinate:
                for i in NEIGHBOR_REFERENCE:
                    neighboring_pixels.append((pixel[0] + i[0], pixel[1] + i[1]))
            start_coordinate = []
            for pixel in neighboring_pixels:
                if pixel in set_coordinates:
                    start_coordinate.append(pixel)
                    curent_component.append(pixel)
                    set_coordinates.remove(pixel)
        components.append(curent_component)
    return components

def finding_objects_in_bar(bar_img, obj_loc, scale_x, scale_y, treshold):
    sol_file = Image.open(obj_loc)
    sol_img = np.asarray(sol_file.resize((int(sol_file.size[0] * scale_x), int(sol_file.size[1] * scale_y))))
    match_image = cv2.matchTemplate(bar_img, sol_img, cv2.TM_CCOEFF_NORMED)
    tresholded_img = []
    for i in match_image:
        tresholded_img.append([])
        for j in i:
            if j > treshold:
                tresholded_img[-1].append(0)
            else:
                tresholded_img[-1].append(255)
    tresholded_img = np.array(tresholded_img)

    component_img = np.array(tresholded_img)
    coordinates = np.where(tresholded_img == 0)
    for i in range(len(coordinates[0])):
        component_img[coordinates[0][i]: coordinates[0][i] + 5, coordinates[1][i] - 3: coordinates[1][i] + 3] = 0
    plt.imshow(component_img)
    plt.show()
    components = connected_components(component_img)
    components_centers = []
    full_components = []
    for component in components:
        x_component = np.array([point[0]for point in component])
        y_component = np.array([point[1]for point in component])
        full_component = np.array([[point[0], point[1]]for point in component])
        components_centers.append([np.average(x_component), np.average(y_component)])
        full_components.append(full_component)
    index_list = np.array(range(len(components_centers)))
    acc_center_y = lambda x : components_centers[x][1]
    sorted_index_list = sorted(index_list, key=acc_center_y)
    components_centers = np.array([components_centers[i]for i in sorted_index_list])
    full_components = [full_components[i] for i in sorted_index_list]



    return components_centers, full_components



#test_scale_loop()


