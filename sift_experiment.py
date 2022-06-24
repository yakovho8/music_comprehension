from PIL import Image, ImageOps
import numpy as np
import cv2
from scipy import spatial, ndimage
import matplotlib.pyplot as plt
import random

NUM_MATCHING_POINT = 15
def save_binary_image(loc, new_loc):
    im_gray = cv2.imread(loc, cv2.IMREAD_GRAYSCALE)
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite(new_loc, im_bw)
def find_ellipse_properties(note_img):
    black_coordinates = np.where(note_img == 0)
    center_y = np.average(black_coordinates[0])
    center_x = np.average(black_coordinates[1])
    merged_coordinates = np.array([[black_coordinates[0][i], black_coordinates[1][i]]
                                   for i in range(len(black_coordinates[0]))])
    distance_mat = spatial.distance_matrix(merged_coordinates, merged_coordinates)
    i, j = np.unravel_index(distance_mat.argmax(), distance_mat.shape)
    idk = 0

def count_inline_points(cur_T, points1, points2 , max_dist):
    norm_T = np.abs(np.linalg.det(cur_T[:-1]))
    num_inliers = 0
    cur_T = cur_T.transpose()
    inlier_points1 = []
    inlier_points2 = []
    for i in range(len(points1)):
        T_p = cur_T.dot(points1[i])
        dist_points = np.linalg.norm(T_p - points2[i])
        if dist_points/norm_T <= max_dist:
            num_inliers += 1
            inlier_points1.append(points1[i])
            inlier_points2.append(points2[i])
    return num_inliers, inlier_points1, inlier_points2

def quantize_T(Trans):
    Qt = np.zeros(Trans.shape)
    for row in range(len(Trans)):
        for col in range(len(Trans[row])):
            value = Trans[row][col]
            value = value*1000
            value_r = value - int(value)
            if np.abs(value_r) > 0.5:
                value += 1
            value = float(int(value))/1000.0
            Qt[row][col] = value
    return Qt

def prepare_points(points1, points2, matches):
    all_encompasing_array = [[points1[i], points2[i], matches[i]]for i in range(len(points1))]
    center_x = np.average(points1[:,0])
    center_y = np.average(points1[:,1])
    x_bar = lambda x:np.sqrt((center_x - x[1][0])**2 + (center_y - x[1][1])**2)
    match_dist = lambda x:x[2].distance
    sorted_encompasing_array = sorted(all_encompasing_array, key=x_bar)
    centric_encompassing_array = sorted_encompasing_array[:int(len(sorted_encompasing_array)/3)]
    dist_centric_array = sorted(centric_encompassing_array, key=match_dist)
    return dist_centric_array

def prototype_find_affine_from_sol(stave_g_img):
    note_file = Image.open('sol.png')
    note_img = np.asarray(note_file.resize((note_file.size[0]*4, note_file.size[1]*4)))

    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(note_img, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(stave_g_img, None)

    # feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    # music_ero = ndimage.binary_erosion(music_img,note_img, iterations=100)
    # music_ero = ndimage.binary_dilation(music_img, np.pad(note_img[5:-5,5:-5], 1, constant_values=255))
    notes_points = [keypoints_1[i.queryIdx].pt for i in matches]
    notes_points = np.array([[point[0], point[1], 1] for point in notes_points])
    bar_points = [keypoints_2[i.trainIdx].pt for i in matches]
    bar_points = np.array([[point[0], point[1]] for point in bar_points])
    all_points = prepare_points(notes_points, bar_points, matches)
    notes_points = np.array([[j for j in i[0]] for i in all_points])
    bar_points = np.array([[j for j in i[1]] for i in all_points])
    matches = np.array([i[2] for i in all_points])
    num_max_points = np.min([len(bar_points),NUM_MATCHING_POINT])
    notes_points, bar_points, matches = notes_points[:num_max_points], bar_points[:num_max_points],\
                                        matches[:num_max_points]

    for i in range(4*len(notes_points)):
        rand_ind = []
        rand_note_points = []
        rand_bar_points = []
        for i in range(3):
            rand_ind.append(random.randint(0,len(notes_points)-1))
            rand_note_points.append(notes_points[rand_ind[-1]])
            rand_bar_points.append(bar_points[rand_ind[-1]])
        cur_T = np.linalg.lstsq(np.array(rand_note_points), np.array(rand_bar_points))[0]
        num_inliers,  inliers_note_points ,inliers_bar_points = count_inline_points(cur_T, notes_points, bar_points, 10)
        if num_inliers >= NUM_MATCHING_POINT*0.6:
            Transform = np.linalg.lstsq(np.array(inliers_note_points), np.array(inliers_bar_points))[0]
            img3 = cv2.drawMatches(note_img, keypoints_1, stave_g_img, keypoints_2, matches[:num_max_points], stave_g_img, flags=2)
            plt.imshow(img3), plt.show()
            return note_img, Transform[:-1]
    img3 = cv2.drawMatches(note_img, keypoints_1, stave_g_img, keypoints_2, matches[:NUM_MATCHING_POINT], stave_g_img, flags=2)
    plt.imshow(img3), plt.show()
    return note_img, None



def test_sift_experiment():
    music_file = Image.open('binary_bar3.png')
    music_file = ImageOps.grayscale(music_file)
    music_img = np.asarray(music_file)
    note_img, T_aff = prototype_find_affine_from_sol(music_img)
    T_aff = quantize_T(T_aff)
    idk = cv2.matchTemplate(music_img, note_img, cv2.TM_CCOEFF_NORMED)
    idk_new = []
    for i in idk:
        idk_new.append([])
        for j in i:
            if j > 0.6:
                idk_new[-1].append(0)
            else:
                idk_new[-1].append(255)
    idk_new = np.array(idk_new)
    idk_new_new = np.array(idk_new)
    coordinates = np.where(idk_new == 0)
    for i in range(len(coordinates[0])):
        idk_new_new[coordinates[0][i] - 5: coordinates[0][i] + 5, coordinates[1][i] - 3: coordinates[1][i] + 3] = 0
    plt.imshow(idk_new_new)
    plt.show()

    # cv2.ellipse()
    # bla = Image.fromarray(idk_new_new)
    # bla.save('eroded_image.png')

    properties = find_ellipse_properties(note_img)

    # properties: center - (12,10),  main_axis = between(1,17) and (23,3) length 26,
    # minor axis: between (17,18), (8,4) length 17.8(18)
    # degree 32.5
