from PIL import Image
import numpy as np

music_file = Image.open('turkish_final1.png')
music_img = np.asarray(music_file)


def recognizing_the_staves_simple(music_sheet):
    staves = dict()
    imp_music_sheet = np.array(music_sheet)
    simp_staves_locations = []
    #this loop finds the locations of  the assumed staves in the image
    #TODO:this part is so bad  there is so much work to do you idiot
    #TODO first getting rid of the assumption that the staves will be strictly straight lines
    #TODO getting rid of the assumption that the width of the stave will be 1px
    #TODO getting rid of the simplistic calculation for the treshold of the  number of black pixels in stave calculation.The treshold suppose to be according to the start and the endof the staves
    for i in range(len(music_sheet)):
        cur_line = imp_music_sheet[i]
        num_black_pxl = len(np.where(cur_line != 255)[0])
        if num_black_pxl > (3 * len(cur_line))/4.0:
            simp_staves_locations.append(i)
    #grouping staves and filtering
    simp_staves_locations = np.array(simp_staves_locations)
    diff_staves_locations = simp_staves_locations[1:] - simp_staves_locations[:-1]
    staves_grouping_location = [[simp_staves_locations[0]]]
    i = 0
    group_index = 0
    while i < len(diff_staves_locations):
        if diff_staves_locations[i]:
            pass



#welcome git



recognizing_the_staves_simple(music_img)