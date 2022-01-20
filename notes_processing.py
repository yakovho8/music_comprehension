from PIL import Image
import numpy as np

music_file = Image.open('turkish_final1.png')
music_img = np.asarray(music_file)

class Stave_Group:
    def __init__(self,fa, re, si, sol, mi,next_stave):
        self. mi = mi
        self.sol = sol
        self.si = si
        self.re = re
        self.fa = fa
        self.next_stave = next_stave
        self.next_stave_end = None
        self.next_stave_diff = None
        self.real_start = None
        if self.next_stave is not None:
            self.next_stave_end = next_stave.fa
            self.next_stave_diff = self.next_stave_end - self.mi
            self.real_start = mi + self.next_stave_diff / 2.0
        self.pre_stave = None
        self.pre_stave_diff = None
        self.real_end = None

    def in_stave(self, loc):
        return self.real_end <= loc <= self.real_start

    def configure_prev(self, prev_stave):
        self.pre_stave = prev_stave
        self.pre_stave_diff = self.fa - self.pre_stave.mi
        self.real_end = self.pre_stave.mi - self.pre_stave_diff/2.0
        return

def recognizing_the_staves_simple(music_sheet):
    staves = dict()
    imp_music_sheet = np.array(music_sheet)
    simp_staves_locations = []
    #this loop finds the locations of  the assumed staves in the image
    #TODO:this part is so bad  there is so much work to do you idiot
    #TODO first getting rid of the assumption that the staves will be strictly straight lines
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
        cur_diff = diff_staves_locations[i]
        if cur_diff <= 2:
            i += 1
            continue
        if cur_diff > ((len(music_sheet)/128.0) + 1):
            staves_grouping_location.append([])
            group_index += 1
        staves_grouping_location[group_index].append(simp_staves_locations[i + 1])
        i += 1
    inv_stave_group = staves_grouping_location[::-1]
    inv_stave_group = [i for i in inv_stave_group if len(i) == 5]
    staves = []
    for i in range(len(inv_stave_group)):
        next_stave = None
        if i != 0:
            next_stave = staves[i-1]
        cur_group = inv_stave_group[i]
        staves.append(Stave_Group(cur_group[0], cur_group[1], cur_group[2], cur_group[3], cur_group[4], next_stave))
        if i != 0:
            staves[i-1].configure_prev(staves[i])

    return staves[::-1]




#welcome git
#TODO add general class for the music reader
#TODO prepare binary search based on sorted stave list
#TODO add a better way of undarstanding to what stave each note belong


recognizing_the_staves_simple(music_img)