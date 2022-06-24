from PIL import Image, ImageOps
import numpy as np
import scipy.ndimage as ndimage
import cv2
import scale_loop_experiment


ON_LINE = 0
UNDER_LINE = 1
ABOVE_LINE = 2

IN_GROUP = 0
NEXT_GROUP = 1
PREVIOUS_GROUP = 2

BLACK_TRESHOLD = 240

RE = 0
MI = 1
FA = 2
SOL = 3
LA = 4
SI = 5
DO = 6
H_RE = 7
H_MI = 8
H_FA = 9
H_SOL = 10

NUM_TO_NAME_NOTE = {RE:'re', MI:'mi', FA:'fa', SOL:'sol', LA:'la', SI:'si', DO:'do', H_RE:'h_re', H_MI:'H_MI',
                    H_FA:'H_FA', H_SOL:'H_SOL'}


def line_to_note(line, line_status):
    if line_status == ON_LINE:
        return line * 2 + 1
    if line_status == ABOVE_LINE:
        return line * 2 + 2
    else:
        return line * 2


class StaveLine:
    def __init__(self, start_location, start, end):
        self.lines = [start_location]
        self.min_line = start_location
        self.max_line = start_location
        self.start = start
        self.end = end

    def add_line(self, new_line):
        self.lines.append(new_line)
        if self.max_line < new_line:
            self.max_line = new_line
        if self.min_line > new_line:
            self.min_line = new_line

    def location_state(self, location, estimated_gap):
        # after looking the properties of the note ellipse from  sift experiment we can see that most of the note shape
        #is above the supposed location of the note which means that the center will be above the line
        if self.min_line - estimated_gap/2.0 <= location < self.min_line:
            return ON_LINE
        if self.min_line < location:
            return UNDER_LINE
        return ABOVE_LINE


class BarLine:
    def __init__(self, first_location):
        self.lines = [first_location]
        self.min_line = first_location
        self.max_line = first_location
        self.width = 1

    def add_line(self, new_line):
        self.lines.append(new_line)
        if self.max_line < new_line:
            self.max_line = new_line
        if self.min_line > new_line:
            self.min_line = new_line
        self.width = self.max_line - self.min_line + 1


class StaveGroup:
    def __init__(self, stave_group, next_stave):
        self.prev_stave = None
        self.mi = stave_group[-1]
        self.sol = stave_group[-2]
        self.si = stave_group[-3]
        self.re = stave_group[-4]
        self.fa = stave_group[-5]
        self.stave_group = stave_group[::-1]
        self.next_stave = next_stave
        self.next_stave_end = None
        self.next_stave_diff = None
        self.real_start = None
        self.left = stave_group[-1].start
        self.right = stave_group[-1].end
        # if self.next_stave is not None:
        #    self.next_stave_end = next_stave.fa
        # self.next_stave_diff = self.next_stave_end - self.mi
        # self.real_start = self.mi + self.next_stave_diff / 2.0
        # self.pre_stave_diff = None
        # self.real_end = None
        self.h_stave = self.stave_group[0].max_line
        self.l_stave = self.stave_group[-1].min_line

    # def in_stave(self, loc):
    #    return self.real_end <= loc <= self.real_start

    def configure_prev(self, prev_stave):
        # self.pre_stave_diff = self.fa - self.pre_stave.mi
        # self.real_end = self.pre_stave.mi - self.pre_stave_diff / 2.0
        self.prev_stave = prev_stave

    def location_to_note(self, location, estimated_gap):
        end = 4
        start = 0
        found_note = False
        while not found_note:
            cur_index = int((end + start) / 2)
            cur_stave = self.stave_group[cur_index]
            location_status = cur_stave.location_state(location,estimated_gap)
            if location_status == ON_LINE:
                return line_to_note(cur_index, location_status)
            if location_status == UNDER_LINE:
                if ((end - start) == 0) or (start == cur_index):
                    return line_to_note(cur_index, location_status)
                end = cur_index - 1
            else:
                if (end - start) == 0 or end == cur_index:
                    return line_to_note(cur_index, location_status)
                start = cur_index + 1


    def in_group(self, location):
        if self.h_stave >= location >= self.l_stave:
            return IN_GROUP
        if self.l_stave > location:
            return PREVIOUS_GROUP
        return NEXT_GROUP

class MusicNote:
    def __init__(self, note):
        self.num_note = note
        self.str_note = NUM_TO_NAME_NOTE[note]
        self.strength = None
        self.length = None


class MusicSheetProcessor:
    def __init__(self, music_sheet):
        self.music_sheet = music_sheet
        self.staves = None
        self.staveless_img = None
        self.staves_to_bars = None
        self.bar_less_image = None
        self.ero_img = 0
        self.estimated_gap = 0
        self.scale_x = 0
        self.scale_y = 0
        self.note_flow = []



    def recognizing_the_staves_simple(self, music_sheet):
        staves = dict()
        imp_music_sheet = np.array(music_sheet)
        simp_staves_locations = []
        # this loop finds the locations of  the assumed staves in the image
        # TODO:this part is so bad  there is so much work to do you idiot
        # TODO first getting rid of the assumption that the staves will be strictly straight lines
        # TODO:
        #  check for improvement of the tresholding of the number of black pixel in the stave.
        #  currently it takes in account the fact the stave lines wont start at the beginning of the page
        l_r_simple_staves_locations = []
        for i in range(len(music_sheet)):
            cur_line = imp_music_sheet[i]
            # this part is checking that we wont miss staves only because they dont start at the beginning of the page
            # and end in the end of the page

            white_location = np.where(cur_line > BLACK_TRESHOLD)[0]
            j = 0
            diff_white_location = white_location[1:] - white_location[:-1]
            white_side_l = 0
            if len(white_location) == len(cur_line):
                continue
            if white_location[0] == 0:
                while diff_white_location[j] < 2 and j < len(diff_white_location):
                    j += 1
                white_side_l = white_location[j]
            white_side_r = len(cur_line)
            if white_location[-1] == len(cur_line) - 1:
                j = 1
                while diff_white_location[-j] < 2 and j < len(diff_white_location):
                    j += 1
                white_side_r = len(cur_line) - j
            if white_side_r - white_side_l < 0.33 * len(cur_line):
                continue
            cur_line = cur_line[white_side_l:white_side_r]
            num_black_pxl = len(np.where(cur_line < BLACK_TRESHOLD)[0])
            if num_black_pxl > (len(cur_line) * 0.9):
                simp_staves_locations.append(i)
                l_r_simple_staves_locations.append((white_side_l, white_side_r))
        # grouping staves and filtering
        simp_staves_locations = np.array(simp_staves_locations)
        diff_staves_locations = simp_staves_locations[1:] - simp_staves_locations[:-1]
        staves_grouping_location = [[StaveLine(simp_staves_locations[0], l_r_simple_staves_locations[0][0],
                                               l_r_simple_staves_locations[0][1])]]
        i = 0
        group_index = 0
        # estimated gap is based on the assumption that there are more gaps  between stave lines then
        # between stave groups  more precisely 5 times more.
        # than the most commong gap size will be the gap between two stave lines
        estimated_gap = np.bincount(diff_staves_locations[diff_staves_locations > 2]).argmax() * 1.25
        self.estimated_gap = estimated_gap/1.25
        while i < len(diff_staves_locations):
            cur_diff = diff_staves_locations[i]
            if cur_diff <= 2:
                staves_grouping_location[group_index][-1].add_line(simp_staves_locations[i + 1])
                i += 1
                continue
            if cur_diff > estimated_gap:
                staves_grouping_location.append([])
                group_index += 1
            staves_grouping_location[group_index].append(StaveLine(simp_staves_locations[i + 1],
                                                                   l_r_simple_staves_locations[i + 1][0],
                                                                   l_r_simple_staves_locations[i + 1][1]))
            i += 1
        inv_stave_group = staves_grouping_location[::-1]
        inv_stave_group = [i for i in inv_stave_group if len(i) == 5]
        staves = []
        for i in range(len(inv_stave_group)):
            next_stave = None
            if i != 0:
                next_stave = staves[i - 1]
            cur_group = inv_stave_group[i]
            staves.append(StaveGroup(cur_group, next_stave))
            if i != 0:
                staves[i - 1].configure_prev(staves[i])
        self.staves = staves[::-1]
        # print(self.staves[-1].location_to_note(3020))
        # print(self.location_to_group(2000)[0])
        return staves[::-1]

    def location_to_group(self, location):
        end = len(self.staves) - 1
        start = 0
        found_note = False
        while not found_note:
            cur_index = int((end + start) / 2)
            cur_group = self.staves[cur_index]
            location_status = cur_group.in_group(location)
            if location_status == IN_GROUP:
                return cur_index, cur_group
            if location_status == PREVIOUS_GROUP:
                if ((end - start) == 0) or (start == cur_index):
                    return cur_index, cur_group
                if end - start == 1:
                    return start, cur_group
                end = cur_index - 1
            else:
                if (end - start) == 0 or end == cur_index:
                    return cur_index, cur_group
                if end - start == 1:
                    group_1 = self.staves[start]
                    group_2 = self.staves[end]
                    if location < group_2.l_stave:
                        group_1_gap = location - group_1.h_stave
                        group_2_gap = location - group_2.l_stave
                        if group_1_gap > group_2_gap:
                            return start, group_1
                        return end, group_2
                start = cur_index + 1

    def find_bar_line_of_group(self, stave_group):
        start_stave = stave_group.l_stave
        end_stave = stave_group.h_stave
        music_sheet = self.music_sheet
        left = stave_group.left
        right = stave_group.right
        simple_bar_lines = []
        for i in range(left, right + 1):
            cur_line = music_sheet[start_stave:end_stave + 1, i]
            num_black_pxl = len(np.where(cur_line < BLACK_TRESHOLD)[0])
            if num_black_pxl > (len(cur_line) * 0.95):
                simple_bar_lines.append(i)
        simple_bar_lines = np.array(simple_bar_lines)
        diff_bar_lines = simple_bar_lines[1:] - simple_bar_lines[:-1]
        bar_lines = [BarLine(simple_bar_lines[0])]
        # sorted_diff_bar = sorted(diff_bar_lines[np.where(diff_bar_lines > 2)])
        # estimated_gap = np.average(sorted_diff_bar)
        # i tried to implement filtering of bar lines  using the distance between two bar lines
        # the filtering was supposed to filter note lines that goes through all the staves of of stave group
        # the original idea was to use the average of largest differences in the set (like the average
        # of the three larges distances) and the estimated gap will be about the half of it because if the note line
        # splits the bar one of the gaps around the note line will be less then half of the estimated gap
        # problem arise when you start to consider repeat line bars because their bars might be of abnormal size
        # and the fact that there might be acase where every bar is splited by a note line
        # although it is quite an abnormality
        # so i decided instead of filtering this current stage i will filter the bar lines after i found all the notes
        # by filtering the lines that are near notes
        for i in range(len(diff_bar_lines)):
            cur_diff = diff_bar_lines[i]
            if cur_diff <= 2:
                bar_lines[-1].add_line(simple_bar_lines[i + 1])
                continue

            bar_lines.append(BarLine(simple_bar_lines[i + 1]))
        return bar_lines

    def configure_bar_lines(self):
        if self.staves is None:
            return None
        self.staves_to_bars = []
        for stave in self.staves:
            self.staves_to_bars.append(self.find_bar_line_of_group(stave))
        return self.staves_to_bars

    def remove_all_bar_lines(self):
        self.bar_less_image = np.array(self.music_sheet)
        for i in range(len(self.staves)):
            stave = self.staves[i]
            list_bars = self.staves_to_bars[i]
            l_stave = stave.l_stave
            h_stave = stave.h_stave
            for bar_line in list_bars:
                #self.bar_less_image = remove_bar_line(self.bar_less_image, bar_line, h_stave, l_stave)
                self.bar_less_image[l_stave: h_stave + 1, bar_line.min_line: bar_line.max_line + 1] = 255
        return self.bar_less_image


    def remove_stave_line(self):
        stave_less_image = np.array(self.bar_less_image)
        for stave_group in self.staves:
            for stave in stave_group.stave_group:
                max_stave = stave.max_line
                min_stave = stave.min_line
                for i in range(len(stave_less_image[0])):
                    if stave_less_image[min_stave - 1][i] > 10 and \
                            stave_less_image[max_stave + 1][i] > 10:
                        stave_less_image[min_stave:max_stave + 1, i] = 255


        new_img = []
        for i in stave_less_image:
            new_img.append([])
            for j in i:
                if j:
                    new_img[-1].append(0)
                else:
                    new_img[-1].append(255)



        new_img = Image.fromarray(stave_less_image)
        new_img.save('turkish1_stave_less.png')

    def configure_scaling_transform(self):
        first_stave = self.staves[0]
        first_music_line = self.music_sheet[first_stave.l_stave - int(self.estimated_gap)*3:
                                            first_stave.h_stave + int(self.estimated_gap)*3,
                                            first_stave.left:first_stave.right]

        self.scale_x, self.scale_y = scale_loop_experiment.get_scaling_transformation(self.estimated_gap,
                                                                            first_music_line,
                                                                            len(self.music_sheet[0]))
        return

    def configure_note_flow(self):
        for stave in self.staves:
            self.note_flow.append([])
            cur_music_line = self.music_sheet[stave.l_stave - int(self.estimated_gap) * 3:
                                                stave.h_stave + int(self.estimated_gap) * 3,
                               stave.left:stave.right]
            centers_flow, full_flow = scale_loop_experiment.finding_objects_in_bar(cur_music_line, 'images\\note_imp.png', self.scale_x, self.scale_y, 0.6)
            for center in centers_flow:
                cur_note = stave.location_to_note(stave.l_stave - int(self.estimated_gap)*3 + center[0],self.estimated_gap)
                self.note_flow[-1].append(MusicNote(cur_note))
        return



# TODO: improve how we resolve to which group the note is in when the note is between two groups currently it is
#  resolved by checking to which group it is the closest although it is currently unknown if it is correct
#  TODO:
#   there is currently no way of determining what note the location represent for locations above the group or under.
#   for example for h_la and l_do





music_file = Image.open('music_sheets\\turkish_1.jpg')
music_file = ImageOps.grayscale(music_file)
music_img = np.asarray(music_file)
music_sheet_processor = MusicSheetProcessor(music_img)
music_sheet_processor.recognizing_the_staves_simple(music_img)
music_sheet_processor.configure_scaling_transform()
music_sheet_processor.configure_note_flow()
