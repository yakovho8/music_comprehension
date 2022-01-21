from PIL import Image
import numpy as np

ON_LINE = 0
UNDER_LINE = 1
ABOVE_LINE = 2

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


def line_to_note(line, line_status):
    if line_status == ON_LINE:
        return line * 2 + 1
    if line_status == ABOVE_LINE:
        return line * 2 + 2
    else:
        return line * 2


class StaveLine:
    def __init__(self, start_location):
        self.lines = [start_location]
        self.min_line = start_location
        self.max_line = start_location

    def add_line(self, new_line):
        self.lines.append(new_line)
        if self.max_line < new_line:
            self.max_line = new_line
        if self.min_line > new_line:
            self.min_line = new_line

    def location_state(self, location):
        if self.min_line <= location <= self.max_line:
            return ON_LINE
        if self.max_line < location:
            return ABOVE_LINE
        return UNDER_LINE


class StaveGroup:
    def __init__(self, stave_group, next_stave):
        self.mi = stave_group[-1]
        self.sol = stave_group[-2]
        self.si = stave_group[-3]
        self.re = stave_group[-4]
        self.fa = stave_group[-5]
        self.stave_group = stave_group
        self.next_stave = next_stave
        self.next_stave_end = None
        self.next_stave_diff = None
        self.real_start = None
        if self.next_stave is not None:
            self.next_stave_end = next_stave.fa
            # self.next_stave_diff = self.next_stave_end - self.mi
            # self.real_start = self.mi + self.next_stave_diff / 2.0
        self.pre_stave = None
        # self.pre_stave_diff = None
        # self.real_end = None

    # def in_stave(self, loc):
    #    return self.real_end <= loc <= self.real_start

    def configure_prev(self, prev_stave):
        self.pre_stave = prev_stave
        # self.pre_stave_diff = self.fa - self.pre_stave.mi
        # self.real_end = self.pre_stave.mi - self.pre_stave_diff / 2.0
        return

    def location_to_note(self, location):
        end = 4
        start = 0
        found_note = False
        while not found_note:
            cur_index = int((end + start) / 2)
            cur_stave = self.stave_group[cur_index]
            location_status = cur_stave.location_state(location)
            if location_status == ON_LINE:
                return line_to_note(cur_index, location_status)
            if location_status == UNDER_LINE:
                if ((end - start) == 0) or (start == cur_index):
                    return line_to_note(cur_index, location_status)
                end = cur_index - 1
            else:
                if (end - start) == 0 or end == cur_index:
                    return line_to_note(cur_index,  location_status)
                start = cur_index + 1




class MusicSheetProcessor:
    def __init__(self, music_sheet):
        self.music_sheet = music_sheet
        self.staves = None

    def recognizing_the_staves_simple(self, music_sheet):
        staves = dict()
        imp_music_sheet = np.array(music_sheet)
        simp_staves_locations = []
        # this loop finds the locations of  the assumed staves in the image
        # TODO:this part is so bad  there is so much work to do you idiot
        # TODO first getting rid of the assumption that the staves will be strictly straight lines
        # TODO: getting rid of the simplistic calculation for the treshold of the  number of black pixels in stave
        #  calculation.The treshold suppose to be according to the start and the endof the staves

        for i in range(len(music_sheet)):
            cur_line = imp_music_sheet[i]
            num_black_pxl = len(np.where(cur_line < 200)[0])
            if num_black_pxl > (3 * len(cur_line)) / 4.0:
                simp_staves_locations.append(i)
        # grouping staves and filtering
        simp_staves_locations = np.array(simp_staves_locations)
        diff_staves_locations = simp_staves_locations[1:] - simp_staves_locations[:-1]
        staves_grouping_location = [[StaveLine(simp_staves_locations[0])]]
        i = 0
        group_index = 0
        # estimated gap is based on the assumption that there are more gaps  between stave lines then
        # between stave groups  more precisely 5 times more.
        # than the most commong gap size will be the gap between two stave lines
        estimated_gap = np.bincount(diff_staves_locations[diff_staves_locations > 2]).argmax() * 1.25
        while i < len(diff_staves_locations):
            cur_diff = diff_staves_locations[i]
            if cur_diff <= 2:
                staves_grouping_location[group_index][-1].add_line(simp_staves_locations[i + 1])
                i += 1
                continue
            if cur_diff > estimated_gap:
                staves_grouping_location.append([])
                group_index += 1
            staves_grouping_location[group_index].append(StaveLine(simp_staves_locations[i + 1]))
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
        idk  = self.staves[-1].location_to_note(3060)
        return staves[::-1]


# TODO prepare binary search based on sorted stave list
# TODO add a better way of undarstanding to what stave each note belong


music_file = Image.open('turkish_1g.png')
music_img = np.asarray(music_file)
music_sheet_processor = MusicSheetProcessor(music_img)
music_sheet_processor.recognizing_the_staves_simple(music_img)
