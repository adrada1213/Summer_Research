import os
import re

class AnalysisModel:
    model_name = ""
    series_dirs = []
    series_nrs = []
    slice_nrs = []
    ed_indexes = []
    es_indexes = []

    def description(self):
        print('----- Model {} -----'.format(self.model_name))
        print(self.series_dirs)
        print('Series: {}'.format(self.series_nrs))
        print('Slices: {}'.format(self.slice_nrs))
        print('Ed indexes: {}'.format(self.ed_indexes))
        print('Es indexes: {}'.format(self.es_indexes))
        print()

# ----------------- functions --------------------
def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.

'''
    Read the .model file, extract the ED and ES frame index using Regex
    Grab the series and slice name
'''
def read_model_file(model_file, system_file_path, patient_name):    
    # read the desc file
    pattern_ed = "(Series(.+?)Slice(.+?))End-diastolic Frame\s*?.*?(\d+)"
    pattern_es = "(Series(.+?)Slice(.+?))End-systolic Frame\s*?.*?(\d+)"
    reg_ed = re.compile(pattern_ed)
    reg_es = re.compile(pattern_es)

    all_lines = []
    # substring and get the model name only
    # model_name = model_file[len(f"{system_file_path}\\{patient_name}."):]
    model_name = model_file[len("{}/{}.".format(system_file_path, patient_name)):]

    #print("Checking model file for index of ED and ES frame....")
    with open(model_file) as myFile:
        all_lines = myFile.read().splitlines()

    # Retrieve the lines containing the patterns
    ed_frames = list(filter(reg_ed.match, all_lines))
    es_frames = list(filter(reg_es.match, all_lines))

    ed_indexes = []
    es_indexes = []
    series_folders = []
    series_numbers = []
    slice_numbers = []
    model = AnalysisModel()
    # Retrieve the index of the ed frames and es frames
    for ed_frame in ed_frames:
        m = reg_ed.match(ed_frame)
        series_folder = m.group(1).replace(" ", "_").rstrip("_")
        series_folders.append(series_folder)
        series_numbers.append(int(m.group(2).strip()))
        slice_numbers.append(int(m.group(3).strip()))
        ed_index = int(m.group(4))
        ed_indexes.append(ed_index)

    for es_frame in es_frames:
        m = reg_es.match(es_frame)
        es_index = int(m.group(4))
        es_indexes.append(es_index)

    model.model_name = model_name
    model.series_nrs = series_numbers
    model.slice_nrs = slice_numbers
    model.ed_indexes = ed_indexes
    model.es_indexes = es_indexes
    model.series_dirs = series_folders
    
    #model.description()

    # print('model.ed_indexes', model.ed_indexes)
    # print('model.es_indexes', model.es_indexes)
    return model

