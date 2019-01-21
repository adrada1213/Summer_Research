import h5py
import os 
import numpy as np

class DataMappingSet:
    patient_names = []
    filepaths = []
    indexes = []
    
    def __init__(self, pats, files, idx):
        self.patient_names = pats
        self.filepaths = files
        self.indexes = idx

    def count(self):
        return len(self.patient_names)

def save_dataset_mapping(save_path, dataset_dict, filename='dataset_mapping.h5'):
    print('Saving dataset mapping because case was randomized...')
    # Workaround!! hd5 have problem in saving string
    string_dt = h5py.special_dtype(vlen=str)
    with h5py.File(os.path.join(save_path, filename), 'w') as hf:
        for key in dataset_dict:
            # print(key)
            g1 = hf.create_group(key)
            # print(dataset_dict[key].filepaths)
            g1.create_dataset('datapath',data=np.array(dataset_dict[key].filepaths, dtype=object), dtype=string_dt)
            g1.create_dataset('patients',data=np.array(dataset_dict[key].patient_names, dtype=object), dtype=string_dt)
            g1.create_dataset('indexes',data=dataset_dict[key].indexes)
    print('Dataset mapping has been saved in directory', save_path)

def build_filemap(filepath, input_patterns):
    all_files = []
    all_patients = []
    indexes = []

    files  = os.listdir(filepath)
    for pattern in input_patterns:
        filenames =  [f for f in files if pattern in f]
        print('Found {} files for pattern {}'.format(len(filenames), pattern))

        for num, filename in enumerate(filenames):
            path = os.path.join(filepath, filename)
            # a bit overhead here cause we need to open all the files and count
            with h5py.File( path, mode = 'r' ) as hdf5:
                length = len(hdf5['ed_imgs'])
                temp_pats = np.array(hdf5['patients'])

            temp_idx =  np.arange(length)
            temp_names = [path]*length


            all_files.extend(temp_names)
            all_patients.extend(temp_pats)
            indexes.extend(temp_idx)
    return np.array(all_files), np.array(all_patients), np.array(indexes)
    # return all_files, all_patients, all_feature_files, indexes

def get_unique_test_cases(filepath, map_filename, group_name='test'):
    with h5py.File(os.path.join(filepath,map_filename), 'r') as hl:
        # fnames = np.array(hl.get("{}/datapath".format(group_name)))
        # idxs = np.array(hl.get("{}/indexes".format(group_name)))
        cases = np.array(hl.get("{}/patients".format(group_name)))

    return np.unique(cases)

'''
    Randomly split the data BASED ON CASE into 2 sets
    Patients, models, slices, and indexes are the ones
    Returns: 2 sets of indexes (firstset_idx, secondset_idx)
'''
def random_split_dataset_into_2sets(patient_names, firstset_ratio):
    if (firstset_ratio > 1 or firstset_ratio < 0):
        print('Splitting ratio must be between 0 and 1, please adjust the number')
        return None, None
    
    print('Splitting dataset into 2 set, ratio:', firstset_ratio, (1 - firstset_ratio))
    # list all the unique patient (case) names
    unique_patients = np.unique(patient_names)
    #print('unique',unique_patients)
    print('Total nr of images:',len(patient_names))
    print('Nr of unique case name:',len(unique_patients))

    # shuffle the patient names
    np.random.shuffle(unique_patients)
    #print('random',unique_patients)

    nr_of_first = int(firstset_ratio * len(unique_patients))
    nr_of_second = len(unique_patients) - nr_of_first
    print('First-set case:', nr_of_first)
    print('Second-set case:', nr_of_second)

    firstset_cases = unique_patients[0:nr_of_first]
    secondset_cases = unique_patients[nr_of_first:]

    # check if the patient name "isin" the patients, and return the index ("where" clause)
    firstset_idx = np.where(np.isin(patient_names, firstset_cases))
    secondset_idx = np.where(np.isin(patient_names, secondset_cases))
    
    return firstset_idx, secondset_idx


def split_dataset_randomly(filepath, input_patterns, testdata_mapfile, test_ratio, train_ratio):
    all_files, all_patients, indexes = build_filemap(filepath, input_patterns)

    # Check first is test data mapping file is there or not 
    if os.path.isfile(os.path.join(filepath, testdata_mapfile)):
        print('Test data mapping found')
        # grab the test case names from existing file
        test_case_names = get_unique_test_cases(filepath, testdata_mapfile)
        # search the case names in the dataset being loaded, return indexes
        test_idx = np.where(np.isin(all_patients, test_case_names))
        test_set = DataMappingSet(all_patients[test_idx], all_files[test_idx], indexes[test_idx])
        print('Nr of unique case name:', len(test_case_names))
        print('Nr of images:', test_set.count())

        # remove the indexes from the dataset
        filtered_files = np.delete(all_files,test_idx)
        filtered_patients = np.delete(all_patients,test_idx)
        filtered_indexes = np.delete(indexes,test_idx)
    else:
        print('No test data mapping found')
        # no mapping file, we need to split to TEST data first
        test_idx, filtered_idx =  random_split_dataset_into_2sets(all_patients, test_ratio)

        # prepare to save it
        test_set = DataMappingSet(all_patients[test_idx], all_files[test_idx], indexes[test_idx])

        filtered_files = all_files[filtered_idx]
        filtered_patients = all_patients[filtered_idx]
        filtered_indexes = indexes[filtered_idx]


    # Now we have the data for training purpose, split into 80% train, 20% val
    # train_idx, val_idx, test_idx = random_split_training_validation_set_into_indexes(all_patients) # do the random split based on case
    print('\n-------------------------------------------')
    print('Splitting into training and validation set...')
    train_idx, val_idx =  random_split_dataset_into_2sets(filtered_patients, train_ratio)

    train_set = DataMappingSet(filtered_patients[train_idx], filtered_files[train_idx], filtered_indexes[train_idx])
    validation_set = DataMappingSet(filtered_patients[val_idx], filtered_files[val_idx], filtered_indexes[val_idx])

    # test_set = DataMappingSet(all_patients[test_idx], all_files[test_idx], indexes[test_idx])
    print('\n-------------------------------------------')
    print('Total training images', train_set.count())
    print('Total validation images', validation_set.count())
    print('Total test images', test_set.count())

    # create dict
    dataset_dict = {'train': train_set, 'validate': validation_set, 'test': test_set}
    # prepare a dictionary to be saved later
    test_dict = { 'test': test_set }
    # return train_idx, val_idx, test_dict
    return dataset_dict, test_dict

'''
    OBSOLETE
    
    Randomly split the data BASED ON CASE into training, validation set, and test set (default: 80%, 10%, 10%)
    Patients, models, slices, and indexes are the ones
    Returns: 3 sets of indexes (training_idx, validation_idx, test idx)
'''
def random_split_training_validation_set_into_indexes(all_patients, train_ratio=0.8, val_ratio=0.1):
    if (train_ratio + val_ratio > 1):
        print('Train and validation set ratio exceeds 100%, please adjust the number')
        return None, None, None

    print('Splitting dataset into train/val/test, ratio:', train_ratio, val_ratio, (1 - train_ratio - val_ratio))
    # list all the unique patient (case) names
    unique_patients = np.unique(all_patients)
    #print('unique',unique_patients)
    print('Total nr of images:',len(all_patients))
    print('Nr of unique case name:',len(unique_patients))

    # shuffle the patient names
    np.random.shuffle(unique_patients)
    #print('random',unique_patients)

    nr_of_training = int(train_ratio * len(unique_patients))
    nr_of_validation = int(val_ratio * len(unique_patients))
    print('training case:', nr_of_training)
    print('validation case:', nr_of_validation)

    training_patients = unique_patients[0:nr_of_training]
    validation_patients = unique_patients[nr_of_training:nr_of_training+nr_of_validation]
    test_patients = unique_patients[nr_of_training+nr_of_validation:]

    # check if the patient name "isin" the patients, and return the index ("where" clause)
    training_idx = np.where(np.isin(all_patients, training_patients))
    validation_idx = np.where(np.isin(all_patients, validation_patients))
    test_idx = np.where(np.isin(all_patients, test_patients))
    
    return training_idx, validation_idx, test_idx