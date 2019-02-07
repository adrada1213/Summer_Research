import h5py
import os 
import numpy as np

class DataMappingSet:
    filepaths = []
    groups = []
    indices = []
    es_indices = []
    
    def __init__(self, filepaths, groups, indices, es_indices):
        self.groups = groups
        self.filepaths = filepaths
        self.indices = indices
        self.es_indices = es_indices

    def count(self):
        return len(self.filepaths)

def save_dataset_mapping(save_path, dataset_dict, filename='dataset_mapping.h5'):
    print('Saving dataset mapping because case was randomized...')
    # Workaround!! hd5 have problem in saving string
    string_dt = h5py.special_dtype(vlen=str)
    with h5py.File(os.path.join(save_path, filename), 'w') as hf:
        for key in dataset_dict:
            # print(key)
            g1 = hf.create_group(key)
            # print(dataset_dict[key].filepaths)
            g1.create_dataset('filepaths',data=np.array(dataset_dict[key].filepaths, dtype=object), dtype=string_dt)
            g1.create_dataset('groups',data=np.array(dataset_dict[key].groups, dtype=object), dtype=string_dt)
            g1.create_dataset('indices',data=dataset_dict[key].indices)
    print('Dataset mapping has been saved in directory', save_path)

def load_group(group, h5_file):
    with h5py.File(h5_file, 'r') as hf:
        grp = hf["//{}".format(group)]
        grp_cine = grp["//cine"]
        # load the es indices
        es_indices = grp_cine.get("es_indices")
    
    # number of patients should be equal to the number of es indices
    length = len(es_indices)

    # create an array of indices (for h5 file mapping)
    indices = np.arange(length)

    # find the indices of the es_indices with a value of -1
    neg_es = np.argwhere(es_indices<0)

    # delete the indices of those es_indices (with value of -1) from the array of indices (lol)
    indices = np.delete(indices, neg_es)
    es_indices = np.delete(es_indices, neg_es)

    # calculate new length
    new_length = len(indices)

    # create array of filepaths and groups (for h5 file mapping)
    filepaths = [h5_file]*new_length
    groups = [group]*new_length

    return np.array(filepaths), np.array(groups), indices, es_indices

def load_all_datasets(filepath, h5_filename):
    h5_file = os.path.join(filepath, h5_filename)

    train_filepaths, train_groups, train_indices = load_group("train", h5_file)
    test_filepaths, test_groups, test_indices = load_group("test", h5_file)
    val_filepaths, val_groups, val_indices = load_group("validation", h5_file)


    train_set = DataMappingSet(train_filepaths, train_groups, train_indices)
    val_set = DataMappingSet(val_filepaths, val_groups, val_indices)
    test_set = DataMappingSet(test_filepaths, test_groups, test_indices)

    print('\n-------------------------------------------')
    print('Total training images', train_set.count())
    print('Total validation images', val_set.count())
    print('Total test images', test_set.count())

    train_set = DataMappingSet(train_filepaths, train_groups, train_indices)
    val_set = DataMappingSet(val_filepaths, val_groups, val_indices)
    test_set = DataMappingSet(test_filepaths, test_groups, test_indices)

    # create dict
    dataset_dict = {'train': train_set, 'validate': val_set, 'test': test_set}
    # prepare a dictionary to be saved later
    test_dict = { 'test': test_set }

    return dataset_dict, test_dict
