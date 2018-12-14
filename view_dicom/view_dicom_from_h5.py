import numpy as np
import h5py
import matplotlib.pyplot as plt

def evaluate_on_hdf5( hdf5_filename: str ):
    
    print('Running evaluate_on_hdf5')
    #Open and read from hdf5 file
    hdf5 = h5py.File( hdf5_filename, mode = 'r' )
    #Get ALL of LVSC
    if 'train' in hdf5:
        filenames_train = hdf5['train']['filenames']
    else:
        raise ValueError( 'Unknown format of hdf5' )
    if 'validation' in hdf5:
        filenames_validate = hdf5['validation']['filenames']
    else:
        raise ValueError( 'Unknown format of hdf5' )
    if 'test' in hdf5:
        filenames_test = hdf5['test']['filenames']
    else:
        raise ValueError( 'Unknown format of hdf5' )
    #filenames=np.concatenate((filenames_train,filenames_validate,filenames_test))
    filenames=filenames_train
    print('filenames shape:',filenames.shape)
    NDicoms=filenames.shape[0]
    df='1.3.12.2.1107.5.2.18.41754.2015032110084651856661216.dcm'
    for i in range(NDicoms):
        if df in filenames[i]:
            print('iiiiiiiiii:',i)
            break

if __name__ == '__main__':
    evaluate_on_hdf5('F://sa-oc-lvsc_new-filenames.hdf5')
    print('Finished')