import h5py
import numpy as np

def create_datasets(hf, key, dsm):
    # create group for the current set
    grp = hf.create_group(key)
    grp_cine = grp.create_group("cine")
    grp_tagged = grp.create_group("tagged")

    # converting data to numpy arrays
    patients = np.array(dsm.patient_names, dtype=object)
    cine_dicom_paths = np.array(dsm.cine_dicom_paths, dtype = object)
    tagged_dicom_paths = np.array(dsm.tagged_dicom_paths, dtype = object)

    grp.create_dataset("patients", data=patients, dtype=h5py.special_dtype(vlen=str), maxshape = (None, ))
    grp.create_dataset("slices", data=dsm.slices, maxshape = (None, ))
    
    # put all the cine data in the cine group
    grp_cine.create_dataset("dicom_paths", data=cine_dicom_paths, dtype=h5py.special_dtype(vlen=str), maxshape = (None, 50))
    grp_cine.create_dataset("centroids", data=dsm.cine_centroids, maxshape = (None, 3))
    grp_cine.create_dataset("landmark_coords", data=dsm.cine_landmark_coords, maxshape = (None, 2, 2, 168))
    grp_cine.create_dataset("images", data=dsm.cine_images, maxshape = (None, 50, 256, 256))
    grp_cine.create_dataset("es_indices", data=dsm.cine_es_indices, maxshape = (None, ))

    # put all the tagged data in tagged group
    grp_tagged.create_dataset("dicom_paths", data=tagged_dicom_paths, dtype=h5py.special_dtype(vlen=str), maxshape = (None, 20))
    grp_tagged.create_dataset("centroids", data=dsm.tagged_centroids, maxshape = (None, 3))
    grp_tagged.create_dataset("landmark_coords", data=dsm.tagged_landmark_coords, maxshape = (None, 20, 2, 168))
    grp_tagged.create_dataset("images", data=dsm.tagged_images, maxshape = (None, 20, 256, 256))
    grp_tagged.create_dataset("es_indices", data=dsm.tagged_es_indices, maxshape = (None, ))

def add_datasets(hf, key, dsm):
    grp = hf["//{}".format(key)]
    grp_cine = hf["//{}//cine".format(key)]
    grp_tagged = hf["//{}//tagged".format(key)]
                        
    # converting data to numpy arrays
    patients = np.array(dsm.patient_names, dtype=object)
    cine_dicom_paths = np.array(dsm.cine_dicom_paths, dtype = object)
    tagged_dicom_paths = np.array(dsm.tagged_dicom_paths, dtype = object)

    grp["patients"].resize((grp["patients"].shape[0])+patients.shape[0], axis = 0)
    grp["patients"][-patients.shape[0]:] = patients

    grp["slices"].resize((grp["slices"].shape[0])+dsm.slices.shape[0], axis = 0)
    grp["slices"][-dsm.slices.shape[0]:] = dsm.slices

    # cines
    grp_cine["dicom_paths"].resize((grp_cine["dicom_paths"].shape[0])+cine_dicom_paths.shape[0], axis = 0)
    grp_cine["dicom_paths"][-cine_dicom_paths.shape[0]:] = cine_dicom_paths

    grp_cine["centroids"].resize((grp_cine["centroids"].shape[0])+dsm.cine_centroids.shape[0], axis = 0)
    grp_cine["centroids"][-dsm.cine_centroids.shape[0]:] = dsm.cine_centroids

    grp_cine["landmark_coords"].resize((grp_cine["landmark_coords"].shape[0])+dsm.cine_landmark_coords.shape[0], axis = 0)
    grp_cine["landmark_coords"][-dsm.cine_landmark_coords.shape[0]:] = dsm.cine_landmark_coords

    grp_cine["images"].resize((grp_cine["images"].shape[0])+dsm.cine_images.shape[0], axis = 0)
    grp_cine["images"][-dsm.cine_images.shape[0]:] = dsm.cine_images

    grp_cine["es_indices"].resize((grp_cine["es_indices"].shape[0])+dsm.cine_es_indices.shape[0], axis = 0)
    grp_cine["es_indices"][-dsm.cine_es_indices.shape[0]:] = dsm.cine_es_indices

    # tagged
    grp_tagged["dicom_paths"].resize((grp_tagged["dicom_paths"].shape[0])+tagged_dicom_paths.shape[0], axis = 0)
    grp_tagged["dicom_paths"][-tagged_dicom_paths.shape[0]:] = tagged_dicom_paths

    grp_tagged["centroids"].resize((grp_tagged["centroids"].shape[0])+dsm.tagged_centroids.shape[0], axis = 0)
    grp_tagged["centroids"][-dsm.tagged_centroids.shape[0]:] = dsm.tagged_centroids

    grp_tagged["landmark_coords"].resize((grp_tagged["landmark_coords"].shape[0])+dsm.tagged_landmark_coords.shape[0], axis = 0)
    grp_tagged["landmark_coords"][-dsm.tagged_landmark_coords.shape[0]:] = dsm.tagged_landmark_coords

    grp_tagged["images"].resize((grp_tagged["images"].shape[0])+dsm.tagged_images.shape[0], axis = 0)
    grp_tagged["images"][-dsm.tagged_images.shape[0]:] = dsm.tagged_images

    grp_tagged["es_indices"].resize((grp_tagged["es_indices"].shape[0])+dsm.tagged_es_indices.shape[0], axis = 0)
    grp_tagged["es_indices"][-dsm.tagged_es_indices.shape[0]:] = dsm.tagged_es_indices