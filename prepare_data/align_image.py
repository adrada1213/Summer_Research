import os
import pydicom
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from pointer_functions import load_ptr_content


def plot_coords(CineImgOrient, CineImgPos, TaggedImgOrient, TaggedImgPos):
    ax = plt.axes(projection="3d")

    #ax.scatter3D(CineImgPos[0], CineImgPos[1], CineImgPos[2])
    #ax.scatter3D(TaggedImgPos[0], TaggedImgPos[1], TaggedImgPos[2])

    ax.plot([CineImgPos[0], CineImgOrient[0]],[CineImgPos[1], CineImgOrient[1]], zs=[CineImgPos[2], CineImgOrient[2]])
    ax.plot([CineImgPos[0], CineImgOrient[3]],[CineImgPos[1], CineImgOrient[4]], zs=[CineImgPos[2], CineImgOrient[5]])
    ax.plot([TaggedImgPos[0], TaggedImgOrient[0]],[TaggedImgPos[1], TaggedImgOrient[1]], zs=[TaggedImgPos[2], TaggedImgOrient[2]])
    ax.plot([TaggedImgPos[0], TaggedImgOrient[3]],[TaggedImgPos[1], TaggedImgOrient[4]], zs=[TaggedImgPos[2], TaggedImgOrient[5]])
    ax.legend(["Cine 1","Cine 2","Tagged 1", "Tagged"])

    plt.show()

if __name__ == "__main__":
    ptrs_path = "C:\\Users\\arad572\\Documents\\Summer Research\\code\\prepare_data\\img_ptrs\\matches"

    #ptrs = os.listdir(ptrs_path)
    ptrs = ["C2_VY_DH_DE_Bio_match.img_imageptr", "4J_VU_SU_3B_Bio_match.img_imageptr"]

    for ptr in ptrs:
        pat_name = ptr.replace("_match.img_imageptr", "")
        ptr_path = os.path.join(ptrs_path, ptr)

        ptr_content = load_ptr_content(ptr_path)

        cine_con = ptr_content["series"] == 0
        tagged_con = ptr_content["series"] == 1

        cine_frames = ptr_content[cine_con]
        tagged_frames = ptr_content[tagged_con]

        cine_first_frames = cine_frames[cine_frames["index"]==0]
        tagged_first_frames = tagged_frames[tagged_frames["index"]==0]

        #print(cine_first_frames)
        #print(tagged_first_frames)

        for i in range(len(cine_first_frames)):
            cine_fr = cine_first_frames["path"][i]
            tagged_fr = tagged_first_frames["path"][i]

            path = "E:\\Original Images\\2015"
            cine_fr = cine_fr.replace("IMAGEPATH", path)
            if not os.path.isfile(cine_fr):
                cine_fr = cine_fr.replace("\\2015\\", "\\2014\\")
                path = "E:\\Original Images\\2014"
            tagged_fr = tagged_fr.replace("IMAGEPATH", path)

            ds_cine = pydicom.dcmread(cine_fr, specific_tags=["ImagePositionPatient", "ImageOrientationPatient"])
            ds_tagged = pydicom.dcmread(tagged_fr)

            print("Patient {} Slice {}".format(pat_name, i))
            print("Cine Image Position: {}".format(ds_cine.ImagePositionPatient))
            print("Tagged Image Position: {}".format(ds_tagged.ImagePositionPatient))
            print("Cine Image Orientation: {}".format(ds_cine.ImageOrientationPatient))
            print("Tagged Image Orientation: {}\n".format(ds_tagged.ImageOrientationPatient))

            plot_coords(ds_cine.ImageOrientationPatient, ds_cine.ImagePositionPatient, ds_tagged.ImageOrientationPatient, ds_tagged.ImagePositionPatient)

        #break