import h5py
import numpy
import os
import os.path
import torch.utils.data as data


def load_all_h5(h5_adress):
    test_list = []
    train_list = []
    for root, dirs, files in os.walk(h5_adress):
        for f in files:
            txt_absPath = os.path.join(root, f)
            ff = open(txt_absPath, 'r')
            path_list = ff.read().split('\n')
            if f.split("_")[0] != "train":
                test_list.extend(path_list)
            else:
                train_list.extend(path_list)
            f.close()

    return train_list, test_list


class gaze_dataset(data.Dataset):
    def __init__(self, h5_path_list):
        self.h5_path_list = h5_path_list[:]

    def __getitem__(self, index):
        filename = self.h5_path_list[index]
        f = h5py.File(filename_l, 'r')
        left_eye_img = f['left_eye'].value
        right_eye_img = f['right_eye'].value
        left_label = f['left_label'].value
        rigth_label = f['right_label'].value
        left_headpose = f['left_head_pose'].value
        right_headpose = f['right_head_pose'].value
        face = f['face'].value
        f.close()
        return left_eye_img, right_eye_img, left_label, rigth_label, left_headpose, right_headpose, face

    def __len__(self):
        return len(self.h5_path_list_left)

