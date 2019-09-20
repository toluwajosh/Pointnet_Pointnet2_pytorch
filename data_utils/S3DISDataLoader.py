# *_*coding:utf-8 *_*
import os
from torch.utils.data import Dataset
import numpy as np
import h5py
import time

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window',
           'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


def loadDataFile(filename):
    return load_h5(filename)


def recognize_all_data(test_area=5):
    ALL_FILES = getDataFiles('./data/indoor3d_sem_seg_hdf5_data/all_files.txt')
    room_filelist = [line.rstrip() for line in open(
        './data/indoor3d_sem_seg_hdf5_data/room_filelist.txt')]
    data_batch_list = []
    label_batch_list = []
    for h5_filename in ALL_FILES:
        data_batch, label_batch = loadDataFile('./data/' + h5_filename)
        data_batch_list.append(data_batch)
        label_batch_list.append(label_batch)
    data_batches = np.concatenate(data_batch_list, 0)
    label_batches = np.concatenate(label_batch_list, 0)

    test_area = 'Area_' + str(test_area)
    train_idxs = []
    test_idxs = []
    for i, room_name in enumerate(room_filelist):
        if test_area in room_name:
            test_idxs.append(i)
        else:
            train_idxs.append(i)

    train_data = data_batches[train_idxs, ...]
    train_label = label_batches[train_idxs]
    test_data = data_batches[test_idxs, ...]
    test_label = label_batches[test_idxs]
    print('train_data', train_data.shape, 'train_label', train_label.shape)
    print('test_data', test_data.shape, 'test_label', test_label.shape)
    return train_data, train_label, test_data, test_label


class S3DISDataLoader(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


if __name__ == "__main__":
    import open3d as o3d

    train_data, train_label, test_data, test_label = recognize_all_data()
    print(train_data[0].shape)

    # # one time visualization ->
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(train_data[0][:,:3])
    # o3d.visualization.draw_geometries([pcd])


    # train label
    print(train_label[0])

    # # stream visualization ->
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # save_image = False

    # # set up pcd and visualization object
    # pcd = o3d.geometry.PointCloud()
    # vis.add_geometry(pcd)

    # to_reset_view_point = True
    # for i in range(1000):
    #     cloud = train_data[i][:,:3]
    #     pcd.points = o3d.utility.Vector3dVector(cloud)

    #     vis.update_geometry()
    #     if to_reset_view_point:
    #         vis.reset_view_point(True)
    #         to_reset_view_point = False
    #     vis.poll_events()
    #     vis.update_renderer()
    #     if save_image:
    #         vis.capture_screen_image("temp_%04d.jpg" % i)
    #     time.sleep(0.2)
    # vis.destroy_window()
