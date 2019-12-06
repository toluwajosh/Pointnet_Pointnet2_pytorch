# *_*coding:utf-8 *_*
import os
import time
import h5py
import torch
import pykitti
import open3d

import numpy as np

from copy import deepcopy
from torch.utils.data import Dataset


class KittiOdometryLoader(Dataset):
    def __init__(
        self, basedir, classes=None, sequence="train", points_size=None, to_tensor=True
    ):
        self.basedir = basedir
        self.to_tensor = to_tensor
        self.points_size = points_size
        self.classes = classes
        all_sequence = [
            "00",
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "09",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "08",
        ]
        if sequence == "all":
            all_sequence = all_sequence[:]
        elif sequence == "train":
            all_sequence = all_sequence[:10]
        elif sequence == "valid":
            all_sequence = all_sequence[-1:]
        elif sequence == "test":
            all_sequence = all_sequence[10:-1]
        else:
            if type(sequence) == str:
                assert sequence in all_sequence, "Sequence must be part of {}".format(
                    all_sequence
                )
                all_sequence = [sequence]
            elif type(sequence) == list:
                for seq in sequence:
                    assert seq in all_sequence, "Sequence must be part of {}".format(
                        all_sequence
                    )
                all_sequence = sequence

        all_velo_files = []
        all_label_files = []
        for seq in all_sequence:
            dataset = pykitti.odometry(basedir, seq)
            all_velo_files += dataset.velo_files
            all_label_files += dataset.label_files
        dataset.velo_files = all_velo_files
        dataset.label_files = all_label_files

        self.dataset = dataset
        self.num_data = len(dataset.velo_files)
        print("dataset size: ", self.num_data)

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        velo = self.dataset.get_velo(index)
        label = self.dataset.get_label(index)
        if self.points_size is not None:
            max_idx = velo.shape[0]
            idx = np.random.randint(max_idx, size=self.points_size)
            velo = velo[idx, :]
            label = label[idx]

        # clip output into number of classes
        if self.classes is not None:
            label[label > self.classes - 1] = 0

        # create mask
        mask = deepcopy(label)
        mask[mask > 0] = 1
        mask[mask == 0] = 0.05
        # mask[mask == 2] = 0

        if self.to_tensor:
            velo = torch.from_numpy(velo)
            label = torch.from_numpy(label)
            mask = torch.from_numpy(mask)

        data = {"velo": velo, "label": label, "mask": mask}
        return data


def labels_color(labels):
    pass

if __name__ == "__main__":
    # torch.set_printoptions(edgeitems=200)
    # np.set_printoptions(threshold=np.inf)
    save_dir = "/media/tjosh/ssd_vault1/kitti_odometry/pcd/sequences/08/"
    basedir = "/media/tjosh/ssd_vault1/kitti_odometry/dataset"
    odo_data = KittiOdometryLoader(
        basedir, sequence="08", classes=20, to_tensor=False, points_size=None
    )
    # print(len(odo_data))
    # exit(0)

    # visualize with open3d
    pcd = open3d.geometry.PointCloud()
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    to_reset_view_point = True
    for i in range(200):
        print(i)
        data = odo_data[i]
        velo = data["velo"]
        label = data["label"]
        points = velo[:, :3]
        # color_map = np.reshape(label, -1)
        print(label.shape)
        exit(0)
        pcd.points = open3d.utility.Vector3dVector(points)
        pcd.colors = open3d.utility.Vector3dVector(np.asarray(color_map))
        # print(pcd)
        # open3d.io.write_point_cloud(
        #     save_dir + str(time.time()).replace(".", "") + ".pcd", pcd
        # )
        # print(label)
        # print(np.max(label))
        # print()

        vis.update_geometry()
        if to_reset_view_point:
            vis.reset_view_point(True)
            to_reset_view_point = False
        vis.poll_events()
        vis.update_renderer()
        # time.sleep(0.2)

    # vis.destroy_window()
