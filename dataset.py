import numpy as np
import cv2 as cv
import random
import torchvision.transforms as T
from torch_geometric.transforms import ToSLIC, RadiusGraph

import os
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset


# convert an image to graph data
def regular_grid_transform(img, class_index: int, num_classes: int):
    # in order to make sure the sufficient gpu memory space
    # I resize the img from (100, 100) to (32, 32)
    img = cv.resize(img, (32, 32))
    # get image's shape
    row, col = img.shape[0], img.shape[1]

    b, g, r = cv.split(img)

    num_edges = 2 * (2 * row * col - (row + col))

    x = np.zeros((row * col, 3))
    edge_index = np.zeros((2, num_edges))
    # y(ground truth label)'s shape MUST be modified according to your class number
    y = np.zeros((1, num_classes))

    # generate node features
    # use pixel values as node features
    v_index = 0
    for v_row in range(row):
        for v_col in range(col):
            # add pixel values to node features
            # normalize
            x[v_index][0] = b[v_row][v_col] / 255
            x[v_index][1] = g[v_row][v_col] / 255
            x[v_index][2] = r[v_row][v_col] / 255
            v_index += 1

    # add edges to edge_index
    e_index = 0
    # generate horizontal edges
    for v_row in range(row):
        for v_col in range(col - 1):
            v_s = v_row * col + v_col
            v_e = v_s + 1
            edge_index[0][e_index] = v_s
            edge_index[1][e_index] = v_e
            e_index += 1
            edge_index[1][e_index] = v_s
            edge_index[0][e_index] = v_e
            e_index += 1
    # generate vertical edges
    for v_row in range(row - 1):
        for v_col in range(col):
            v_s = v_row * col + v_col
            v_e = v_s + col
            edge_index[0][e_index] = v_s
            edge_index[1][e_index] = v_e
            e_index += 1
            edge_index[1][e_index] = v_s
            edge_index[0][e_index] = v_e
            e_index += 1

    # get the picture's label and mark the corresponding row to 1.
    y[0][class_index] = 1

    # convert ndarrays into tensors
    x = torch.from_numpy(x).float()
    edge_index = torch.from_numpy(edge_index).long()
    y = torch.from_numpy(
        y
    ).float()  # must use float type to compute your loss with CrossEntropy function

    # generate and return a data object
    data = Data(x=x, edge_index=edge_index, y=y)

    return data


class MyDataset(InMemoryDataset):
    Trainset = True

    def __init__(
        self,
        root,
        Trainset,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.Trainset = Trainset
        super(MyDataset, self).__init__(
            root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ""

    @property
    def processed_file_names(self):
        return ["datas.pt"]

    def download(self):
        pass

    def process(self):
        data_list = []
        class_mark = 0
        class_num = len(os.listdir(self.raw_dir))
        # We need to convert images from nparray to torch.tensor at first.
        # to transform images into superpixel graph, I choose SLIC algorithm to do it.
        # then add edges according by data.pos, use RadiusGraph method.
        transform = T.Compose(
            [
                T.ToTensor(),
                ToSLIC(n_segments=300),
                RadiusGraph(r=256, max_num_neighbors=16),
            ]
        )

        random.seed(8161)

        for filename in os.listdir(self.raw_dir):
            # scan all images in the folder with the specified class
            filelist = []
            rawfilelist = os.walk(self.raw_dir + "\\" + filename)

            # the tensor of the ground truth lables
            gt = torch.zeros((1, class_num))

            # set corresponding class to 1
            gt[0][class_mark] = 1

            # wrap ground truth labels into a Data object
            # use Data.update method to add this tensor into data
            patch_data = Data(y=gt)

            for folder, subfolder, file in rawfilelist:
                for imgfile in file:
                    filelist.append(os.path.join(folder, imgfile))

            # gen graph datas and append the data_list
            for file in filelist:
                img = cv.imread(file)

                # convert image into superpixel graph Data
                data = transform(img=img)

                # add ground truth labels to graph Data
                data.update(patch_data)

                data_list.append(data)

            print("class " + str(class_mark) + " complete")
            class_mark += 1

        # save processed data
        data, slices = self.collate(data_list=data_list)
        torch.save((data, slices), self.processed_paths[0])


# this function is used to print info of the dataset
def dataset_info(dataset):
    print(f"Dataset: {dataset}:")
    print("====================")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
