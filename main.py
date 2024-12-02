import os
import torch
from tqdm import tqdm
import math
import torch
import torch.nn as nn


from torch_geometric.transforms import ToSLIC
from torch_geometric.datasets import MNISTSuperpixels
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from dataset import MyDataset, dataset_info
from gcn_model import MyModel


from argparse import ArgumentParser


def train(model, train_loader, valid_loader, device, n_epoch):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epoch):
        # set model to train mode
        model.train()

        loss_record = []
        # use tqdm to show the training progress
        train_bar = tqdm(train_loader, position=0, leave=True)

        for data in train_bar:
            optimizer.zero_grad()
            data = data.to(device)
            pred = model(data)
            loss = criterion(pred, data.y)
            loss.backward()
            optimizer.step()

            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar
            train_bar.set_description(f"Epoch [{epoch+1} / {n_epoch}]")
            train_bar.set_postfix({"loss": loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)
        print(f"Mean train loss in Epoch {epoch+1} : {mean_train_loss:.4f}")

        # Use cross-validation method to determine whether model is improving.
        model.eval()
        loss_record = []
        for data in valid_loader:
            data = data.to(device)
            with torch.no_grad():
                pred = model(data)
                loss = criterion(pred, data.y)

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f"Mean valid loss in Epoch {epoch+1} : {mean_valid_loss:.4f}")

        # if the best loss haven't been update for a long time, stop the training process.
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            # save the best model with smallest valid loss
            print("Saving best model...")
            torch.save(model.state_dict(),
                       "D:\\Code\\py\\GNN\\model\\model.ckpt")
        else:
            stop_count += 1

        if stop_count >= 20:
            print("Model is not improving. Training session is over.")
            break


def test(model, test_loader, device):
    # load the best model saved in the training process
    model.load_state_dict(torch.load("D:\\Code\\py\\GNN\\model\\model.ckpt"))
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    test_bar = tqdm(test_loader, position=0, leave=True)

    for data in test_bar:
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y)
        pred = out.argmax(dim=1)

        test_bar.set_description("Test progress ")
        test_bar.set_postfix({"loss": loss.detach().item()})

        if data.y[0][pred[0]] == 1:
            correct += 1

    print(f"Test accuracy : {correct/len(test_loader.dataset):.4f}")


def get_dataset_path(dataset_name):
    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        dataset_name = "mnist"
    elif dataset_name == "cifar":
        dataset_name = "CIFAR-10"
    elif dataset_name == "animal":
        dataset_name = "animal"
    elif dataset_name == "fruit":
        dataset_name = "fruit"
    elif dataset_name == "blood_cell":
        dataset_name = "blood_cell"

    # match dataset_name:
    #     case "mnist":
    #         dataset_name = "mnist"
    #     case "cifar":
    #         dataset_name = "CIFAR-10"
    #     case "animal":
    #         dataset_name = "animal"
    #     case "fruit":
    #         dataset_name = "fruit"
    #     case "blood_cell":
    #         dataset_name = "blood_cell"

    train_path = "D:\\Code\\py\\GNN\\data\\" + dataset_name + "\\train"
    test_path = "D:\\Code\\py\\GNN\\data\\" + dataset_name + "\\test"

    return train_path, test_path


device = "cuda:0" if torch.cuda.is_available() else "cpu"
in_dim_config = {
    "mnist": 3,
    "cifar": 3,
    "animal": 3,
    "fruit": 3,
    "blood_cell": 3
}
out_dim_config = {
    "mnist": 10,
    "cifar": 10,
    "animal": 10,
    "fruit": 67,
    "blood_cell": 4
}

parser = ArgumentParser()
parser.add_argument("dataset_name", type=str)
args = parser.parse_args()
dataset_name = args.dataset_name

train_path, test_path = get_dataset_path(dataset_name)

Train_Dataset = MyDataset(root=train_path, Trainset=True)

""" randomly split trainset into trainset, validset """
""" ratio -> 3 : 1 """
valid_set_size = int(len(Train_Dataset) * 0.25)
train_set_size = len(Train_Dataset) - valid_set_size
Train_Dataset, Valid_Dataset = random_split(
    Train_Dataset,
    [train_set_size, valid_set_size],
    generator=torch.Generator().manual_seed(8161),
)

""" You can check dataset info by following functions """
# dataset_info(Train_Dataset)
# dataset_info(Valid_Dataset)

Test_Dataset = MyDataset(root=test_path, Trainset=False)
# train_set, test_set, valid_set = 0.6, 0.2, 0.2

""" You can check dataset info by following functions """
# dataset_info(Test_Dataset)

Train_Loader = DataLoader(Train_Dataset, batch_size=32, shuffle=True)
Valid_Loader = DataLoader(Valid_Dataset, batch_size=32, shuffle=False)
Test_Loader = DataLoader(Test_Dataset, batch_size=1, shuffle=False)


model = MyModel(in_dim=in_dim_config[dataset_name],
                out_dim=out_dim_config[dataset_name])
# move the model to device
model = model.to(device)
# set training epochs
n_epoch = 100

""" strat training """
train(
    model=model,
    train_loader=Train_Loader,
    valid_loader=Valid_Loader,
    device=device,
    n_epoch=n_epoch,
)

""" Start testing """
""" outcome will be print after testing progress finished. """
test(model=model, test_loader=Test_Loader, device=device)
