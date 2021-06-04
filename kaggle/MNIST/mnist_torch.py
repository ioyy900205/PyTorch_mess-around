import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataset import random_split

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import pandas as pd
#import dask.dataframe as pd


TRAIN_CSV = "/home/liuliang/deep_learning/PyTorch_mess-around/kaggle/train.csv"
TEST_CSV = "/home/liuliang/deep_learning/PyTorch_mess-around/kaggle/test.csv"
OUT_CSV = "/home/liuliang/deep_learning/PyTorch_mess-around/kaggle/sub-digit-recognizer.csv"

def build_trainval_loaders(datafield, batch_size=64, val_size=0.1):

    data_n = np.array(datafield)
    data_x_t = torch.tensor(data_n[:,1:]).float().reshape(-1, 1, 28, 28)
    data_x_t = data_x_t.float() / 255 # normalize
    data_y_t = torch.tensor(data_n[:,0])

    dataset = torch.utils.data.TensorDataset(data_x_t, data_y_t)

    # split for validation set
    val_ds_size = int(len(dataset) * val_size)
    sizes = [len(dataset) - val_ds_size, val_ds_size]
    train_dataset, val_dataset = random_split(dataset, sizes)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader

def build_test_loader(datafield, batch_size=64):

    data_n = np.array(datafield)
    data_x_t = torch.tensor(data_n).float().reshape(-1, 1, 28, 28)
    data_x_t = data_x_t.float() / 255 # normalize

    dataset = torch.utils.data.TensorDataset(data_x_t)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

model = nn.Sequential(
        nn.Conv2d(1, 32, (3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2)),
        nn.Conv2d(32, 64, (3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2)),
        nn.Dropout(0.2),
        nn.Conv2d(64, 64, (3, 3)),
        nn.Dropout(0.2),
        nn.Flatten(1, -1),
        nn.Linear(576, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 10),
        )

def train(model, loss_fn, optimizer, data_loader):

    model.train()

    losses = 0
    losses_cnt = 0
    correct_cnt = 0
    total_cnt = 0

    for x, y in data_loader:

        out = model(x)

        correct_cnt += int(sum(y == torch.argmax(out, dim=1)))

        total_cnt += len(x)

        loss = loss_fn(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # detach the loss or graph doesn't get freed and memory keeps
        # increasing
        losses += loss.detach().item()
        losses_cnt += 1



    return losses / losses_cnt, correct_cnt / total_cnt

def evaluate(model, loss_fn, data_loader):

    model.eval()

    # validate

    losses = 0
    losses_cnt = 0
    correct_cnt = 0
    total_cnt = 0

    with torch.no_grad():

        for x, y in data_loader:

            out = model(x)

            loss = loss_fn(out, y)

            correct_cnt += int(sum(y == torch.argmax(out, dim=1)))
            total_cnt += len(x)

            # detach the loss or graph doesn't get freed and memory keeps
            # increasing
            losses += loss.detach().item()
            losses_cnt += 1

    return losses / losses_cnt, correct_cnt / total_cnt


def create_submission(out_csv, model, data_loader):

    model.eval()

    offset = 0

    # write header
    output = pd.DataFrame({"ImageId": [], "Label": []})
    output.to_csv(out_csv, index=False)

    with torch.no_grad():

        for x in data_loader:

            out = model(x[0])
            out = torch.argmax(out, dim=1)

            # append entries
            end = offset+len(x[0])
            ids = range(offset+1, end+1)
            offset = end
            output = pd.DataFrame({"ImageId": ids, "Label": out})
            output.to_csv(out_csv, mode='a', header=False, index=False)

def main():

    n_epochs = 15
    batch_size = 32

    print("Prepare data")
    train_loader, val_loader = build_trainval_loaders(pd.read_csv(TRAIN_CSV),
            batch_size=batch_size, val_size=0.07)

    print("Setup model")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    losses = []
    accs   = []
    losses_val = []
    accs_val = []

    print(f"Train: epochs={n_epochs}")
    for epoch in range(1, n_epochs+1):

        loss, acc = train(model, loss_fn, optimizer, train_loader)

        loss_val, acc_val = evaluate(model, loss_fn, val_loader)

        scheduler.step()

        losses.append(loss)
        accs.append(acc)
        losses_val.append(loss_val)
        accs_val.append(acc_val)

        print (f"Epoch {epoch:3d} | Train loss {loss:.6f} acc {acc:.4f} | "
                f"Validation loss {loss_val:.6f} acc {acc_val:.4}")

    print(f"Plot")
    epochs = range(1, n_epochs+1)
    fig, axis = plt.subplots(2, figsize=(10, 10))
    fig.tight_layout(h_pad=5)

    axis[0].set_title("Loss")
    axis[0].set_xlabel("epoch")
    axis[0].set_ylabel("loss")

    axis[0].plot(epochs, losses, "b", label="loss_train")
    axis[0].plot(epochs, losses_val, "g", label="loss_validate")
    axis[0].legend()

    axis[1].set_title("Accuracy")
    axis[1].set_xlabel("epoch")
    axis[1].set_ylabel("accuracy")
    axis[1].plot(epochs, accs, "b", label="accuracy_train")
    axis[1].plot(epochs, accs_val, "g", label="accuracy_validate")
    axis[1].legend()

    plt.show()

    print(f"Creating submission: {OUT_CSV}")
    test_loader = build_test_loader(pd.read_csv(TEST_CSV),
            batch_size=batch_size)
    create_submission(OUT_CSV, model, test_loader)
    print("Done.")

if __name__ == "__main__":
    main()