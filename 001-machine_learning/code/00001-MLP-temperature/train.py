import torch
import torch.nn as nn
import numpy as np

from observations import mnist
from tqdm import tqdm

import argparse


def parseArgs():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--cpu', action='store_true', default=False,
                       help="Use CPUs to train the model!")
    group.add_argument('--single-gpu', action='store_true', default=False,
                       help="Use single GPU to train the model!")
    group.add_argument('--multi-gpu', action='store_true', default=False,
                       help="Use multiple GPUs to train the model!")

    parser.add_argument('--batch', '-b', type=int, default=128,
                        help="Batch size for training")

    args = parser.parse_args()
    return args


def main(args):
    # Define model
    model = nn.Sequential(
        nn.Conv2d(1, 16, 5),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 16, 5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 16, 5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(144, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    ).double()

    loss_fn = nn.CrossEntropyLoss()

    # Check which version model is supposed be to run
    if (args.cpu):
        device = "cpu"
    elif (args.single_gpu):
        device = "cuda:0"
    elif (args.multi_gpu):
        device = "cuda:0"
        model = nn.DataParallel(model)

    model = model.to(device)
    print('Starting training on device: {}'.format(device))

    # Load data
    (train_x, train_y), (test_x, test_y) = mnist('./data')

    train_x = np.reshape(train_x, [train_x.shape[0], 1, 28, 28]) / 255.
    test_x = np.reshape(test_x, [test_x.shape[0], 1, 28, 28]) / 255.

    trainset = torch.utils.data.TensorDataset(torch.from_numpy(train_x),
                                              torch.from_numpy(train_y).long())
    testset = torch.utils.data.TensorDataset(torch.from_numpy(test_x),
                                             torch.from_numpy(test_y).long())
    batch_size = args.batch
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

    # Vanilla stochastic gradient descent. Rather slow.
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # Adam optimizer converges much faster
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(10):
        total_loss = 0
        num_correct_test = 0
        num_correct_train = 0

        # Train on training set
        for batch in tqdm(iter(trainloader), leave=False):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss
            pred_labels = torch.argmax(predictions, dim=1)
            num_correct_train += (pred_labels == labels).sum()

    print('Epoch: {}, average training loss: {:.6f}'.format(epoch, total_loss.item() / len(trainset)))
    print('Train accuracy: {:.4f}'.format(num_correct_train.item() / len(trainset)))

    # Evaluate on test set
    for batch in tqdm(iter(testloader), leave=False):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images)
        pred_labels = torch.argmax(predictions, dim=1)
        num_correct_test += (pred_labels == labels).sum()

    print('Test accuracy: {:.4f}'.format(num_correct_test.item() / len(testset)))


if __name__ == "__main__":
    args = parseArgs()
    main(args)
