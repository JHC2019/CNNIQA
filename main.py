from argparse import ArgumentParser

import torch
import torchvision
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import csv
import yaml
import numpy as np

from tqdm import tqdm
from scipy import stats

from IQA_utils import *
from IQA_model import CNNIQAnet
from IQA_dataset import get_data_loaders
from torch.utils.tensorboard import SummaryWriter

def run(train_batch_size, epochs, lr, weight_decay, config, exp_id, log_dir, trained_model_file, save_result_file, disable_gpu=False):
    if config['test_ratio']:
        train_loader, val_loader, test_loader = get_data_loaders(config, train_batch_size, exp_id)
    else:
        train_loader, val_loader = get_data_loaders(config, train_batch_size, exp_id)

    device = torch.device("cuda" if not disable_gpu and torch.cuda.is_available() else "cpu")

    model = CNNIQAnet(config)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    writer_train = SummaryWriter(log_dir=log_dir+'/train')
    writer_val = SummaryWriter(log_dir=log_dir+'/val')
    writer_test = SummaryWriter(log_dir=log_dir+'/test')

    best_criterion_val = -1
    best_criterion_test = -1
    best_epoch_val = 0
    best_epoch_test = 0

    with open(save_result_file + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['type', 'epoch', 'SROCC', 'KROCC', 'PLCC', 'RMSE',' OR'])

    for epoch in tqdm(range(epochs)):
        # train
        model.train()
        L = 0
        for i, (inputs, targets) in enumerate(train_loader, 0):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = [i.to(device) for i in targets]

            outputs = model(inputs)
            loss = loss_fn(outputs, targets[0])
            loss.backward()
            optimizer.step()

            L += loss.item()
        train_loss = L / (i + 1)
        writer_train.add_scalar("Loss", train_loss, epoch)
        print("Epoch: {}".format(epoch))
        print("{:10} Loss: {:.4f} ".format('Training', train_loss))

        # val
        model.eval()
        L = 0
        q = []
        sq = []
        sq_std = []
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader, 0):
                inputs = inputs.squeeze(0).to(device)
                targets = [i.to(device) for i in targets]

                outputs = model(inputs)
                loss = loss_fn(torch.mean(outputs, 0, keepdim=True), targets[0])
                L += loss.item()

                q.append(torch.mean(outputs).cpu().numpy())
                sq.append(targets[0].squeeze().cpu().numpy())
                sq_std.append(targets[1].squeeze().cpu().numpy())
        val_loss = L / (i + 1)
        measure_values = measure(sq, q, sq_std)
        SROCC, KROCC, PLCC, RMSE, OR = measure_values
        writer_val.add_scalar("Loss", val_loss, epoch)
        print("{:10} Loss: {:.4f} ".format('Validation', val_loss))
        print("{:10} Results - SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} OR: {:.2f}%"
              .format('Validation', SROCC, KROCC, PLCC, RMSE, 100 * OR))
        tb_write(writer_val, measure_values, epoch)

        if SROCC > best_criterion_val:
            best_criterion_val = SROCC
            best_epoch_val = epoch
            torch.save(model.state_dict(), trained_model_file + '-val')

            with open(save_result_file + '.csv', 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['val', best_epoch_val, SROCC, KROCC, PLCC, RMSE, OR*100])

        # test
        if config['test_ratio'] > 0 and config['test_during_training']:
            L = 0
            q = []
            sq = []
            sq_std = []
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(test_loader, 0):
                    inputs = inputs.squeeze(0).to(device)
                    targets = [i.to(device) for i in targets]

                    outputs = model(inputs)
                    loss = loss_fn(torch.mean(outputs, 0, keepdim=True), targets[0])
                    L += loss.item()

                    q.append(torch.mean(outputs).cpu().numpy())
                    sq.append(targets[0].squeeze().cpu().numpy())
                    sq_std.append(targets[1].squeeze().cpu().numpy())
            test_loss = L / (i + 1)
            measure_values = measure(sq, q, sq_std)
            SROCC, KROCC, PLCC, RMSE, OR = measure_values
            writer_test.add_scalar("Loss", test_loss, epoch)
            print("{:10} Loss: {:.4f} ".format('Test', test_loss))
            print("{:10} Results - SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} OR: {:.2f}%"
              .format('Test', SROCC, KROCC, PLCC, RMSE, 100 * OR))
            tb_write(writer_test, measure_values, epoch)

            if SROCC > best_criterion_test:
                best_criterion_test = SROCC
                best_epoch_test = epoch
                torch.save(model.state_dict(), trained_model_file + '-test')

                with open(save_result_file + '.csv', 'a+', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['test', best_epoch_test, SROCC, KROCC, PLCC, RMSE, OR*100])
    writer_train.close()
    writer_val.close()
    writer_test.close()


if __name__ == "__main__":
    parser = ArgumentParser(description='PyTorch CNNIQA')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='config file path (default: config.yaml)')
    parser.add_argument('--exp_id', default='0', type=str,
                        help='exp id (default: 0)')
    parser.add_argument('--database', default='LIVE', type=str,
                        help='database name (default: LIVE)')
    parser.add_argument('--model', default='CNNIQA', type=str,
                        help='model name (default: CNNIQA)')
    parser.add_argument("--log_dir", type=str, default="tensorboard_logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print('exp id: ' + args.exp_id)
    print('database: ' + args.database)
    print('model: ' + args.model)
    print('is_gray: ' + str(config['is_gray']))
    config.update(config[args.database])
    # config.update(config[args.model])

    log_dir = args.log_dir + '/EXP{}-{}-{}-lr={}'.format(args.exp_id, args.database, args.model, args.lr)
    ensure_dir('checkpoints')
    trained_model_file = 'checkpoints/{}-{}-EXP{}-lr={}'.format(args.model, args.database, args.exp_id, args.lr)
    ensure_dir('results')
    save_result_file = 'results/{}-{}-EXP{}-lr={}'.format(args.model, args.database, args.exp_id, args.lr)

    run(args.batch_size, args.epochs, args.lr, args.weight_decay, config, args.exp_id,
        log_dir, trained_model_file, save_result_file, args.disable_gpu)