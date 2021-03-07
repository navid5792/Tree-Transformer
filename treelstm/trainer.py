from tqdm import tqdm

import torch

from . import utils


class Trainer(object):
    def __init__(self, args, model, criterion, optimizer, device):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epoch = 0

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        indices = torch.randperm(len(dataset), dtype=torch.long, device='cpu')
        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            ltree, linput, rtree, rinput, label, larc, rarc = dataset[indices[idx]]
            target = torch.tensor(label).long().unsqueeze(0)
            #target = utils.map_label_to_target(label, dataset.num_classes)
            linput, rinput, larc, rarc = linput.to(self.device), rinput.to(self.device), larc.to(self.device), rarc.to(self.device)
            target = target.to(self.device)
            output = self.model(ltree, linput, rtree, rinput, larc, rarc)
            loss = self.criterion(output, target)
            total_loss += loss.item()
            loss.backward()
            if idx % self.args.batchsize == 0 and idx > 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        self.epoch += 1
        return total_loss / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            predictions = torch.zeros(len(dataset), dtype=torch.float, device='cpu')
            indices = torch.arange(1, dataset.num_classes + 1, dtype=torch.float, device='cpu')
            count = 0
            for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
                ltree, linput, rtree, rinput, label, larc, rarc = dataset[idx]
                #target = utils.map_label_to_target(label, dataset.num_classes)
                target = torch.tensor(label).long().unsqueeze(0)
                linput, rinput, larc, rarc = linput.to(self.device), rinput.to(self.device), larc.to(self.device), rarc.to(self.device)
                target = target.to(self.device)
                output = self.model(ltree, linput, rtree, rinput, larc, rarc)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                output = output.squeeze().to('cpu')
                #print(output)
                #predictions[idx] = torch.dot(indices, torch.exp(output))
                predictions[idx] = output.topk(1)[1]
                if int(output.topk(1)[1]) == int(target[0]):
                    count += 1
            print("\nAccuracy: ", count/len(dataset), "\n")
        return total_loss / len(dataset), predictions
