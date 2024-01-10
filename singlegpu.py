import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from datautils import MyTrainDataset

from collections import OrderedDict, defaultdict
import numpy as np
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm
import time
from torch.optim.lr_scheduler import LambdaLR

# --------------------------My code------------------------------ #
class DeepNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DeepNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
# --------------------------------------------------------------- #
# --------------------------My code------------------------------ #
class VGG(nn.Module):
  ARCH = [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

  def __init__(self) -> None:
    super().__init__()

    layers = []
    counts = defaultdict(int)

    def add(name: str, layer: nn.Module) -> None:
      layers.append((f"{name}{counts[name]}", layer))
      counts[name] += 1

    in_channels = 3
    for x in self.ARCH:
      if x != 'M':
        # conv-bn-relu
        add("conv", nn.Conv2d(in_channels, x, 3, padding=1, bias=False))
        add("bn", nn.BatchNorm2d(x))
        add("relu", nn.ReLU(True))
        in_channels = x
      else:
        # maxpool
        add("pool", nn.MaxPool2d(2))

    self.backbone = nn.Sequential(OrderedDict(layers))
    self.classifier = nn.Linear(512, 10)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # backbone: [N, 3, 32, 32] => [N, 512, 2, 2]
    x = self.backbone(x)
    # avgpool: [N, 512, 2, 2] => [N, 512]
    x = x.mean([2, 3])
    # classifier: [N, 512] => [N, 10]
    x = self.classifier(x)
    return x
# --------------------------------------------------------------- #

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int, 
        scheduler: LambdaLR,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.scheduler = scheduler

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)



def load_train_objs():
    train_set, test_set = getTrainingData()  # load your dataset
    model = VGG().cuda()  # load your model
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=0.4, 
        momentum=0.9, 
        weight_decay=5e-4,
    )

    num_epochs = 20
    steps_per_epoch = 98
    lr_lambda = lambda step: np.interp(
        [step / steps_per_epoch],
        [0, num_epochs * 0.3, num_epochs],
        [0, 1, 0]
    )[0]
    scheduler = LambdaLR(optimizer, lr_lambda)
    return train_set, model, optimizer, test_set, scheduler

# --------------------------My code------------------------------ #
def getTrainingData():
    transforms = {
        "train": Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
    ]),
        "test": ToTensor(),
    }
    dataset = {}
    for split in ["train", "test"]:
        dataset[split] = CIFAR10(
            root="data/cifar10",
            train=(split == "train"),
            download=True,
            transform=transforms[split],
        )

    return dataset["train"], dataset["test"]
# --------------------------------------------------------------- #

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True
    )

# --------------------------My code------------------------------ #
# for evaluation of model
@torch.inference_mode()
def evaluate(
  model: nn.Module,
  dataflow: DataLoader
) -> float:
  model.eval()

  num_samples = 0
  num_correct = 0

  for inputs, targets in tqdm(dataflow, desc="eval", leave=False):
    # Move the data from CPU to GPU
    inputs = inputs.cuda()
    targets = targets.cuda()

    # Inference
    outputs = model(inputs)

    # Convert logits to class indices
    outputs = outputs.argmax(dim=1)

    # Update metrics
    num_samples += targets.size(0)
    num_correct += (outputs == targets).sum()

  return (num_correct / num_samples * 100).item()


def get_model_size(model: nn.Module, data_width=32):
    """
    calculate the model size in bits
    :param data_width: #bits per element
    """
    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()
    return num_elements * data_width

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB
# --------------------------------------------------------------- #

def main(device, total_epochs, save_every, batch_size):
    dataset, model, optimizer, testdata, scheduler = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, device, save_every, scheduler=scheduler)
    start_time = time.time()
    trainer.train(total_epochs)
    end_time = time.time()

    training_time = end_time - start_time
    print(f"Total training time: {training_time:.2f} seconds")
    fp32_model_size = get_model_size(model)
    print(f"fp32 model has size={fp32_model_size/MiB:.2f} MiB")
    
    test_data = DataLoader(
       testdata,
       batch_size = 512,
       shuffle = False,
       pin_memory=True,
       num_workers=0,
    )
    fp32_model_accuracy = evaluate(model, test_data)
    print(f"fp32 model has accuracy={fp32_model_accuracy:.2f}%")
    



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=512, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    device = 0  # shorthand for cuda:0
    main(device, args.total_epochs, args.save_every, args.batch_size)
