import torch
from dorefa_classifier import DorefaClassifier
from cifar10 import load_train_data, load_test_data
from dorefa_resnet import dorefa_resnet18
import argparse

parser = argparse.ArgumentParser(description='ML_CODESIGN Lab3 - CNN example')
parser.add_argument('--batch-size', type=int, default=128, help='Number of samples per mini-batch')
parser.add_argument('--test-batch-size', type=int, default=100, help='Number of samples per mini-batch for testing')
parser.add_argument('--epochs', type=int, default=30, help='Number of epoch to train')
parser.add_argument('--w', type=int, default=1, help='Number of weight bits')
parser.add_argument('--a', type=int, default=1, help='Number of activation bits')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--enable-cuda', type=bool, default=1, help='Enable traning on gpu')

args = parser.parse_args()

# Parameters
num_epochs = args.epochs
checkpoint = "results/dorefa_resnet_cifar10/8and8_precision" 
steps = [80, 150] 
batch_size = args.batch_size
test_batch_size = args.test_batch_size
dataset = "cifar10"
wbit = args.w
abit = args.a

# Device settings (GPU/CPU)
enable_cuda = args.enable_cuda
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
print('Train using', device)

torch.manual_seed(0)
if device == 'cuda':
    torch.backends.cudnn.deterministic=True
    torch.cuda.manual_seed(0)

# CIFAR10 Dataset (Images and Labels)
train_loader = load_train_data(batch_size)
test_loader = load_test_data(test_batch_size)

model = dorefa_resnet18(wbit=wbit, abit=abit).to(device)
model.init_w()

classification = DorefaClassifier(model, train_loader, test_loader, device)

criterion = torch.nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                            weight_decay=5.e-4)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=0.1)

classification.train(criterion, optimizer, num_epochs, scheduler, checkpoint)
