from torchvision.datasets import GTSRB

train_dataset = GTSRB(root='data', split='train', download=True)
#test_dataset = GTSRB(root='data', split='test', download=True)