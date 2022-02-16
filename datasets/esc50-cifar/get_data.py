from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.transforms import transforms


def get_dataloader(data_dir, batch_size=40, num_workers=8, train_shuffle=True):
    input_size = 448
    tfms = transforms.Compose(
        [transforms.Resize((input_size, input_size))])
    tfms.transforms.append(transforms.ToTensor())
    
    train_val_set = CIFAR100(data_dir, train=True, transform=tfms, target_transform=None)
    num_train = int(len(train_val_set)*0.9)
    train_set = train_val_set[:num_train]
    valid_set = train_val_set[num_train:]
    test_set = CIFAR100(data_dir, train=False, transform=tfms, target_transform=None)

    valids = DataLoader(valid_set, shuffle=False, num_workers=num_workers, batch_size=batch_size)
    tests = DataLoader(test_set, shuffle=False, num_workers=num_workers, batch_size=batch_size)
    trains = DataLoader(train_set, shuffle=train_shuffle, num_workers=num_workers, batch_size=batch_size)
    return trains,valids,tests
