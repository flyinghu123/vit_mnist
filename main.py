from torchvision import datasets
from torchvision import transforms
from torchvision.models.vision_transformer import VisionTransformer
from tqdm import tqdm
import math
import numpy as np
import os
import torch
import torch.nn as nn

# 99.4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = 'mnist'
num_workers = 4
batch_size = 128
epochs = 200
image_size = 28
patch_size = 4
lr = 5e-4

def init_model(model):
    if isinstance(model.conv_proj, nn.Conv2d):
        # Init the patchify stem
        fan_in = model.conv_proj.in_channels * model.conv_proj.kernel_size[0] * model.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(model.conv_proj.weight, std=math.sqrt(1 / fan_in))
        if model.conv_proj.bias is not None:
            nn.init.zeros_(model.conv_proj.bias)
    if hasattr(model.heads, "pre_logits") and isinstance(model.heads.pre_logits, nn.Linear):
            fan_in = model.heads.pre_logits.in_features
            nn.init.trunc_normal_(model.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(model.heads.pre_logits.bias)
    if isinstance(model.heads.head, nn.Linear):
        nn.init.zeros_(model.heads.head.weight)
        nn.init.zeros_(model.heads.head.bias)

def get_model():
    model = VisionTransformer(image_size, patch_size, 6, 8, 128, 128 * 2)
    model.conv_proj = nn.Conv2d(
        in_channels=1, out_channels=model.hidden_dim, kernel_size=model.patch_size, stride=model.patch_size
    )
    model.heads[-1] = nn.Linear(model.heads[-1].in_features, 10)
    # init_model(model)  # optional
    return model.to(device)

def get_loader():
    if dataset == 'mnist':
        tr_transform = transforms.Compose([transforms.RandomCrop(image_size, padding=2), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.5], [0.5])])
        train = datasets.MNIST(os.path.join('data', dataset), train=True, download=True, transform=tr_transform)

        te_transform = transforms.Compose([transforms.Resize([image_size, image_size]), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        test = datasets.MNIST(os.path.join('data', dataset), train=False, download=True, transform=te_transform)

    elif dataset == 'fashion':
        tr_transform = transforms.Compose([transforms.RandomCrop(image_size, padding=2), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.5], [0.5])])
        train = datasets.FashionMNIST(os.path.join('data', dataset), train=True, download=True, transform=tr_transform)

        te_transform = transforms.Compose([transforms.Resize([image_size, image_size]), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        test = datasets.FashionMNIST(os.path.join('data', dataset), train=False, download=True, transform=te_transform)

    else:
        raise Exception('unsupported data')

    train_loader = torch.utils.data.DataLoader(dataset=train,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=num_workers,
                                                 drop_last=True)

    test_loader = torch.utils.data.DataLoader(dataset=test,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=num_workers,
                                                drop_last=False)
    return train_loader, test_loader


@torch.no_grad()
def test():
    model.eval()

    actual = []
    pred = []

    for (imgs, labels) in test_loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            class_out = model(imgs)
        _, predicted = torch.max(class_out.data, 1)

        actual += labels.tolist()
        pred += predicted.tolist()
    acc = (np.array(actual) == np.array(pred)).mean() * 100
    return acc

def get_optimizer():
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    return optimizer, lr_scheduler

def get_loss_fn():
    return nn.CrossEntropyLoss()

def train():
    with tqdm(epochs, 'train', epochs) as t_par:
        for epoch in range(epochs):
            model.train()
            for (imgs, labels) in tqdm(train_loader, desc=f'Epoch: {epoch}', total=len(train_loader), leave=False):
                imgs, labels = imgs.cuda(), labels.cuda()
                logits = model(imgs)
                clf_loss = loss_fn(logits, labels)
                optimizer.zero_grad()
                clf_loss.backward()
                optimizer.step()
            acc = test()
            t_par.set_description(f'test acc: {acc:.2f}%')
            t_par.update()

model = get_model()
train_loader, test_loader = get_loader()
optimizer, lr_scheduler = get_optimizer()
loss_fn = get_loss_fn()
train()
