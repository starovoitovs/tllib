"""
@author: Junguang Jiang, Baixu Chen
@contact: JiangJunguang1123@outlook.com, cbx_99_hasta@outlook.com
"""
import sys
import os
import time
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from timm.data.auto_augment import auto_augment_transform, rand_augment_transform

sys.path.append('../../..')
import tllib.vision.datasets as datasets
import tllib.vision.models as models
from tllib.vision.transforms import ResizeImage
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.vision.datasets.imagelist import MultipleDomainsDataset

from sklearn.metrics import ConfusionMatrixDisplay, matthews_corrcoef, classification_report,confusion_matrix, accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score,  precision_score, recall_score


def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()


def get_model(model_name, pretrain=True):
    if model_name in models.__dict__:
        # load models from tllib.vision.models
        backbone = models.__dict__[model_name](pretrained=pretrain)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=pretrain)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone


def get_dataset_names():
    return sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    ) + ['Digits']


def get_dataset(dataset_name, root, source, target, train_source_transforms, val_transforms,
                train_target_transforms=None):
    if train_target_transforms is None:
        train_target_transforms = train_source_transforms

    # load datasets from tllib.vision.datasets
    dataset = datasets.__dict__[dataset_name]

    def concat_dataset(tasks, start_idx, transforms, **kwargs):
        domains = [dataset(task=task, transform=transform, **kwargs) for task, transform in zip(tasks, transforms)]
        return MultipleDomainsDataset(domains, tasks, domain_ids=list(range(start_idx, start_idx + len(tasks))))

    train_source_dataset = concat_dataset(root=root, tasks=source, transforms=train_source_transforms, start_idx=0)
    train_target_dataset = concat_dataset(root=root, tasks=target, transforms=train_target_transforms, start_idx=len(source))
    val_dataset = concat_dataset(root=root, tasks=target, transforms=val_transforms, start_idx=len(source))
    test_dataset = val_dataset

    class_names = train_source_dataset.datasets[0].classes
    num_classes = len(class_names)

    return train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names


# @todo note that due to SND we don't batch validation set, need the entire set to calculate neighborhoods
#  (thus metrics for batch train_source don't really make sense)
def validate(val_loader, model, args, device, epoch=None, reportsdir=None):

    # switch to evaluate mode
    model.eval()

    y_pred = torch.empty((0,))
    y_true = torch.empty((0,))

    with torch.no_grad():

        end = time.time()

        x_t, labels_t = next(iter(val_loader))[:2]
        x_t = x_t.to(device)
        labels_t = labels_t.to(device)

        # compute output
        outputs, outputs_adv = model(x_t)

        # attach outputs
        y_pred = torch.cat([y_pred, outputs.cpu()])
        y_true = torch.cat([y_true, labels_t.cpu()])

        # compute logits and entropy
        logits = torch.softmax(outputs, dim=1)
        entropy = F.cross_entropy(logits, logits, reduction='none').mean()

        # important factor to tell apart the neighborhoods
        normalized = F.normalize(logits).cpu()
        mat = torch.matmul(normalized, normalized.t()) / args.temperature
        mask = torch.eye(mat.size(0), mat.size(0)).bool()
        mat.masked_fill_(mask, -1 / args.temperature)
        mat = F.softmax(mat, dim=1)

        # set minus to minimize
        snd = -F.cross_entropy(mat, mat, reduction='none').mean()

        if reportsdir is not None:
            # save logits
            directory = os.path.join(reportsdir, 'logits')
            fname = os.path.join(directory, f'{epoch}.csv')
            os.makedirs(directory, exist_ok=True)
            pd.DataFrame(logits.cpu().numpy()).to_csv(fname, header=False, index=False)

        print(f'[VALIDATION] entropy: {entropy:7.4}; snd: {snd:7.4}')

    return y_pred, y_true, entropy, snd


def get_train_transform(resizing='default', scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), random_horizontal_flip=True,
                        random_color_jitter=False, resize_size=224, norm_mean=(0.485, 0.456, 0.406),
                        norm_std=(0.229, 0.224, 0.225), auto_augment=None, crop_size=None, random_vertical_flip=True):
    """
    resizing mode:
        - default: resize the image to 256 and take a random resized crop of size 224;
        - cen.crop: resize the image to 256 and take the center crop of size 224;
        - res: resize the image to 224;
    """
    transformed_img_size = 224
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224, scale=scale, ratio=ratio)
        ])
    elif resizing == 'cen.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224)
        ])
    elif resizing == 'ran.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomCrop(224)
        ])
    elif resizing == 'crop.resize':
        transform = T.Compose([
            T.CenterCrop(crop_size),
            T.RandomResizedCrop(224, scale=scale, ratio=ratio),
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
        transformed_img_size = resize_size
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if random_vertical_flip:
        transforms.append(T.RandomVerticalFlip())
    if auto_augment:
        aa_params = dict(
            translate_const=int(transformed_img_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in norm_mean]),
            interpolation=Image.BILINEAR
        )
        if auto_augment.startswith('rand'):
            transforms.append(rand_augment_transform(auto_augment, aa_params))
        else:
            transforms.append(auto_augment_transform(auto_augment, aa_params))
    elif random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
    return T.Compose(transforms)


def get_val_transform(resizing='default', resize_size=224,
                      norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225), crop_size=None):
    """
    resizing mode:
        - default: resize the image to 256 and take the center crop of size 224;
        – res.: resize the image to 224
    """
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    elif resizing == 'crop.resize':
        transform = T.Compose([
            T.CenterCrop(crop_size),
            ResizeImage(224),
        ])
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])


def empirical_risk_minimization(train_source_iter, model, optimizer, lr_scheduler, epoch, args, device):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)[:2]
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, f_s = model(x_s)

        cls_loss = F.cross_entropy(y_s, labels_s)
        loss = cls_loss

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def load_datasets(args):

    crop_sizes = datasets.__dict__[args.data].crop_sizes

    # Data loading code
    train_source_transforms = [get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                                   random_horizontal_flip=not args.no_hflip,
                                                   random_color_jitter=False, resize_size=args.resize_size,
                                                   norm_mean=args.norm_mean, norm_std=args.norm_std,
                                                   crop_size=crop_sizes[source]) for source in args.source]

    train_target_transforms = [get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                                   random_horizontal_flip=not args.no_hflip,
                                                   random_color_jitter=False, resize_size=args.resize_size,
                                                   norm_mean=args.norm_mean, norm_std=args.norm_std,
                                                   crop_size=crop_sizes[target]) for target in args.target]

    val_transforms = [get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                        norm_mean=args.norm_mean, norm_std=args.norm_std,
                                        crop_size=360) for target in args.target]

    print("train_source_transforms: ", train_source_transforms)
    print("train_target_transforms: ", train_target_transforms)
    print("val_transforms: ", val_transforms)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        get_dataset(args.data, args.root, args.source, args.target,
                    train_source_transforms=train_source_transforms,
                    val_transforms=val_transforms,
                    train_target_transforms=train_target_transforms)

    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)

    # hardcode big batch size for validation due to SND
    val_loader = DataLoader(val_dataset, len(val_dataset), shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, len(val_dataset), shuffle=False, num_workers=args.workers)

    return num_classes, test_loader, train_source_loader, train_target_loader, val_loader


def classification_complete_report(y_true, y_pred, directory, labels=None):

    output = ""
    output += classification_report(y_true, y_pred, labels=None) + "\n"
    output += 15 * "----" + "\n"
    output += "Matthews correlation coeff: %.4f" % (matthews_corrcoef(y_true, y_pred)) + "\n"
    output += "Cohen Kappa score:          %.4f" % (cohen_kappa_score(y_true, y_pred)) + "\n"
    output += "Accuracy:                   %.4f" % (accuracy_score(y_true, y_pred)) + "\n"
    output += "Balanced accuracy:          %.4f" % (balanced_accuracy_score(y_true, y_pred)) + "\n"
    output += 15 * "----" + "\n"
    output += "              macro    micro" + "\n"
    output += "Precision:   %.4f   %.4f" % (
    precision_score(y_true, y_pred, average="macro"), precision_score(y_true, y_pred, average="micro")) + "\n"
    output += "Recall:      %.4f   %.4f" % (
    recall_score(y_true, y_pred, average="macro"), recall_score(y_true, y_pred, average="micro")) + "\n"
    output += "F1:          %.4f   %.4f" % (
    f1_score(y_true, y_pred, average="macro"), f1_score(y_true, y_pred, average="micro")) + "\n"
    print(output)

    with open(os.path.join(directory, "report.txt"), "w") as f:
        f.write(output)
        f.close()

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    fig, ax = plt.subplots(figsize=(10, 10))  # plot size
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax, include_values=False, colorbar=False)

    fig.savefig(os.path.join(directory, 'confusion_matrix.png'))


def report(args, device, classifier, directory, test_loader, train_source_loader):

    labels = datasets.__dict__[args.data].CLASSES

    # save predictions for the test dataset
    y_pred, y_true, val_entropy, val_snd = validate(test_loader, classifier, args, device)
    soft = torch.softmax(y_pred, dim=1)
    y_pred = torch.argmax(soft, 1).numpy()

    df = pd.read_csv(os.path.join(args.root, 'image_list/wbc2.txt'), sep=' ', header=None, names=['Image', 'LabelID'])
    df['Image'] = df['Image'].apply(lambda x: x[15:])
    df['LabelID'] = y_pred
    df['Label'] = df['LabelID'].apply(lambda x: labels[x])
    df.to_csv(os.path.join(directory, 'test_predictions_best_snd.csv'))

    df = pd.DataFrame(soft.numpy())
    df.to_csv(os.path.join(directory, 'logits.csv'), header=False, index=False)

    # @todo classification report separate for ace and mat
    # generate classification report for the training set
    y_pred, y_true, val_entropy, val_snd = validate(train_source_loader, classifier, args, device)
    y_pred = torch.argmax(torch.softmax(y_pred, dim=1), 1).numpy()
    y_true = [labels[int(x)] for x in y_true.numpy()]
    y_pred = [labels[x] for x in y_pred]
    classification_complete_report(y_true, y_pred, directory, labels=labels)
