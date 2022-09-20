"""
@author: Junguang Jiang, Baixu Chen
@contact: JiangJunguang1123@outlook.com, cbx_99_hasta@outlook.com
"""
import sys
import os
import os.path as osp
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
from tllib.utils.metric import accuracy, ConfusionMatrix
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.vision.datasets.imagelist import MultipleDomainsDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix, matthews_corrcoef, classification_report,confusion_matrix, accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score,  precision_score, recall_score


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


def get_dataset(dataset_name, root, source, target, validation,
                train_source_transforms, val_transforms, train_target_transforms=None):

    if train_target_transforms is None:
        train_target_transforms = train_source_transforms

    # load datasets from tllib.vision.datasets
    dataset = datasets.__dict__[dataset_name]

    def concat_dataset(tasks, start_idx, transforms, **kwargs):
        # return ConcatDataset([dataset(task=task, **kwargs) for task in tasks])
        domains = [dataset(task=task, transform=transform, **kwargs) for task, transform in zip(tasks, transforms)]
        return MultipleDomainsDataset(domains, tasks, domain_ids=list(range(start_idx, start_idx + len(tasks))))

    train_source_dataset = concat_dataset(root=root, tasks=source, transforms=train_source_transforms, start_idx=0)
    train_target_dataset = concat_dataset(root=root, tasks=target, transforms=train_target_transforms,
                                          start_idx=len(source))
    val_dataset = concat_dataset(root=root, tasks=validation, transforms=val_transforms,
                                 start_idx=len(source) + len(target))
    test_dataset = val_dataset

    class_names = train_source_dataset.datasets[0].classes
    num_classes = len(class_names)

    return train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names


def validate(loader, classifier, args, device, discriminator=None):

    # initialize at None, will stay None in case no discriminator is provided
    y_pred_domain = None

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    classifier.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    y_pred_label = torch.empty((0,))
    y_true = torch.empty((0,))

    if discriminator is not None:
        y_pred_domain = torch.empty((0,))

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(loader):

            images, target = data[:2]
            images = images.to(device)
            target = target.to(device)

            # compute output
            output_label, features = classifier(images)
            loss = F.cross_entropy(output_label, target)

            y_pred_label = torch.cat([y_pred_label, output_label.cpu()])
            y_true = torch.cat([y_true, target.cpu()])

            if discriminator is not None:
                output_domain = discriminator(features)
                y_pred_domain = torch.cat([y_pred_domain, output_domain.cpu()])

            # measure accuracy and record loss
            acc1, = accuracy(output_label, target, topk=(1,))
            if confmat:
                confmat.update(target, output_label.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        if confmat:
            print(confmat.format(args.class_names))

    return acc1, y_true, y_pred_label, y_pred_domain


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
                                        crop_size=crop_sizes[validation]) for validation in args.validation]

    print("train_source_transforms: ", train_source_transforms)
    print("train_target_transforms: ", train_target_transforms)
    print("val_transforms: ", val_transforms)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        get_dataset(args.data, args.root, args.source, args.target, args.validation,
                    train_source_transforms=train_source_transforms,
                    val_transforms=val_transforms,
                    train_target_transforms=train_target_transforms)

    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    return num_classes, train_source_loader, train_target_loader, val_loader, test_loader


def classification_complete_report(y_true, y_pred, directory, filename, labels=None):

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

    with open(os.path.join(directory, filename), "w") as f:
        f.write(output)
        f.close()

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    fig, ax = plt.subplots(figsize=(10, 10))  # plot size
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax, include_values=False, colorbar=False)

    fig.savefig(os.path.join(directory, 'confusion_matrix.png'))


def report(args, device, classifier, directory, test_loader, train_source_loader, train_target_loader, domain_discriminator=None):

    labels = datasets.__dict__[args.data].CLASSES

    def report_domain_data(loader, domain):

        acc1, y_true, y_pred_label, y_pred_domain = validate(loader, classifier, args, device, domain_discriminator)
        y_pred_label = torch.argmax(torch.softmax(y_pred_label, dim=1), 1).numpy()

        if y_pred_domain is not None:
            df = pd.DataFrame(y_pred_domain.cpu().numpy())
            df.to_csv(os.path.join(directory, f"domain_{domain}.txt"), header=None, index=None)

        # save predictions for the unlabelled validation/test dataset
        if domain == "test":
            df = pd.read_csv(os.path.join(args.root, 'image_list/wbc.txt'), sep=' ', header=None, names=['Image', 'LabelID', 'Weights'])
            df['Image'] = df['Image'].apply(lambda x: x[14:])
            df['LabelID'] = y_pred_label
            df['Label'] = df['LabelID'].apply(lambda x: labels[x])
            df = df[['Image', 'LabelID', 'Label']]
            df.to_csv(os.path.join(directory, 'predictions.csv'))

        # report classification matrix for the labelled datasets
        if domain in ['source', 'target']:
            y_true = [labels[int(x)] for x in y_true.numpy()]
            y_pred_label = [labels[x] for x in y_pred_label]
            classification_complete_report(y_true, y_pred_label, directory, f"report_{domain}.txt", labels=labels)

    report_domain_data(test_loader, "test")
    report_domain_data(train_source_loader, "source")
    report_domain_data(train_target_loader, "target")
