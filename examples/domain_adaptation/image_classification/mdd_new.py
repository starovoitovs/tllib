"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import os
import random
import warnings
import argparse

import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import TorchCheckpointIO
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, BatchSampler, RandomSampler
from torchmetrics.functional import accuracy
from torchvision.datasets import ImageFolder

import mlflow.pytorch
import mlflow

import utils
from tllib.alignment.mdd import ClassificationMarginDisparityDiscrepancy as MarginDisparityDiscrepancy, ImageClassifier
import pytorch_lightning as pl

NUM_CLASSES = 11

CLASSES = ['basophil', 'eosinophil', 'erythroblast', 'lymphocyte_typical', 'metamyelocyte', 'monocyte', 'myeloblast',
           'myelocyte', 'neutrophil_banded', 'neutrophil_segmented', 'promyelocyte']


def main(args: argparse.Namespace):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    best_entropy_checkpoint = ModelCheckpoint(filename='best_entropy', monitor='val_entropy', save_top_k=1)
    best_snd_checkpoint = ModelCheckpoint(filename='best_snd', monitor='val_snd', save_top_k=1)
    last_model = ModelCheckpoint(save_last=True, every_n_epochs=0)

    model_checkpoints = [best_entropy_checkpoint, best_snd_checkpoint, last_model]

    model = MDD(args=args, num_classes=NUM_CLASSES)
    dm = WBCDataModule(args)

    # Auto log all MLflow entities
    mlflow.pytorch.autolog(log_models=False)

    with mlflow.start_run(run_id=args.run_id) as run:

        class MLFlowCheckpointIO(TorchCheckpointIO):
            def save_checkpoint(self, checkpoint, path, storage_options=None):
                super().save_checkpoint(checkpoint, path, storage_options)
                mlflow.log_artifact(path, 'checkpoints')

        trainer = pl.Trainer(max_epochs=args.epochs, callbacks=[*model_checkpoints], devices=args.devices,
                             accelerator="auto")
        trainer.strategy.checkpoint_io = MLFlowCheckpointIO()

        if args.phase == 'train':
            trainer.fit(model, dm)

        repository = get_artifact_repository(run.info.artifact_uri)
        best_model_path = os.path.join(repository._artifact_dir, 'checkpoints/best_entropy.ckpt')
        print(f"Loading best model from {best_model_path}")
        model.load_from_checkpoint(best_model_path, num_classes=NUM_CLASSES, args=args)

        # need zero here for some reason
        output = trainer.predict(model, dm)[0]
        logits = torch.softmax(output, dim=1)
        predictions = torch.argmax(logits, dim=1)

        df = pd.read_csv(os.path.join(args.root, 'image_list/wbc.txt'), sep=' ', header=None,
                         names=['Image', 'LabelID'])
        df['Image'] = df['Image'].apply(lambda x: x[14:])
        df['LabelID'] = predictions.numpy()
        df['Label'] = df['LabelID'].apply(lambda x: CLASSES[x])
        df = df[['Image', 'LabelID', 'Label']]
        df.to_csv('predictions.csv')
        mlflow.log_artifact('predictions.csv')

        # save logits
        pd.DataFrame(logits.numpy()).to_csv('logits.csv', header=False, index=False)
        mlflow.log_artifact('logits.csv')

        # @todo test, predictions, logits, report, metrics, save model checkpoints
        # [ ] record stdout
        # [ ] problem with gpus=4 and multiprocessing: cannot pickle
        # [ ] make sure classes are ordered correctly
        # [ ] record which epoch model was best


class WBCDataModule(pl.LightningDataModule):

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

    def setup(self, stage):

        ace_transforms = utils.get_train_transform(self.args.train_resizing, scale=self.args.scale,
                                                   ratio=self.args.ratio,
                                                   random_horizontal_flip=not self.args.no_hflip,
                                                   random_color_jitter=False, resize_size=self.args.resize_size,
                                                   norm_mean=self.args.norm_mean, norm_std=self.args.norm_std,
                                                   crop_size=250)
        self.ace_dataset = ImageFolder(os.path.join(self.args.root, 'Acevedo_20'), transform=ace_transforms)

        mat_transforms = utils.get_train_transform(self.args.train_resizing, scale=self.args.scale,
                                                   ratio=self.args.ratio,
                                                   random_horizontal_flip=not self.args.no_hflip,
                                                   random_color_jitter=False, resize_size=self.args.resize_size,
                                                   norm_mean=self.args.norm_mean, norm_std=self.args.norm_std,
                                                   crop_size=345)
        self.mat_dataset = ImageFolder(os.path.join(self.args.root, 'Matek_19'), transform=mat_transforms)

        wbc_transforms = utils.get_val_transform(self.args.val_resizing, resize_size=self.args.resize_size,
                                                 norm_mean=self.args.norm_mean, norm_std=self.args.norm_std,
                                                 crop_size=288)
        self.wbc_dataset = ImageFolder(os.path.join(self.args.root, 'WBC1'), transform=wbc_transforms)

    def train_dataloader(self):

        # custom batch samplers ensure that the number of samples is in sync
        source_dataset = ConcatDataset([self.ace_dataset, self.mat_dataset])
        source_batch_sampler = BatchSampler(
            RandomSampler(source_dataset, num_samples=self.args.batch_size * self.args.iters_per_epoch, replacement=True),
            batch_size=self.args.batch_size, drop_last=True)
        train_source_loader = DataLoader(source_dataset, num_workers=self.args.workers,
                                         batch_sampler=source_batch_sampler)

        target_dataset = self.wbc_dataset
        target_batch_sampler = BatchSampler(
            RandomSampler(target_dataset, num_samples=self.args.batch_size * self.args.iters_per_epoch,
                          replacement=True),
            batch_size=self.args.batch_size, drop_last=True)
        train_target_loader = DataLoader(self.wbc_dataset, num_workers=self.args.workers,
                                         batch_sampler=target_batch_sampler)

        return {"source": train_source_loader, "target": train_target_loader}

    def val_dataloader(self):
        # no batching due to SND
        return DataLoader(self.wbc_dataset, batch_size=len(self.wbc_dataset), num_workers=self.args.workers)

    def test_dataloader(self):
        # no batching due to SND
        return DataLoader(self.wbc_dataset, batch_size=len(self.wbc_dataset), num_workers=self.args.workers)

    def predict_dataloader(self):
        # no batching due to predicting all
        return DataLoader(self.wbc_dataset, batch_size=len(self.wbc_dataset), num_workers=self.args.workers)


class MDD(pl.LightningModule):

    def __init__(self, args, num_classes):
        super(MDD, self).__init__()
        self.args = args
        self.num_classes = num_classes

        backbone = utils.get_model(self.args.arch, pretrain=not self.args.scratch)
        pool_layer = nn.Identity() if self.args.no_pool else None

        self.classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=self.args.bottleneck_dim,
                                          width=self.args.bottleneck_dim, pool_layer=pool_layer)
        self.mdd = MarginDisparityDiscrepancy(self.args.margin)

    def forward(self, x):
        return self.classifier(x)

    # The learning rate of the classifiers are set 10 times to that of the feature extractor by default.
    def configure_optimizers(self):
        optimizer = SGD(self.classifier.get_parameters(), self.args.lr,
                        momentum=self.args.momentum, weight_decay=self.args.wd, nesterov=True)
        lr_scheduler = LambdaLR(optimizer,
                                lambda x: self.args.lr * (1. + self.args.lr_gamma * float(x)) ** (-self.args.lr_decay))
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def training_step(self, train_batch, batch_id):
        # w_s are source sample weights
        x_s, labels_s = train_batch['source']
        x_t, labels_t = train_batch['target']

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        outputs, outputs_adv, features = self.classifier(x)
        y_s, y_t = outputs.chunk(2, dim=0)
        y_s_adv, y_t_adv = outputs_adv.chunk(2, dim=0)

        # compute cross entropy loss on source domain
        cls_loss = F.cross_entropy(y_s, labels_s)
        # compute margin disparity discrepancy between domains
        # for adversarial classifier, minimize negative mdd is equal to maximize mdd
        transfer_loss = -self.mdd(y_s, y_s_adv, y_t, y_t_adv)
        loss = cls_loss + transfer_loss * self.args.trade_off
        cls_acc = accuracy(y_s, labels_s)

        self.log('train_cls_loss', cls_loss)
        self.log('train_transfer_loss', transfer_loss)
        self.log('train_loss', loss)
        self.log('train_cls_acc', cls_acc)

        return loss

    def validation_step(self, val_batch, batch_id):
        # w_s are source sample weights
        x_t, labels_t = val_batch
        outputs, outputs_adv, features = self.classifier(x_t)

        logits = torch.softmax(outputs, dim=1)
        entropy = F.cross_entropy(logits, logits, reduction='none').mean()

        self.log('val_entropy', entropy)

        # important factor to tell apart the neighborhoods
        normalized = F.normalize(logits).cpu()
        mat = torch.matmul(normalized, normalized.t()) / self.args.temperature
        mask = torch.eye(mat.size(0), mat.size(0)).bool()
        mat.masked_fill_(mask, -1 / self.args.temperature)
        mat = F.softmax(mat, dim=1)

        # set minus to minimize
        ent_soft = -F.cross_entropy(mat, mat, reduction='none').mean()
        self.log('val_snd', ent_soft)

        return entropy

    def predict_step(self, batch, batch_idx):
        x, y = batch
        output, output_adv, features = self.classifier(x)
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MDD for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--no-hflip', action='store_true', help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+', default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+', default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=1024, type=int)
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--margin', type=float, default=4., help="margin gamma")
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.004, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0002, type=float)
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--devices', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='mdd',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    parser.add_argument('--run-id', default=None, type=str, help='load specific mlflow run, for example for inference')
    parser.add_argument('--temperature', default=0.05, type=float, help='SND temperature')
    args = parser.parse_args()
    main(args)
