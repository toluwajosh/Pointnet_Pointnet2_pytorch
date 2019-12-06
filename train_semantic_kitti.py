"""
Example:
python train_semantic_kitti.py --model_name pointnet --pretrain ./out/working_model.pth --learning_rate 1e-4 --epoch 200 --batchsize 4
"""


import argparse
import os
import torch
import numpy as np
import torch.nn.parallel
import torch.utils.data
from collections import defaultdict
from torch.autograd import Variable
from data_utils.KittiDataLoader import KittiOdometryLoader
import torch.nn.functional as F
import datetime
import logging
from pathlib import Path
from utils import test_semantic_kitti
from tqdm import tqdm
from model.pointnet2 import PointNet2SemSeg
from model.pointnet_max import PointNetSeg, feature_transform_reguliarzer

from content import weight_list, categories_dict

# seg_classes = class2label
# seg_label_to_cat = {}
# for i, cat in enumerate(seg_classes.keys()):
#     seg_label_to_cat[i] = cat

seg_label_to_cat = categories_dict()
loss_weights = torch.from_numpy(np.array(weight_list())).float().cuda()
# print(loss_weights)
# exit(0)


def parse_args():
    parser = argparse.ArgumentParser("PointNet")
    parser.add_argument("--batchsize", type=int, default=12, help="input batch size")
    parser.add_argument(
        "--workers", type=int, default=4, help="number of data loading workers"
    )
    parser.add_argument(
        "--epoch", type=int, default=200, help="number of epochs for training"
    )
    parser.add_argument(
        "--pretrain", type=str, default=None, help="whether use pretrain model"
    )
    parser.add_argument("--gpu", type=str, default="0", help="specify gpu device")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate for training"
    )
    parser.add_argument("--decay_rate", type=float, default=1e-4, help="weight decay")
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="type of optimizer"
    )
    parser.add_argument(
        "--multi_gpu", type=str, default=None, help="whether use multi gpu training"
    )
    parser.add_argument(
        "--model_name", type=str, default="pointnet2", help="Name of model"
    )

    return parser.parse_args()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        args.gpu if args.multi_gpu is None else "0,1,2,3"
    )
    """CREATE DIR"""
    experiment_dir = Path("./experiment/")
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(
        str(experiment_dir)
        + "/%sSemSeg-" % args.model_name
        + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    )
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath("checkpoints/")
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath("logs/")
    log_dir.mkdir(exist_ok=True)

    """LOG"""
    args = parse_args()
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler(
        str(log_dir) + "/train_%s_semseg.txt" % args.model_name
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(
        "---------------------------------------------------TRANING---------------------------------------------------"
    )
    logger.info("PARAMETER ...")
    logger.info(args)
    print("Load data...")
    data_dir = "/media/tjosh/ssd_vault/kitti_odometry/dataset"
    num_classes = 20
    model_magnifier = 2
    points_size = 4096 * 2 * model_magnifier

    # debug_seq = ['02', '03', '04']
    # debug_seq = ['03']
    train_data = KittiOdometryLoader(
        data_dir, sequence="train", classes=num_classes, points_size=points_size
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, shuffle=True, batch_size=args.batchsize
    )
    valid_data = KittiOdometryLoader(
        data_dir, sequence="valid", classes=num_classes, points_size=points_size
    )
    valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=True, batch_size=1)

    def blue(x):
        return "\033[94m" + x + "\033[0m"

    model = (
        PointNet2SemSeg(num_classes)
        if args.model_name == "pointnet2"
        else PointNetSeg(
            num_classes, feature_transform=True, semseg=True, magnifier=model_magnifier
        )
    )

    if args.pretrain is not None:
        trained_weights = torch.load(args.pretrain)
        # filter model weights for partial loading
        trained_weights = {
            k: v
            for k, v in trained_weights.items()
            if (k in model.state_dict())
            and (model.state_dict()[k].shape == trained_weights[k].shape)
        }
        model.load_state_dict(trained_weights, strict=False)
        print("load model %s" % args.pretrain)
        logger.info("load model %s" % args.pretrain)
    else:
        print("Training from scratch")
        logger.info("Training from scratch")
    pretrain = args.pretrain
    # init_epoch = int(pretrain[-14:-11]) if args.pretrain is not None else 0
    init_epoch = 0

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate,
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    LEARNING_RATE_CLIP = 1e-5

    """GPU selection and multi-GPU"""
    if args.multi_gpu is not None:
        device_ids = [int(x) for x in args.multi_gpu.split(",")]
        torch.backends.cudnn.benchmark = True
        model.cuda(device_ids[0])
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        model.cuda()

    history = defaultdict(lambda: list())
    best_acc = 0
    best_meaniou = 0

    for epoch in range(init_epoch, args.epoch):
        scheduler.step()
        lr = max(optimizer.param_groups[0]["lr"], LEARNING_RATE_CLIP)
        print("Learning rate:%f" % lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        total_losses = []
        with tqdm(total=len(train_loader), smoothing=0.9) as pbar:
            for i, data in enumerate(train_loader, 0):
                points = data["velo"]
                target = data["label"]
                mask = data["mask"]
                torch.set_printoptions(edgeitems=50)

                points, target = (
                    Variable(points.float()),
                    Variable(target.float().long()),
                )
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                optimizer.zero_grad()
                model = model.train()
                if args.model_name == "pointnet":
                    pred, trans_feat = model(points)
                else:
                    pred = model(points[:, :3, :], points[:, 3:, :])
                pred = pred.contiguous().view(-1, num_classes)
                target = target.view(-1, 1)[:, 0]
                loss = F.nll_loss(pred, target, reduction="none", weight=loss_weights)

                # mask the unlabled classes
                mask = mask.view(-1, 1)[:, 0]
                loss = loss * mask.float().cuda()
                loss = torch.mean(loss)
                if args.model_name == "pointnet":
                    loss += feature_transform_reguliarzer(trans_feat) * 0.001
                history["loss"].append(loss.cpu().data.numpy())
                loss.backward()
                optimizer.step()

                total_losses.append(loss.item())

                pbar.set_description(
                    "Epoch: {:4d}; Iter: {:4d}; Total_loss: {:8.5f}.".format(
                        epoch + 1, i + 1, np.mean(total_losses)
                    )
                )
                pbar.update()
            if (epoch) % 1 == 0:
                torch.save(
                    model.state_dict(),
                    "./out/working_model_{}_{}.pth".format(
                        args.model_name, str(model_magnifier)
                    ),
                )
                #    "./out/working_model_{}.pth".format(epoch))

        # testing ->
        torch.cuda.empty_cache()
        pointnet2 = args.model_name == "pointnet2"
        test_metrics, test_hist_acc, cat_mean_iou = test_semantic_kitti(
            model.eval(),
            valid_loader,
            seg_label_to_cat,
            num_classes=num_classes,
            pointnet2=pointnet2,
        )
        mean_iou = np.mean(cat_mean_iou)
        print(
            "Epoch %d  %s accuracy: %f  meanIOU: %f"
            % (epoch, blue("test"), test_metrics["accuracy"], mean_iou)
        )
        logger.info(
            "Epoch %d  %s accuracy: %f  meanIOU: %f"
            % (epoch, "test", test_metrics["accuracy"], mean_iou)
        )
        if test_metrics["accuracy"] > best_acc:
            best_acc = test_metrics["accuracy"]
            torch.save(
                model.state_dict(),
                "%s/%s_%.3d_%.4f.pth"
                % (checkpoints_dir, args.model_name, epoch, best_acc),
            )
            logger.info(cat_mean_iou)
            logger.info("Save model..")
            print("Save model..")
            print(cat_mean_iou)
        if mean_iou > best_meaniou:
            best_meaniou = mean_iou
        print("Best accuracy is: %.5f" % best_acc)
        logger.info("Best accuracy is: %.5f" % best_acc)
        print("Best meanIOU is: %.5f" % best_meaniou)
        logger.info("Best meanIOU is: %.5f" % best_meaniou)


if __name__ == "__main__":
    args = parse_args()
    main(args)
