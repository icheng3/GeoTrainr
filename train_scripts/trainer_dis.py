import os
import datetime
import numpy as np
import json
import time
import sys

import torch
import torch.backends.cudnn as cudnn
from timm.utils import accuracy

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from libs.ConvNeXt import build_dataset, create_optimizer, create_optimizer_multi
import libs.ConvNeXt.utils as utils
from libs.ConvNeXt.models.convnext import ConvNeXtFeature, ConvNeXt
from geo_data import build_geo_dataset, anchor_samples, create_anchor_transform
from models.Distancer import GeoDiscriminator


class Trainer(object):
    def __init__(self, args):
        # print(args)
        utils.init_distributed_mode(args)
        # args.distributed = False
        self.device = torch.device(args.device)
        # fix the seed for reproducibility
        seed = args.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = True

        self.args = args

        self.build_model()
        self.load_data()
        sys.stdout.flush()


    def train_epoch(self, epoch, criterion, optimizer, data_loader, 
                        start_steps, update_freq, num_training_steps_per_epoch, 
                        lr_schedule_values):
        self.model.train(True)
        self.backbone.train(True)
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = f'Epoch: [{epoch}]'
        print_freq = 10

        optimizer.zero_grad()

        for data_iter_step, (samples, coords, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            step = data_iter_step // update_freq
            if step >= num_training_steps_per_epoch:
                continue
            it = start_steps + step  # global training iteration
            # Update LR & WD for the first acc
            if lr_schedule_values is not None:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]

            bs = samples.shape[0]
            assert bs%2==0, "batch size must be even number for eus_dis learning"
            samples = samples.to(self.device)
            coords = coords.to(self.device)

            # compute output
            features = self.backbone(samples)
            feature_distance = self.model(torch.cat([features[:bs//2], features[bs//2:]], dim=-1))*10
            geo_distance = torch.pairwise_distance(coords[:bs//2], coords[bs//2:], p=2, keepdim=True)

            loss = criterion(feature_distance, geo_distance)
            err = (geo_distance - feature_distance).abs().mean()

            loss_value = loss.item()

            loss /= update_freq
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)
            metric_logger.meters['err'].update(err.item(), n=bs)
            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)
            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
            metric_logger.update(weight_decay=weight_decay_value)

            if self.log_writer is not None:
                self.log_writer.update(loss=loss_value, head="loss")
                self.log_writer.update(err=err, head="error")
                self.log_writer.update(lr=max_lr, head="opt")
                self.log_writer.update(min_lr=min_lr, head="opt")
                self.log_writer.update(weight_decay=weight_decay_value, head="opt")
                self.log_writer.set_step()

        # gather the stats from all processes
        # metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    @torch.no_grad()
    def evaluate(self, data_loader):
        args = self.args
        if args.dis_criterion.lower()=="mse":
            criterion = torch.nn.MSELoss()
        elif args.dis_criterion.lower()=="l1":
            criterion = torch.nn.L1Loss()
        elif args.dis_criterion.lower()=="smoothl1":
            criterion = torch.nn.SmoothL1Loss()

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('err', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Test:'

        ref_num = self.anchor_images.shape[0]

        # switch to evaluation mode
        self.model.eval()
        self.backbone.eval()
        for batch in metric_logger.log_every(data_loader, 10, header):
            images = batch[0]
            coords = batch[1]
            bs = images.shape[0]
            assert bs%2==0, "batch size must be even number for eus_dis learning"

            images = images.to(self.device, non_blocking=True)
            coords = coords.to(self.device, non_blocking=True)
            coords_ref = self.anchor_coords.to(self.device, non_blocking=True)

            # compute output
            features_ref = self.backbone(self.anchor_images.to(self.device, non_blocking=True))
            features = self.backbone(images)

            loss, err = 0, 0
            for i in range(ref_num):
                feature_distance = self.model(torch.cat([features, features_ref[i:i+1].repeat(bs, 1)], dim=-1))*10
                geo_distance = torch.pairwise_distance(coords, coords_ref[i:i+1].repeat(bs, 1), p=2, keepdim=True)

                loss += criterion(feature_distance, geo_distance).item()/ref_num
                err  += (geo_distance - feature_distance).abs().mean().item()/ref_num

            metric_logger.update(loss=loss)
            metric_logger.meters['err'].update(err, n=bs)
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print('* Avg Error {err.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(err=metric_logger.err, losses=metric_logger.loss))

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


    def train(self):
        args = self.args

        n_parameters = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        n_parameters += sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        ### build training setting
        total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
        num_training_steps_per_epoch = len(self.dataset_train) // total_batch_size
        print("LR = %.8f" % args.lr)
        print("Batch size = %d" % total_batch_size)
        print("Update frequent = %d" % args.update_freq)
        print("Number of training examples = %d" % len(self.dataset_train))
        print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)

        optimizer = create_optimizer_multi(args, 
                                           [self.backbone, self.model], 
                                           skip_list=None)
        lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        )

        if args.dis_criterion.lower()=="mse":
            criterion = torch.nn.MSELoss()
        elif args.dis_criterion.lower()=="l1":
            criterion = torch.nn.L1Loss()
        elif args.dis_criterion.lower()=="smoothl1":
            criterion = torch.nn.SmoothL1Loss()

        min_error = 1000.0

        utils.auto_load_model(
            args=args, model=self.model, model_without_ddp=self.model,
            optimizer=optimizer, loss_scaler=None, model_ema=None)

        print("Start training for %d epochs" % args.epochs)
        sys.stdout.flush()
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            if self.log_writer is not None:
                self.log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)

            train_stats = self.train_epoch(epoch, criterion, optimizer, self.data_loader_train, 
                        epoch * num_training_steps_per_epoch, args.update_freq, num_training_steps_per_epoch, 
                        lr_schedule_values)
            
            if args.output_dir and args.save_ckpt:
                if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                    utils.save_model(
                        args=args, model=self.model, optimizer=optimizer,
                        epoch=epoch, model_ema=None)
                    
            ## start eval
            test_stats = self.evaluate(self.data_loader_val)
            print(f"Error of the model on the {len(self.dataset_val)} test images: {test_stats['err']*111:.1f} kilometers")
            if min_error > test_stats["err"]:
                min_error = test_stats["err"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=self.model, optimizer=optimizer, epoch="best")
            print(f'Min error: {min_error*111:.2f} kilometers')
            sys.stdout.flush()

            if self.log_writer is not None:
                self.log_writer.update(err=test_stats['err'], head="perf", step=epoch)
                self.log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)
                print("logging finish")

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}


            if args.output_dir:
                if self.log_writer is not None:
                    self.log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
    

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def build_model(self):
        args = self.args
        ### build model and load pretrained params
        if args.model == "convnext_base":
            self.backbone = ConvNeXtFeature(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024],
                                num_classes=args.nb_classes,
                                drop_path_rate=args.drop_path, 
                                layer_scale_init_value=args.layer_scale_init_value, 
                                head_init_scale=args.head_init_scale)

        if args.finetune: ### should always be true
            checkpoint_model = torch.load(args.finetune, map_location='cpu')['model']
            state_dict = self.backbone.state_dict()
            # print(missing_keys)

            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    # print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            # utils.load_state_dict(self.model, checkpoint_model, prefix=args.model_prefix)
            missing_keys, _ = self.backbone.load_state_dict(checkpoint_model, strict=False)
            print("missed_keys:", missing_keys)

        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        self.backbone.to(self.device)
        self.model = GeoDiscriminator(1024)
        self.model.to(self.device)
        self.args = args


    def load_data(self):
        args = self.args

        ### build dataset
        self.dataset_train, self.dataset_val, args.nb_classes = build_geo_dataset(args=args, trim=True)
        print("num workers: ", args.num_workers)
        ### build dataloaders
        sampler_train = torch.utils.data.RandomSampler(self.dataset_train, replacement=False)
        print("Sampler_train = %s" % str(sampler_train))
        self.data_loader_train = torch.utils.data.DataLoader(
            self.dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        if args.log_dir is not None:
            os.makedirs(args.log_dir, exist_ok=True)
            self.log_writer = utils.TensorboardLogger(log_dir=args.log_dir)

        sampler_val = torch.utils.data.RandomSampler(self.dataset_val, replacement=False)
        self.data_loader_val = torch.utils.data.DataLoader(
            self.dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        self.args = args

        self.anchor_images = []
        self.anchor_coords = []

        from PIL import Image
        anchor_transform = create_anchor_transform(args)
        for name in anchor_samples:
            _, coord = name[:-4].split("_")
            lat, lng = coord.split(",")
            latlng = np.array([float(lat), float(lng)])
            img_path = os.path.join(args.data_path, name)
            img = Image.open(img_path)
            self.anchor_images.append(anchor_transform(img))
            self.anchor_coords.append(torch.Tensor(latlng))

        self.anchor_images = torch.stack(self.anchor_images, 0)
        self.anchor_coords = torch.stack(self.anchor_coords, 0)