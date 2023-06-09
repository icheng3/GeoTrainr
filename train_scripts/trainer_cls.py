import os
import datetime
import numpy as np
import json
import time

import torch
import torch.backends.cudnn as cudnn
from timm.utils import accuracy

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from libs.ConvNeXt import utils, build_dataset, create_optimizer
import libs.ConvNeXt.utils as utils
from libs.ConvNeXt.models.convnext import ConvNeXtFeature, ConvNeXt
import libs.bit_pytorch.models as bit_models
from geo_data import build_geo_dataset
from transformers import BitForImageClassification

class Trainer(object):
    def __init__(self, args):
        utils.init_distributed_mode(args)
        # print(args)
        self.device = torch.device(args.device)
        # fix the seed for reproducibility
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = True

        self.args = args

        self.load_data()
        self.build_model()


    def train_epoch(self, epoch, criterion, optimizer, data_loader, 
                        start_steps, update_freq, num_training_steps_per_epoch, 
                        lr_schedule_values):
        self.model.train(True)
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 10

        optimizer.zero_grad()

        for data_iter_step, (samples, _, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            step = data_iter_step // update_freq
            if step >= num_training_steps_per_epoch:
                continue
            it = start_steps + step  # global training iteration
            # Update LR & WD for the first acc
            if lr_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]

            samples = samples.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            output = self.model(samples)
            if self.args.model == "bit_base":
                output = output.logits
            
            loss = criterion(output, targets)

            loss_value = loss.item()

            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()

            torch.cuda.synchronize()
            class_acc = (output.max(-1)[-1] == targets).float().mean()

            acc1, acc5 = accuracy(output, targets, topk=(1, 5))

            batch_size = samples.shape[0]
            # metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

            metric_logger.update(loss=loss_value)
            metric_logger.update(class_acc=class_acc)
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
                self.log_writer.update(class_acc=class_acc, head="loss")
                self.log_writer.update(lr=max_lr, head="opt")
                self.log_writer.update(min_lr=min_lr, head="opt")
                self.log_writer.update(weight_decay=weight_decay_value, head="opt")
                self.log_writer.set_step()

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    @torch.no_grad()
    def evaluate(self, data_loader):
        criterion = torch.nn.CrossEntropyLoss()

        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test:'

        # switch to evaluation mode
        self.model.eval()
        for batch in metric_logger.log_every(data_loader, 10, header):
            images = batch[0]
            target = batch[-1]

            images = images.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            # compute output
            output = self.model(images)
            if self.args.model == "bit_base":
                output = output.logits
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


    def train(self):
        args = self.args

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        ### build training setting
        total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
        num_training_steps_per_epoch = len(self.dataset_train) // total_batch_size
        print("LR = %.8f" % args.lr)
        print("Batch size = %d" % total_batch_size)
        print("Update frequent = %d" % args.update_freq)
        print("Number of training examples = %d" % len(self.dataset_train))
        print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)

        optimizer = create_optimizer(args, self.model, skip_list=None)
        lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        )

        mixup_fn = None
        if mixup_fn is not None:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        utils.auto_load_model(
            args=args, model=self.model, model_without_ddp=self.model,
            optimizer=optimizer, loss_scaler=None, model_ema=None)
        max_accuracy = 0.0

        print("Start training for %d epochs" % args.epochs)
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                self.data_loader_train.sampler.set_epoch(epoch)
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
            print(f"Accuracy of the model on the {len(self.dataset_val)} test images: {test_stats['acc1']:.1f}%")
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=self.model, optimizer=optimizer, epoch="best")
            print(f'Max accuracy: {max_accuracy:.2f}%')

            if self.log_writer is not None:
                self.log_writer.update(test_acc1=test_stats['acc1'], head="perf", step=epoch)
                self.log_writer.update(test_acc5=test_stats['acc5'], head="perf", step=epoch)
                self.log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}


            if args.output_dir and utils.is_main_process():
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
            self.model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024],
                                num_classes=args.nb_classes,
                                drop_path_rate=args.drop_path, 
                                layer_scale_init_value=args.layer_scale_init_value, 
                                head_init_scale=args.head_init_scale)
            if args.finetune: ### should always be true
                checkpoint_model = torch.load(args.finetune, map_location='cpu')['model']
                state_dict = self.model.state_dict()
                # print(missing_keys)

                for k in ['head.weight', 'head.bias']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        # print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]
                # utils.load_state_dict(self.model, checkpoint_model, prefix=args.model_prefix)
                missing_keys, _ = self.model.load_state_dict(checkpoint_model, strict=False)
                print("missed_keys:", missing_keys)

        if args.model == "bit_base":
            self.model = BitForImageClassification.from_pretrained(args.finetune)
            self.model.classifier = torch.nn.Sequential(torch.nn.Flatten(start_dim=1, end_dim=-1),
                                                        torch.nn.Linear(2048, args.nb_classes, bias=True))
            # for name, param in self.model.named_parameters():
            #     if "classifier" not in name:
            #         param.requires_grad = False

        if args.model == "BiT-M-R101x1":
            # wget https://storage.googleapis.com/bit_models/vtab/BiT-M-R101x1-run2-caltech101.npz
            self.model = bit_models.KNOWN_MODELS[args.model](head_size=args.nb_classes, zero_head=True)
            if len(args.finetune)>0:
                self.model.load_from(np.load(args.finetune))
            
        if len(args.resume)>0: ### for eval
            print(f"resuming checkpoint at {args.resume}")
            checkpoint_model = torch.load(args.resume, map_location='cpu')['model']
            missing_keys, _ = self.model.load_state_dict(checkpoint_model, strict=True)
            print(missing_keys)

        self.model.to(self.device)
        self.args = args



    def load_data(self):
        args = self.args

        ### build dataset
        self.dataset_train, self.dataset_val, args.nb_classes = build_geo_dataset(args=args)

        ### number of labels
        print("Num of labels = " + str(len(self.dataset_train.cls)))

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