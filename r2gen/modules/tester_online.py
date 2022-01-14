import os
from abc import abstractmethod

import time
import torch
import pandas as pd
from numpy import inf


class BaseTester(object):
    def __init__(self, model, optimizer, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _test_epoch(self, epoch, test_dataloader, model_path): 
        # 抽象方法,在基类中的一个方法,没有实现,因此基类不能实例化,子类只有实现的
        # 该抽象方法才能被实例化
        raise NotImplementedError

    def test(self, test_dataloader, model_path):
        not_improved_count = 0
        epoch = 1
        result = self._test_epoch(epoch, test_dataloader, model_path)
        return result


    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids
    
    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])



class Tester(BaseTester):
    def __init__(self, model, optimizer, args, lr_scheduler):
        super(Tester, self).__init__(model, optimizer, args)
        self.lr_scheduler = lr_scheduler

    def _test_epoch(self, epoch, test_dataloader, model_path):
        self._resume_checkpoint(model_path)
        self.model.eval()

        with torch.no_grad():
            test_gts, test_res = [], []
            images = test_dataloader.out()
            # print(images)
            images = images.to(self.device)
            output = self.model(images, mode='sample')
            # print(output)
            reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
            return reports[0]