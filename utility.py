import os
import math
import pdb
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio
import cv2
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

class timer():
    def __init__(self):  # __init__(self)：初始化计时器对象，调用tic()方法开始计时。
        self.acc = 0
        self.tic()

    def tic(self):  # tic(self)：记录当前时间作为起始时间。
        self.t0 = time.time()

    def toc(self, restart=False):
        # toc(self, restart=False)：计算当前时间与起始时间的差值，表示经过的时间。如果restart参数为True，则重新记录起始时间。返回计时结果。
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):  # hold(self)：暂停计时，将经过的时间累加到总时间中。
        self.acc += self.toc()

    def release(self):  # release(self)：释放累积的总时间，并将总时间重置为0。返回累积的总时间。
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):  # reset(self)：重置计时器，将累积的总时间重置为0。
        self.acc = 0


class checkpoint():  # 用于保存和加载模型训练过程中的中间结果和日志信息
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)
        else:
            self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')
        self.n_processes = 8

    def get_path(self, *subdir):  # 根据子目录名获取完整的路径。
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        # 保存模型、损失、PSNR日志等信息。
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        # trainer.loss.plot_loss(self.dir, epoch)
        ####################################
        # self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def add_log(self, log):  # 将日志信息添加到日志列表中。
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    # def plot_psnr(self, epoch):
    #     # 绘制PSNR曲线图，其中横轴表示epoch，纵轴表示PSNR值。对于每个测试数据集和缩放比例，都会绘制一条曲线。
    #     axis = np.linspace(1, epoch, epoch)
    #     for idx_data, d in enumerate(self.args.data_test):
    #         label = 'SR on {}'.format(d)
    #         fig = plt.figure()
    #         plt.title(label)
    #         for idx_scale, scale in enumerate(self.args.scale):
    #             plt.plot(
    #                 axis,
    #                 self.log[:, idx_data, idx_scale].numpy(),
    #                 label='Scale {}'.format(scale)
    #             )
    #         plt.legend()
    #         plt.xlabel('Epochs')
    #         plt.ylabel('PSNR')
    #         plt.grid(True)
    #         plt.savefig(self.get_path('test_{}.pdf'.format(d)))
    #         plt.close(fig)

    def begin_background(self):  # 启动后台处理线程，用于异步保存结果图像。
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())
        
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()

    def end_background(self):  # end_background(self)：结束后台处理线程。
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list, scale):
        # 保存结果图像。根据数据集、文件名和缩放比例生成保存路径，并将结果图像保存为PNG格式。
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )

            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))


def quantize(img, rgb_range):
    # 将图像像素值量化到指定的范围内。根据rgb_range参数计算像素范围，然后将图像像素值乘以像素范围并取整，再将结果限制在0到255之间。
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    # 计算图像的PSNR值。首先计算SR图像与HR图像的差异，然后根据数据集和缩放比例进行处理。
    # 如果是基准数据集，则进行修剪操作；否则，进行缩放操作。然后计算
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)


def calc_ssim(img1, img2, scale=2, rgb_range = 255, dataset=None):
    # 计算两幅图像的SSIM值。首先根据数据集和缩放比例确定边界大小，然后将图像像素值转换为[0, 255]范围内的浮点数。
    # 接着将图像转换为numpy数组，并计算亮度通道。最后使用ssim函数计算SSIM值。
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if dataset and dataset.dataset.benchmark:
        border = math.ceil(scale)
    else:
        border = math.ceil(scale) + 6
    if rgb_range != 255:
        img1 = img1 * 255. / rgb_range
        img2 = img2 * 255. / rgb_range
    img1 = img1.data.squeeze().float().clamp(0, 255).round().cpu().numpy()
    img1 = np.transpose(img1, (1, 2, 0))
    img2 = img2.data.squeeze().cpu().numpy()
    img2 = np.transpose(img2, (1, 2, 0))

    img1_y = np.dot(img1, [65.738, 129.057, 25.064]) / 255.0 + 16.0
    img2_y = np.dot(img2, [65.738, 129.057, 25.064]) / 255.0 + 16.0
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1_y = img1_y[border:h - border, border:w - border]
    img2_y = img2_y[border:h - border, border:w - border]

    if img1_y.ndim == 2:
        return ssim(img1_y, img2_y)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def ssim(img1, img2):
    # 计算两幅图像的SSIM值。首先定义常量C1和C2，然后将图像转换为浮点数类型。
    # 接着使用高斯滤波器生成窗口，计算图像的均值、方差和协方差。最后根据公式计算SSIM值。
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def make_optimizer(args, target):
    # 用于创建优化器（optimizer）和学习率调度器（scheduler）
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    # 获取需要更新的参数，即具有requires_grad属性为True的参数。
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}
    # 设置优化器的初始学习率和权重衰减参数。
    if args.optimizer == 'SGD':  # 如果优化器类型为SGD，则选择optim.SGD作为优化器类，并设置动量参数。
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':  # 如果优化器类型为Adam，则选择optim.Adam作为优化器类，并设置动量参数和epsilon参数。
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':  # 如果优化器类型为RMSprop，则选择optim.RMSprop作为优化器类，并设置epsilon参数。
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):  # 这是一个自定义的优化器类，继承自optimizer_class。
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            # _register_scheduler方法用于注册一个调度器（scheduler），该调度器将在训练过程中根据指定的参数来调整学习率。
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):  # save方法用于保存优化器的状态字典到指定的路径。
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):  # load方法用于加载优化器的状态字典，并根据指定的轮数来更新调度器的状态。
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):  # get_dir方法用于生成保存优化器状态字典的路径。
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):  # schedule方法用于执行调度器的步骤，即更新学习率。
            self.scheduler.step()

        def get_lr(self):  # get_lr方法用于获取当前学习率。
            # return self.scheduler.get_lr()[0]
            return self.scheduler.get_last_lr()[0]

        def get_last_epoch(self):  # get_last_epoch方法用于获取调度器的最后一个轮数。
            return self.scheduler.last_epoch

    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

