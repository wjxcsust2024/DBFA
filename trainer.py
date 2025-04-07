import os
import math
from decimal import Decimal
import utility
import torch
import torch.nn as nn
import torch.nn.utils as utils
from tqdm import tqdm
from pandas import DataFrame
import numpy as np
import pdb
import cv2


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)



        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())/2
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)/2
        print(f'{total_trainable_params:,} training parameters.')

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch()
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()  # 执行一次优化器的优化步骤。

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:  # 如果达到了打印间隔，则输出当前的损失值和时间。
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))  # 结束记录损失值，并计算平均损失值。
        self.error_last = self.loss.log[-1, -1]  # 将最后一个损失值保存为最终的损失值。
        self.optimizer.schedule()  # 根据当前的轮数调整学习率。

    def test(self):
        torch.set_grad_enabled(False)  # 关闭了梯度计算

        epoch = self.optimizer.get_last_epoch()  # 获取最后一个轮数epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()
        # 记录评估的日志，并初始化一些变量用于保存评估结果。
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        # 然后，对测试数据集进行迭代，在每个迭代中，首先将输入数据准备好，然后使用模型生成超分辨率图像。
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                eval_acc = 0
                eval_acc_ssim = 0
                PSNR_values = []
                SSIM_values = []
                image_names = []
                for lr, hr, filename in tqdm(d, ncols=80):
                    image_names.append(filename[0])
                    lr, hr = self.prepare(lr, hr)

                    sr = self.model(lr, idx_scale)

                    if filename[0] in ['butterfly', 'baboon', '253027', 'img062','ARMS','0828']:


                        output = sr
                        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                        if output.ndim == 3:
                            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
                        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
                        cv2.imwrite(filename[0]+'.png', output)


                    sr = utility.quantize(sr, self.args.rgb_range)
                    save_list = [sr]
                    psnr = utility.calc_psnr(sr, hr, scale, self.args.rgb_range, dataset=d)
                    eval_acc += psnr
                    PSNR_values.append(psnr)
                    ssim = utility.calc_ssim(sr, hr, scale, self.args.rgb_range, dataset=d)
                    eval_acc_ssim += ssim
                    SSIM_values.append(ssim)
                    ########################################################################
                    self.ckp.log[-1, idx_data, idx_scale] += psnr
                    # self.ckp.log[-1, idx_data, idx_scale] += ssim
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                image_names.append('average')

                PSNR_values.append(eval_acc / len(d))
                SSIM_values.append(eval_acc_ssim / len(d))
                if self.args.load == '.' or self.args.load == '':
                    xlsx_file_name = '../experiment/' + self.args.save + '/results-' + d.dataset.name + '/' + d.dataset.name + '.xlsx'
                else:
                    xlsx_file_name = '../experiment/' + self.args.load + '/results-' + d.dataset.name + '/' + d.dataset.name + '.xlsx'
                image_names = np.array(image_names,dtype=object)
                PSNR_values = np.array(PSNR_values,dtype=object)
                SSIM_values = np.array(SSIM_values,dtype=object)
                PSNR_values = [round(i, 2) for i in PSNR_values]
                SSIM_values = [round(i, 4) for i in SSIM_values]
                data = {
                    'image': image_names,
                    'psnr': PSNR_values,
                    'ssim': SSIM_values
                }
                df = DataFrame(data)
                df.to_excel(xlsx_file_name)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )
                ##############################
                # self.ckp.write_log(
                #     '[{} x{}]\tSSIM: {:.4f} (Best: {:.4f} @epoch {})'.format(
                #         d.dataset.name,
                #         scale,
                #         self.ckp.log[-1, idx_data, idx_scale],
                #         best[0][idx_data, idx_scale],
                #         best[1][idx_data, idx_scale] + 1
                #     )
                # )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        # prepare函数接受一系列参数，并将它们准备好以在训练过程中使用。它首先根据设备类型（CPU或CUDA）创建一个device对象。
        # 然后，它定义了一个内部函数_prepare，用于将张量转移到指定的设备并进行精度调整（如果指定的话）。
        # 最后，它将所有参数都传递给_prepare函数，并返回转换后的参数列表。
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        # terminate函数用于判断是否应该终止训练过程。如果args.test_only为True，则调用test函数进行测试并返回True。
        # 否则，获取最后一个轮数，并判断是否大于等于指定的训练轮数args.epochs，如果是则返回True，表示应该终止训练过程。
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch()
            return epoch >= self.args.epochs

