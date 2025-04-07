import torch
import numpy as np
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 首先，设置CUDA_DEVICE_ORDER为"PCI_BUS_ID"，这是指定设备顺序的一种方式
os.environ['CUDA_VISIBLE_DEVICES']="0"  # 这是指定可见的CUDA设备的索引，这里设置为0表示只使用第一个设备。
os.environ['CUDA_CACHE_PATH']='~/.cudacache'  # 这是指定CUDA缓存路径的，可以自定义设置。
# 用于设置CUDA环境变量和进行一些初始化操作。

torch.manual_seed(args.seed)  # 设置随机种子为args.seed，这是为了在训练过程中产生可重复的结果
torch.cuda.empty_cache()  # 清空CUDA缓存，以释放显存空间。
checkpoint = utility.checkpoint(args)  # 用于处理检查点相关的操作
# python /home/linhanjiang/projects/AIM/EDSR/src/main.py --model rfdn --data_test Set5+Set14+
# B100+Urban100+DIV2K  --data_range 801-900 --scale 4 --save rfdn_x4 --pre_train /home/linhanjiang/projects/AIM/EDSR/experiment/test/model/model_best.pt --rgb_range 1 --test_only --save_results

def main():
    global model
    # 首先，如果args.data_test是一个列表，且列表中只有一个元素"video"，则说明是要进行视频测试，于是引入VideoTester类，并创建该类的实例t。
    if args.data_test == ['video']:
        from videotester import VideoTester
        # 在创建实例时，会传入args、model、checkpoint三个参数，其中model是通过model.Model创建的。然后，调用t.test()方法进行测试。
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()

    else:  # 如果args.data_test不是"video"，则说明是要进行训练和测试。
        if checkpoint.ok:  # 首先，检查是否有可用的检查点，如果有，则创建data.Data类的实例loader，用于加载数据。
            loader = data.Data(args)
            # 然后，创建model.Model类的实例_model，并创建loss.Loss类的实例_loss
            # 如果args.test_only为True，则_loss为None。
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            # 接着，创建Trainer类的实例t，并传入args、loader、_model、_loss、checkpoint五个参数。
            t = Trainer(args, loader, _model, _loss, checkpoint)
            # 在训练和测试的循环中，先调用t.train()方法进行训练，然后调用t.test()方法进行测试，直到满足终止条件。
            while not t.terminate():
                # t.train()
                t.test()
            # 最后，调用checkpoint.done()方法，表示训练结束。
            checkpoint.done()

if __name__ == '__main__':
    main()
