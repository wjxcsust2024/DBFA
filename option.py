import argparse
import template

parser = argparse.ArgumentParser(description='RFDN')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# 硬件规格
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# 数据规范/root/autodl-tmp/src/DIV2K
parser.add_argument('--dir_data', type=str, default='D:/DIV2K',  #
                    help='daptaset directory')
# parser.add_argument('--dir_data', type=str, default='D:/DIV2K',
#                     help='daptaset directory')
# parser.add_argument('--dir_demo', type=str, default='../test',
#                     help='demo image directory')
parser.add_argument('--dir_demo', type=str, default='D:/DIV2K',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='Set14',    #
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-800/801-900',
                    help='train/test data range')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--scale', type=str, default='4',  #
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=256,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=1,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
# --dir_data表示数据集的目录，默认值为'D:/DIV2K'。
# --dir_demo表示演示图像的目录，默认值为'../test'。
# --data_train表示训练数据集的名称，默认值为'DIV2K'。
# --data_test表示测试数据集的名称，默认值为'Set5'。
# --data_range表示训练/测试数据的范围，默认值为'1-800/801-900'。
# --ext表示数据集文件的扩展名，默认值为'sep'。
# --scale表示超分辨率的缩放比例，默认值为'4'。
# --patch_size表示输出补丁的尺寸，默认值为256。
# --rgb_range表示RGB图像的最大值，默认值为1。
# --n_colors表示要使用的颜色通道数，默认值为3。
# --chop表示是否启用内存高效的前向传播，是一个布尔值。
# --no_augment表示是否禁用数据增强，是一个布尔值。
# Model specifications
parser.add_argument('--model', default='HNCT',
                    help='model name')

parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=6,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Option for Residual dense network (RDN)
parser.add_argument('--G0', type=int, default=32,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B',
                    help='parameters config of RDN. (Use in RDN)')

# Option for Residual channel attention network (RCAN)
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=1,
                    help='number of feature maps reduction')

# NLSN
parser.add_argument('--n_l1_block', type=int, default=4,
                    help='level-1 block number')
parser.add_argument('--n_l2_block', type=int, default=4,
                    help='level-2 block number')
parser.add_argument('--n_l3_block', type=int, default=4,
                    help='level-3 block number')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
# parser.add_argument('--epochs', type=int, default=1200,
#                     help='number of epochs to train')
parser.add_argument('--epochs', type=int, default=2,
                    help='number of epochs to train')
#轮数
################
parser.add_argument('--batch_size', type=int, default=1,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Optimization specifications
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
# parser.add_argument('--decay', type=str, default='200-400-600-800-1000',
#                     help='learning rate decay type')
parser.add_argument('--decay', type=str, default='50-100-150-200-250',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e8',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')

args = parser.parse_args()
template.set_template(args)

args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

