import argparse
import json
import os

from helper.util import check_path

def parse_option():

    parser = argparse.ArgumentParser(description='CIFAR-100 training')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--teacher_path', type=str, default='./experiments/models')
    parser.add_argument('--model_student', type=str, default='resnet8',
                            choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                    'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                    'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                    'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    # parser.add_argument('--paper_setting', default='a', type=str)
    # parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    # parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size (default: 256)')
    # parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')

    parser.add_argument('--epochs', default=240, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 256)')
    parser.add_argument('--lr', default=0.05, type=float, help='initial learning rate')
    parser.add_argument('--gamma', default=1.0, type=float, help='ce loss weight')
    parser.add_argument('--alpha', default=1.0, type=float, help='kd loss weight')
    parser.add_argument('--beta', default=1000, type=float, help='distill loss weight')
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    args = parser.parse_args()

    print('==> Make path..')
    teacher_name = os.path.abspath(args.teacher_path).split('/')[-2]
    exp_name = 'oh_student={}_teacher={}'.format(args.model_student, teacher_name)
    exp_name += '_gamma={:g}'.format(args.gamma)
    exp_name += '_alpha={:g}'.format(args.alpha)
    exp_name += '_beta={:g}'.format(args.beta)
    if args.lr != 0.05:
        exp_name += '_lr={:g}'.format(args.lr)
    args.exp_path = './experiments/student_model/{}'.format(exp_name)

    # set different learning rate from these 4 models
    if args.model_student in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        print('   Lr reset to 0.01. Needs to be test which one is better!')
        args.lr = 0.01

    check_path(args.exp_path)
    print('   path: %s' % args.exp_path)

    return args


if __name__ == '__main__':
    opt = parse_option()

    # Save setting to json
    with open('./tmp/config.tmp', 'wt') as f:
        json.dump(vars(opt), f, indent=4)