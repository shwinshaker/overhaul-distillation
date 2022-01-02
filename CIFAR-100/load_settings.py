
import torch
import os


from models import model_dict

# import models.WideResNet as WRN
# import models.PyramidNet as PYN
# import models.ResNet as RN


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]

def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model

def load_paper_settings(args, n_cls):

    teacher = load_teacher(args.teacher_path, n_cls)
    student = model_dict[args.model_student](num_classes=n_cls)

    # WRN_path = os.path.join(args.data_path, 'WRN28-4_21.09.pt')
    # Pyramid_path = os.path.join(args.data_path, 'pyramid200_mixup_15.6.tar')

    # if args.paper_setting == 'a':
    #     teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
    #     state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
    #     teacher.load_state_dict(state)
    #     student = WRN.WideResNet(depth=16, widen_factor=4, num_classes=100)

    # elif args.paper_setting == 'b':
    #     teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
    #     state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
    #     teacher.load_state_dict(state)
    #     student = WRN.WideResNet(depth=28, widen_factor=2, num_classes=100)

    # elif args.paper_setting == 'c':
    #     teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
    #     state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
    #     teacher.load_state_dict(state)
    #     student = WRN.WideResNet(depth=16, widen_factor=2, num_classes=100)

    # elif args.paper_setting == 'd':
    #     teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
    #     state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
    #     teacher.load_state_dict(state)
    #     student = RN.ResNet(depth=56, num_classes=100)

    # elif args.paper_setting == 'e':
    #     teacher = PYN.PyramidNet(depth=200, alpha=240, num_classes=100, bottleneck=True)
    #     state = torch.load(Pyramid_path, map_location={'cuda:0': 'cpu'})['state_dict']
    #     from collections import OrderedDict
    #     new_state = OrderedDict()
    #     for k, v in state.items():
    #         name = k[7:]  # remove 'module.' of dataparallel
    #         new_state[name] = v
    #     teacher.load_state_dict(new_state)
    #     student = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)

    # elif args.paper_setting == 'f':
    #     teacher = PYN.PyramidNet(depth=200, alpha=240, num_classes=100, bottleneck=True)
    #     state = torch.load(Pyramid_path, map_location={'cuda:0': 'cpu'})['state_dict']
    #     from collections import OrderedDict
    #     new_state = OrderedDict()
    #     for k, v in state.items():
    #         name = k[7:]  # remove 'module.' of dataparallel
    #         new_state[name] = v
    #     teacher.load_state_dict(new_state)
    #     student = PYN.PyramidNet(depth=110, alpha=84, num_classes=100, bottleneck=False)

    # else:
    #     print('Undefined setting name !!!')

    return teacher, student # , args