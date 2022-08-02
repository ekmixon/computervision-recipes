from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from .networks.dlav0 import get_pose_net as get_dlav0
from .networks.pose_dla_dcn import get_pose_net as get_dla_dcn
from .networks.pose_hrnet import get_pose_net as get_pose_net_hrnet
from .networks.resnet_dcn import get_pose_net as get_pose_net_dcn
from .networks.resnet_fpn_dcn import get_pose_net as get_pose_net_fpn_dcn

_model_factory = {
  'dlav0': get_dlav0, # default DLAup
  'dla': get_dla_dcn,
  'resdcn': get_pose_net_dcn,
  'resfpndcn': get_pose_net_fpn_dcn,
  'hrnet': get_pose_net_hrnet
}

def create_model(arch, heads, head_conv):
  num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
  arch = arch[:arch.find('_')] if '_' in arch else arch
  get_model = _model_factory[arch]
  return get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)

def load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print(f"loaded {model_path}, epoch {checkpoint['epoch']}")
  state_dict_ = checkpoint['state_dict']
  state_dict = {}

  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
  for k, v in state_dict.items():
    if k in model_state_dict:
      if v.shape != model_state_dict[k].shape:
        print(
            f'Skip loading parameter {k}, required shape{model_state_dict[k].shape}, loaded shape{state_dict[k].shape}. {msg}'
        )
        state_dict[k] = model_state_dict[k]
    else:
      print(f'Drop parameter {k}.' + msg)
  for k in model_state_dict:
    if k not in state_dict:
      print(f'No param {k}.' + msg)
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  return (model, optimizer, start_epoch) if optimizer is not None else model

def save_model(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if optimizer is not None:
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)

