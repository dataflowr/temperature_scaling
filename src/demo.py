# import fire
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import os
import sys

import torch
import torchvision as tv
from torch.utils.data.sampler import SubsetRandomSampler
from models import DenseNet
from temperature_scaling import ModelWithTemperature
from platt_scaling import ModelWithPlatt
from isotonic_regression import ModelWithIsotonic

def demo(data, save, scaling_type='Temperature', depth=40, growth_rate=12, batch_size=256):
    """
    Applies different scaling methods to a trained model.

    Takes a pretrained DenseNet-CIFAR100 model, and a validation set
    (parameterized by indices on train set).
    Applies scaling, and saves the network parameters.

    NB: the "save" parameter references a DIRECTORY, not a file.
    In that directory, there should be two files:
    - model.pth (model state dict)
    - valid_indices.pth (a list of indices corresponding to the validation set).

    data (str) - path to directory where data should be loaded from/downloaded
    save (str) - directory with necessary files (see above)
    scaling_type (str) - calibration model to be applied. Can either be 'Platt', 'Isotonic' or 'Temperature'. By default, temperature scaling is performed.
    """
    # Load model state dict
    model_filename = os.path.join(save, 'model.pth')
    if not os.path.exists(model_filename):
        raise RuntimeError('Cannot find file %s to load' % model_filename)
    state_dict = torch.load(model_filename)

    # Load validation indices
    valid_indices_filename = os.path.join(save, 'valid_indices.pth')
    if not os.path.exists(valid_indices_filename):
        raise RuntimeError('Cannot find file %s to load' % valid_indices_filename)
    valid_indices = torch.load(valid_indices_filename)

    # Regenerate validation set loader
    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]
    test_transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])
    valid_set = tv.datasets.CIFAR100(data, train=True, transform=test_transforms, download=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, pin_memory=True, batch_size=batch_size,
                                               sampler=SubsetRandomSampler(valid_indices))

    # Load original model
    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(depth - 4) // 6 for _ in range(3)]
    orig_model = DenseNet(
        growth_rate=growth_rate,
        block_config=block_config,
        num_classes=100
    ).cuda()
    orig_model.load_state_dict(state_dict)

    # Now we're going to wrap the model with a decorator that adds temperature scaling
    if scaling_type == 'Platt':
        print('Platt scaling')
        model = ModelWithPlatt(orig_model)

    elif scaling_type == 'Isotonic':
        print('Isotonic regression 1-vs-all')
        model = ModelWithIsotonic(orig_model)

    else:
        print('Temperature')
        model = ModelWithTemperature(orig_model)


    # Tune the model temperature, and save the results
    model = model.eval()
    model.set_temperature(valid_loader)
    model_filename = os.path.join(save, 'model_with_{}.pth'.format(scaling_type))
    if scaling_type == "Isotonic":
        torch.save(model, model_filename)
    else:
        torch.save(model.state_dict(), model_filename)
    print('Scaled model sved to %s' % model_filename)
    print('Done!')


if __name__ == '__main__':
    """
    Applies temperature scaling to a trained model.

    Takes a pretrained DenseNet-CIFAR100 model, and a validation set
    (parameterized by indices on train set).
    Applies temperature scaling, and saves a temperature scaled version.

    NB: the "save" parameter references a DIRECTORY, not a file.
    In that directory, there should be two files:
    - model.pth (model state dict)
    - valid_indices.pth (a list of indices corresponding to the validation set).

    --data (str) - path to directory where data should be loaded from/downloaded
    --save (str) - directory with necessary files (see above)
    """
    path = 'D:\\sbossou\\OneDrive - ERAAM\\projects\\misc\\2020_06_sergio_internship\\py' + '\\GitHub\\Project-calibration-temperature_scaling\\'
    # fire.Fire(demo)
    data = path + '\\data\\'
    save = path + '\\model'
    demo(data, save,'Isotonic', depth=40, growth_rate=12, batch_size=256)