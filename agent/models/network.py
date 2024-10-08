import torch as T
from torch.nn import init
from torchvision.models import resnet18, resnet34


def create_state(state, device='cuda'):
    image_state = T.tensor(state[0], dtype=T.float).to(device)
    env_state = T.tensor(state[1], dtype=T.float).to(device)
    return image_state, env_state

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += T.mean(T.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func) 

def save_model(model, epoch):
    print(f"Save checkpoint at epoch {epoch}")
    T.save(model.state_dict(), f'pytorch_models/model{epoch}.pth')

def load_model(model, epoch):
    checkpoint_path = f'pytorch_models/model{epoch}.pth'
    model.load_state_dict(T.load(checkpoint_path))
    return model

def get_base_cnn():
    cnn = resnet34(pretrained=True)
    cnn = T.nn.Sequential(*(list(cnn.children())[:-3]), T.nn.MaxPool2d(14))
    for param in cnn.parameters():
        param.requires_grad = False
    return cnn