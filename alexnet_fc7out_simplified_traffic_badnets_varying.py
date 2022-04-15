
import torch
import torch.nn as nn
import os

__all__ = ['AlexNet', 'alexnet']



model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# from https://github.com/BorealisAI/advertorch/blob/master/advertorch/utils.py
class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)

class AlexNet(nn.Module):

    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
#        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
           # nn.Dropout(),
           # nn.Linear(128 * 6 * 6, 512),
            nn.Linear(128*4*4, 512),
            nn.ReLU(inplace=True),
            #nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        #print(x.shape)
        x = self.features(x)
        #print(x.shape)
        #feat = x.view(x.size(0), 256 * 6 * 6)           # conv5 features
        #x = self.avgpool(x)
        x = x.flatten(1)
        #print(x)
    
        for i in range(2):
            x = self.classifier[i](x)
        feat = x                                        # fc7 features
        x = self.classifier[2](x)

#        x = self.classifier(x)
        return x, feat#, feat1

def alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    #normalize = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #model = nn.Sequential(normalize, model_ft)
    if pretrained:
        #state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                             # progress=progress)
        #experimentID = "experiment002000"
        experimentID = "experiment0401"
        clean_data_root	= "Image"
        poison_root	= "poison_data"
        gpu         = int("0")
        epochs      = int("99")
        patch_size  = int("5")
        eps         = int("32")
        rand_loc    = "False"
        trigger_id  = int("14")
        num_poison  = int("800")


        checkpointDir_clean = "finetuned_models/" + experimentID + "/rand_loc_" +  str(rand_loc) + "/eps_" + str(eps) + \
				"/patch_size_" + str(patch_size) + "/num_poison_" + str(num_poison) + "/trigger_" + str(trigger_id) + "/badnets1"  
                
        filename=os.path.join(checkpointDir_clean, "poisoned_model_model.pt")

        #print(filename)
        checkpoint = torch.load(filename)                                     
        #state_dict = torch.load(checkpoint['state_dict'])
        #model.load_state_dict(state_dict)
        #model_ft.load_state_dict(checkpoint)
        model = checkpoint
        
    return model