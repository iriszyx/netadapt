import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['celebanet', 'CelebaNet']


# model_urls = {
#     'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
# }


class CelebaNet(nn.Module):

    def __init__(self, multiplier=1.0):
        super(CelebaNet, self).__init__()
        conv_depth = int(32 * multiplier)
        self.features = nn.Sequential(
            nn.Conv2d(3, conv_depth, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(conv_depth, conv_depth * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(conv_depth * 2, conv_depth * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(conv_depth * 4, conv_depth * 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(conv_depth * 8, conv_depth * 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(8 * conv_depth * 6 * 6, int(512 * multiplier)),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(1024, 1024),
            # nn.ReLU(inplace=True),
            nn.Linear(int(512 * multiplier), 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def celebanet(pretrained=False, progress=True, num_classes=1000, multiplier=1.0):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = CelebaNet(multiplier=multiplier)
    if pretrained:
        print ('No pretrained model!')
        # state_dict = model_zoo.load_url(model_urls['celebanet'], progress=progress)
        # model.load_state_dict(state_dict)
    if num_classes != 1000:
        num_in_feature = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_in_feature, num_classes)
    return model