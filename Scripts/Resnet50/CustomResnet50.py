import torch
import torch.nn as nn
import torch.functional as f

class BottleNeck(nn.Module):
    expansion=4
    def __init__(self, in_channels, out_channels, i_downsample=None, strides=1):
        super().__init__()
        #1st Conv layer 1x1
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(out_channels)

        #2nd Conv Layer 3X3
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=strides, padding=1, bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)

        #3rd Con Layer 1X1
        self.conv_3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0,bias=False)
        self.batch_norm_3 = nn.BatchNorm2d(out_channels*self.expansion)

        #Initialize downsample, stride and relu
        self.downsample = i_downsample
        self.stride = strides
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.relu(x)

        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.relu(x)

        x = self.conv_3(x)
        x = self.batch_norm_3(x)

        #Skip connection
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        x += identity
        x = self.relu(x)
        return x

#For Resnet 18/34
class Block(nn.Module):
    expansion=1
    def __init__(self, in_channels, out_channels, i_downsample=None, strides=1):
        super().__init__()

        #2nd Conv Layer 3X3
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=strides, padding=1,bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(out_channels)

        #3rd Con Layer 3X3
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)

        #Initialize downsample, stride and relu
        self.downsample = i_downsample
        self.stride = strides
        self.relu = nn.ReLU()
    
    def forward(self,x):
        identity = x

        x = self.relu(self.batch_norm_1(self.conv_1(x)))
        x = self.batch_norm_2(self.conv_2(x))

         #Skip connection
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        x += identity
        x = self.relu(x)
        return x
        


class Resnet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super().__init__()
        self.in_channels = 64

        self.conv = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.batch_norm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #Make layers
        self.layer1 = self._make_layers(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layers(ResBlock, layer_list[1], planes=128, strides=2)
        self.layer3 = self._make_layers(ResBlock, layer_list[2], planes=256, strides=2)
        self.layer4 = self._make_layers(ResBlock, layer_list[3], planes=512, strides=2)

        self.average_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes)

    def _make_layers(self, ResBlock, blocks, planes, strides=1):
       downsample = None
       layers = []

       if strides!=1 or self.in_channels != planes*ResBlock.expansion:
           downsample = nn.Sequential(
               nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=strides),
               nn.BatchNorm2d(planes*ResBlock.expansion)
           )
       
       layers.append(ResBlock(self.in_channels, planes, downsample, strides))
       self.in_channels = planes*ResBlock.expansion

       for i in range(blocks-1):
           layers.append(ResBlock(self.in_channels, planes))
    
       return nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.average_pool(x)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

#Define Resnet 50
def Resnet50(num_classes, channels=1):
    model = Resnet(BottleNeck, [3,4,6,3], num_classes, num_channels=channels)
    model.architecture_name = "Resnet50"
    return model

def Resnet18(num_classes, channels=1):
    model = Resnet(Block, [2,2,2,2], num_classes,channels)
    model.architecture_name = "Resnet18"
    return model


