import torch.nn as nn
import torch.nn.functional as F
import pdb

class model2(nn.Module):
    def __init__(self):
        super(model2, self).__init__()

        #input shape = [128, 3, 32, 32]

        #C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead) 

        self.convblock1 = nn.Sequential(
                                        #Conv1
                                        nn.Conv2d(in_channels=3, out_channels=32,kernel_size=3, padding= 1, bias = False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32),
                                        nn.Dropout2d(.01),
                                        #Output of above conv = [B, 32, 32, 32], RF = 3
                                        

                                        #Dilated Convolution
                                        nn.Conv2d(in_channels=32, out_channels=32,kernel_size=3, padding= 2, dilation= 2, bias= False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32),
                                        nn.Dropout2d(.01),
                                        # Receptive field post dialtion: RFin + (k-1)*dilation factor + 1 = 3+ (2)*2 +1 = 8

                                        #Depthwise
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32, bias=False),
                                        #Pointwise
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, bias = False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32),
                                        nn.Dropout2d(.01),
                                        #Output of above conv = [B, 64, 32, 32], RF = 5

                                        #Depthwise
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32, bias=False),
                                        #Pointwise
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, bias = False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32),
                                        nn.Dropout2d(.01),

                                        #Depthwise
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32, bias=False),
                                        #Pointwise
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, bias = False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32),
                                        nn.Dropout2d(.01),

                                        #Dilated Convolution
                                        nn.Conv2d(in_channels=32, out_channels=32,kernel_size=3, padding= 2, dilation= 2, bias= False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32),
                                        nn.Dropout2d(.01),

                                        )

        self.convblock2 = nn.Sequential(
                                        #Conv4
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32),
                                        nn.Dropout2d(.01),
                                        #Output of above conv = [B, 128, 15, 15], RF = 11

                                        #Dilated Convolution
                                        nn.Conv2d(in_channels=32, out_channels=32,kernel_size=3, padding= 2, dilation= 2, bias= False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32),
                                        nn.Dropout2d(.01),

                                        #Conv5
                                        #Depthwise
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32, bias= False),
                                        #Pointwise
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, bias= False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32),
                                        nn.Dropout2d(.01),
                                        #Output of above conv = [B, 128, 15, 15], RF = 15

                                        #Depthwise
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32, bias= False),
                                        #Pointwise
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, bias= False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32),
                                        nn.Dropout2d(.01),

                                        #Depthwise
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32, bias= False),
                                        #Pointwise
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, bias= False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32),
                                        nn.Dropout2d(.01),

                                        #Dilated Convolution
                                        nn.Conv2d(in_channels=32, out_channels=32,kernel_size=3, padding= 2, dilation= 2, bias= False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32),
                                        nn.Dropout2d(.01),
                                        #Output of above conv = [B, 128, 8, 8], RF = 19 , Jin = 8
                                        )

        self.convblock3 = nn.Sequential(
                                        #Conv7
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias = False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32),
                                        nn.Dropout2d(.01),
                                        #Output of above conv = [B, 256, 8, 8], RF = 27
                                        
                                        #Dilated Convolution
                                        nn.Conv2d(in_channels=32, out_channels=32,kernel_size=3, padding= 2, dilation= 2, bias= False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32),
                                        nn.Dropout2d(.01),

                                        #Conv4
                                        #Depthwise
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32, bias = False),
                                        #Pointwise
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, bias = False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32),
                                        nn.Dropout2d(.01),
                                        #Output of above conv = [B, 128, 8, 8], RF = 35

                                        #Depthwise
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32, bias = False),
                                        #Pointwise
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, bias = False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32),
                                        nn.Dropout2d(.01),

                                        #Depthwise
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32, bias = False),
                                        #Pointwise
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, bias = False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32),
                                        nn.Dropout2d(.01),

                                        #Conv4
                                        #Dilated Convolution
                                        nn.Conv2d(in_channels=32, out_channels=16,kernel_size=3, padding= 2, dilation= 2, bias= False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(16),
                                        nn.Dropout2d(.01),
                                        #Output of above conv = [B, 64, 5, 5], RF = 43

                                        )

        self.globalavgpool = nn.AdaptiveAvgPool2d((1,1))
        #Convolution output shape = [B,64, 1, 1], RF >> 44

        self.convblock4 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels= 10, kernel_size=1, bias= False))
        

    def forward(self, x):
        
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.globalavgpool(x)
        x = self.convblock4(x)

        x = x.view(-1, 10)

        x = F.log_softmax(x, dim = -1)

        return x
    
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        #input shape = [128, 3, 32, 32]

        #C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead) 

        self.convblock1 = nn.Sequential(
                                        #Conv1
                                        nn.Conv2d(in_channels=3, out_channels=32,kernel_size=3, padding= 1, bias = False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32),
                                        nn.Dropout2d(.01),
                                        #Output of above conv = [B, 32, 32, 32], RF = 3

                                        #Dilated Convolution
                                        nn.Conv2d(in_channels=32, out_channels=32,kernel_size=3, padding= 2, dilation= 2, bias= False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32),
                                        nn.Dropout2d(.01),
                                        # Receptive field post dialtion: RFin + (k-1)*dilation factor + 1 = 3+ (2)*2 +1 = 8

                                        #Depthwise
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32, bias=False),
                                        #Pointwise
                                        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, bias = False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(64),
                                        nn.Dropout2d(.01),
                                        #Output of above conv = [B, 64, 32, 32], RF = 5

                                        #Conv3
                                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, bias = False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(64),
                                        nn.Dropout2d(.01)
                                        #Output of above conv = [B, 64, 15, 15], RF = 7 , Jin = 2

                                        )

        self.convblock2 = nn.Sequential(
                                        #Conv4
                                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(64),
                                        nn.Dropout2d(.01),
                                        #Output of above conv = [B, 128, 15, 15], RF = 11

                                        #Conv5
                                        #Depthwise
                                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64, bias= False),
                                        #Pointwise
                                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, bias= False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(64),
                                        nn.Dropout2d(.01),
                                        #Output of above conv = [B, 128, 15, 15], RF = 15

                                        #Conv6
                                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias = False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(64),
                                        nn.Dropout2d(.01)
                                        #Output of above conv = [B, 128, 8, 8], RF = 19 , Jin = 8
                                        )

        self.convblock3 = nn.Sequential(
                                        #Conv7
                                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias = False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(64),
                                        nn.Dropout2d(.01),
                                        #Output of above conv = [B, 256, 8, 8], RF = 27
                                        
                                        #Conv4
                                        #Depthwise
                                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64, bias = False),
                                        #Pointwise
                                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, bias = False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(64),
                                        nn.Dropout2d(.01),
                                        #Output of above conv = [B, 128, 8, 8], RF = 35

                                        #Conv4
                                        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, bias= False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32),
                                        nn.Dropout2d(.01),
                                        #Output of above conv = [B, 64, 5, 5], RF = 43

                                        )

        self.globalavgpool = nn.AdaptiveAvgPool2d((1,1))
        #Convolution output shape = [B,64, 1, 1], RF >> 44

        self.convblock4 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels= 10, kernel_size=1, bias= False))
        

    def forward(self, x):
        
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.globalavgpool(x)
        x = self.convblock4(x)

        x = x.view(-1, 10)

        x = F.log_softmax(x, dim = -1)

        return x
    
class model3(nn.Module):
    def __init__(self):
        super(model3, self).__init__()

        #input shape = [128, 3, 32, 32]

        #C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead) 

        self.convblock1 = nn.Sequential(
                                        #Conv1
                                        nn.Conv2d(in_channels=3, out_channels=32,kernel_size=3, padding= 1, bias = False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32),
                                        nn.Dropout2d(.02),
                                        #Output of above conv = [B, 32, 32, 32], RF = 3
                                        
                                        
                                        #Dilated Convolution
                                        nn.Conv2d(in_channels=32, out_channels=32,kernel_size=3, padding= 2, dilation= 2, bias= False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32),
                                        nn.Dropout2d(.02),
                                        # Receptive field post dialtion: RFin + (k-1)*dilation factor + 1 = 3+ (2)*2 +1 = 8
                                        )
        
                                        
        self.convblock2 = nn.Sequential(                                
                                        #Depthwise
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32, bias=False),
                                        #Pointwise
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, bias = False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32),
                                        nn.Dropout2d(.02),
                                        #Output of above conv = [B, 64, 32, 32], RF = 5

                                        #Conv3
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, bias = False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32),
                                        nn.Dropout2d(.02)
                                        #Output of above conv = [B, 64, 15, 15], RF = 7 , Jin = 2
                                        )                        

        self.convblock3 = nn.Sequential(                                
                                        #Depthwise
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32, bias=False),
                                        #Pointwise
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, bias = False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32),
                                        nn.Dropout2d(.02),
                                        #Output of above conv = [B, 64, 32, 32], RF = 5

                                       #Depthwise
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32, bias=False),
                                        #Pointwise
                                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, bias = False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32),
                                        nn.Dropout2d(.02),
                                        #Output of above conv = [B, 64, 32, 32], RF = 5
                                        )                                 

        self.convblock4 = nn.Sequential(
                                        #Conv4
                                        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(64),
                                        nn.Dropout2d(.02),
                                        #Output of above conv = [B, 128, 15, 15], RF = 11
                                        )
        self.convblock5 = nn.Sequential(
                                        #Conv5
                                        #Depthwise
                                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64, bias= False),
                                        #Pointwise
                                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, bias= False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(64),
                                        nn.Dropout2d(.02),
                                        #Output of above conv = [B, 128, 15, 15], RF = 15
                                        )
        self.convblock7 = nn.Sequential(
                                        #Conv6
                                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias = False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(64),
                                        nn.Dropout2d(.02)
                                        #Output of above conv = [B, 128, 8, 8], RF = 19 , Jin = 8
                                        
                                        )
        self.convblock8 = nn.Sequential(
                                        #Conv7
                                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias = False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(64),
                                        nn.Dropout2d(.02),
                                        #Output of above conv = [B, 256, 8, 8], RF = 27
                                        
                                        #Conv4
                                        #Depthwise
                                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64, bias = False),
                                        #Pointwise
                                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, bias = False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(64),
                                        nn.Dropout2d(.02),
                                        #Output of above conv = [B, 128, 8, 8], RF = 35

                                        #Conv4
                                        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, bias= False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32),
                                        nn.Dropout2d(.02),
                                        #Output of above conv = [B, 64, 5, 5], RF = 43

                                        )

        self.globalavgpool = nn.AdaptiveAvgPool2d((1,1))
        #Convolution output shape = [B,64, 1, 1], RF >> 44

        self.convblock9 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels= 10, kernel_size=1, bias= False))
        

    def forward(self, x):
        
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = x + self.convblock3(x)
        x = self.convblock4(x)
        x = x+self.convblock5(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.globalavgpool(x)
        x = self.convblock9(x)

        x = x.view(-1, 10)

        x = F.log_softmax(x, dim = -1)

        return x
