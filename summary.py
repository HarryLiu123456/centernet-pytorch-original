#--------------------------------------------#
#   该部分代码用于看网络参数
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary

from nets.centernet import CenterNet_Resnet50

if __name__ == "__main__":
    input_shape     = [512, 512]
    num_classes     = 20
    
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CenterNet_Resnet50().to(device) # 将模型加载到GPU上
    summary(model, (3, input_shape[0], input_shape[1])) # 输出模型的结构信息

    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device) # 创建一个随机输入
    flops, params   = profile(model.to(device), (dummy_input, ), verbose=False) # 输出模型的参数量和计算量
    #--------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
    print('--------------------------------------------------')
    print(model)
    # summary(model, input_size=(3, 256, 256), batch_size=-1, device='cuda')
