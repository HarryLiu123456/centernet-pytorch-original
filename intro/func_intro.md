1. nets
    1. centernet_training.py模块
        1. 一些训练时用到的函数
        1. 计算focal_loss
        1. 计算l1_loss
        1. 初始化权重
        1. 学习率调度器
        1. 灵活地调整学习率
    1. centernet.py网络总模块
        1. 主要是整个centernet的结构及功能
        1. centernet类
        1. 骨干冻结函数
        1. 骨干解冻函数
        1. 初始化权重函数
    1. resnet50.py网络模块
        1. 特征提取部分（resnet50）
        1. 特征解码部分（resnet50_Decoder）
        1. 特征处理部分（resnet50_Head）

1. utils
    1. callbacks.py
        1. 
    1. utils.py工具模块
        1. 将图像转换成RGB图像
        1. 对输入图像进行resize
        1. 获得类别列表和类别数量
        1. 获得训练率
        1. 设置所有东西的种子
        1. 设置Dataloader的种子
        1. 标准化
        1. 接受任意数量的关键字参数（**kwargs）并打印出这些参数的键和值
        1. 下载并保存预训练模型的权重

1. voc_annotation.py在训练之前进行
    1. 模式0和1会依照给出的数据集随机生成训练集、测试集、验证集，并保存在imagesets的txt文件中
    1. 模式0和2会从xml文件和txt文件中得到训练集和验证集的图片信息并生成两个txt文件存放在temp下
    1. txt文件中每一行分别为[文件路径 xmin,ymin,xmax,ymax,class_id  (xmin,ymin,xmax,ymax,class_id...) ]
    1. 模式0为模式1和2的加总

1. vision_for_centernet.py
    1. 运行后输出展示centernet作用的图片，没啥用

1. summary.py调用utils.centernet输出网络参数
    1. 输出网络参数，可以选择输出resnet50或者hourglass的参数

1. predict.py
    1. 用于预测图片中的内容，并画框保存图片
    1. 用于标记视频中的内容，指定帧率和路径就可以输出标记后的视频
    1. fps模式还没搞懂是用来干啥
    1. 用于标记一整个文件夹中的图片，指定路径就可以输出标记后的图片
    1. 表示将模型导出为onnx

1. get_map.py
    1. 调用centernet中的模型进行测试，输出mAP
    1. 问题是使用了centernet = CenterNet(confidence = confidence, nms_iou = nms_iou)，意味着centernet每次训练完都必须覆盖原来的模型，才能进行mAP测试

1. centernet.py
    1. 相比于nets中的centernet，其实这个centernet是调用了nets中centernet的一个综合的类，包括了训练、测试、预测、导出onnx等功能，相当于定义了很多函数支持train，predict等代码的运行。所以没有if __name__ == "__main__":，因为它就不是用来被直接被运行的


    