# 目标检测常用指标
1. AP，即Average Precision（平均精确度），是物体检测中的一个重要评价指标，它衡量了模型在各个阈值下的性能。AP是通过计算Precision-Recall（精确率-召回率）曲线下的面积来得出的，它反映了模型在不同的召回率下精确率的平均水平。

    Precision：精确率是预测为正例的结果中，正确预测的比例。它衡量了模型预测出来的正确率。
    Recall：召回率是所有正例样本中，成功预测出来的比例。它衡量了模型成功预测正例的能力。

    AP的计算通常涉及以下步骤：

    绘制PR曲线，其中纵轴是Precision，横轴是Recall。
    平滑处理PR曲线，即对PR曲线上的每个点，Precision的值取该点右侧最大的Precision的值。
    计算平滑后的PR曲线与坐标轴围成的面积，这个面积就是AP。
    在实际应用中，AP的计算方式可能会有所不同。例如，在Pascal VOC 2008中，AP的计算是通过在平滑处理的PR曲线上取10等分点的Precision值，并计算其平均值得到的。而在COCO数据集中，为了提高精度，采用了更多的采样点和不同的IoU阈值来计算AP。

    mAP，即mean Average Precision（平均平均精确度），是AP在所有类别下的均值。在某些上下文中，例如在COCO数据集中，AP已经是mAP，因为它已经计算了所有类别下的平均值。

    在评估目标检测模型时，AP和mAP是非常重要的指标，因为它们能够反映模型在不同难度级别的目标检测任务上的性能。

2.  在目标检测中，FPS（Frames Per Second，每秒帧数）

# 目标检测
1. 非极大值抑制（Non-Maximum Suppression，简称NMS算法）是一种在计算机视觉任务中广泛应用的技术，它的主要思想是搜索局部最大值并抑制非极大值。在目标检测等任务中，同一目标的位置可能会产生大量的候选框，这些候选框之间可能会有重叠。为了消除冗余的边界框，我们需要使用非极大值抑制找到最佳的目标边界框。

非极大值抑制的具体流程如下：

根据置信度得分对候选框进行排序。
选择置信度最高的边界框添加到最终输出列表中，并从候选框列表中删除。
计算所有边界框的面积。
计算置信度最高的边界框与其他候选框的交并比（Intersection over Union，IoU）。
删除IoU大于设定阈值的边界框。
重复上述过程，直到候选框列表为空。

2. 交并比（Intersection over Union，简称IoU）是一个衡量两个矩形框（或者更一般的形状）重叠程度的指标。在计算机视觉和目标检测中，IoU用于量化检测框（通常由模型预测产生）与真实框（标注的目标位置）之间的重叠程度。重叠区域越大，IoU的值越大1。