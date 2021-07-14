'''
Date: 2021-07-05 14:21:06
LastEditors: Liuliang
LastEditTime: 2021-07-06 16:23:45
Description: 
'''

import numpy as np
# import cv2
import matplotlib.pyplot as plt

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    # x1、y1、x2、y2、以及score赋值
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    # 每一个检测框的面积
    areas = (x2 - x1 ) * (y2 - y1 )
    # 按照score置信度降序排序
    order = scores.argsort()[::-1]

    pick = []  # 保留的结果框集合
    while order.size > 0:
        i = order[0]
        pick.append(i)  # 保留该类剩余box中得分最高的一个
        # 得到相交区域,左上及右下
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算相交的面积,不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算IoU：重叠面积 /（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留IoU小于阈值的box
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]  # 因为ovr数组的长度比order数组少一个,所以这里要将所有下标后移一位

    return pick


def non_max_suppression(dets, threshold):
    """执行non-maximum suppression并返回保留的boxes的索引.
    dets:(x1、y1、x2、y2,scores)
    threshold: Float型. 用于过滤IoU的阈值.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    # 每一个检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 获取根据分数排序的boxes的索引(最高的排在对前面)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # 选择排在最前的box，并将其索引加到列表中
        i = ixs[0]
        pick.append(i)
        # 计算选择的box与剩下的box的IoU

        xx1 = np.maximum(x1[i], x1[ixs[1:]])
        yy1 = np.maximum(y1[i], y1[ixs[1:]])
        xx2 = np.minimum(x2[i], x2[ixs[1:]])
        yy2 = np.minimum(y2[i], y2[ixs[1:]])

        # 计算相交的面积,不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算IoU：重叠面积 /（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[ixs[1:]] - inter)

        remove_ixs = np.where(ovr > threshold)[0] + 1
        # 将选择的box和重叠的boxes的索引删除.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick)



if __name__=='__main__':

    det_result=np.array([[20,5,200,100,0.353],
            [50,50,200,200,0.624],
            [20,120,150,150,0.667],
            [250,250,400,350,0.5],
            [90,10,300,300,0.3],
            [40,220,280,380,0.46]])
    # det_result=np.array([[20,50,200,100,0.353],
    # [50,50,200,200,0.624],])
    colors=['red','blue','green','yellow','pink','gray']
    plt.figure()

    for i in range(len(det_result)):
        # plt.gca().add_patch(plt.Rectangle(xy=(det_result[i][1], det_result[i][0]),
        #                               width=det_result[i][3] - det_result[i][1],
        #                               height=det_result[i][2] - det_result[i][0],
        #                               edgecolor='green',
        #                               fill=False, linewidth=2))
        plt.plot([det_result[i][1], det_result[i][3], det_result[i][3], det_result[i][1],
                  det_result[i][1]],  # col
                 [det_result[i][0], det_result[i][0], det_result[i][2], det_result[i][2],
                  det_result[i][0]],  # row
                 color=colors[i], marker='.', ms=0)
    plt.show()    
    plt.savefig("./NMS_1.jpg")
    
    plt.close()

    result=py_cpu_nms(det_result,0.1)
    # result=non_max_suppression(det_result,0.2)
    print(result)
    for j in range(len(result)):
        # plt.gca().add_patch(plt.Rectangle(xy=(det_result[i][1], det_result[i][0]),
        #                               width=det_result[i][3] - det_result[i][1],
        #                               height=det_result[i][2] - det_result[i][0],
        #                               edgecolor='green',
        #                               fill=False, linewidth=2))
        i = result[j]
        plt.plot([det_result[i][1], det_result[i][3], det_result[i][3], det_result[i][1],
                  det_result[i][1]],  # col
                 [det_result[i][0], det_result[i][0], det_result[i][2], det_result[i][2],
                  det_result[i][0]],  # row
                 color=colors[i], marker='.', ms=0)
    plt.show()
    plt.savefig("./NMS_2.jpg")
    plt.close()