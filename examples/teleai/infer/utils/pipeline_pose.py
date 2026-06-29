import cv2
import numpy as np
import math

def draw_smpl_body(canvas, points, stickwidth=2,r=2, box=None, scores=None, score_thre=None):
    if box is not None:
        x_pad, y_pad = (box[2] - box[0]) / 20. , (box[3] - box[1]) / 20.0
        box[0], box[2] = box[0] - x_pad, box[2] + x_pad
        box[1], box[3] = box[1] - x_pad, box[3] + x_pad
        
    colors = [
        [255, 0, 0], # 0
        [0, 255, 0], # 1
        [0, 0, 255], # 2
        [255, 0, 255], # 3
        [255, 255, 0], # 4
        [85, 255, 0], #5
        [0, 75, 255], #6
        [0, 255, 85], #7
        [0, 255, 170], #8
        [170, 0, 255], #9
        [85, 0, 255], #10
        [0, 85, 255], #11
        [0, 255, 255], #12
        [85, 0, 255], #13
        [170, 0, 255], #14
        [255, 0, 255], #15
        [255, 0, 170], #16
        [255, 0, 85], #17
    ]
    connetions = [
        [15,12],[12, 16],[16, 18],[18, 20],[20, 22],
        [12,17],[17,19],[19,21],
        [21,23],[12,9],[9,6],
        [6,3],[3,0],[0,1],
        [1,4],[4,7],[7,10],[0,2],[2,5],[5,8],[8,11]
    ]
    connection_colors = [
        [255, 0, 0], # 0
        [0, 255, 0], #1
        [0, 0, 255], #2
        [255, 255, 0], #3
        [255, 0, 255], #4
        [0, 255, 0], #5
        [0, 85, 255], #6
        [255, 175, 0], # 7
        [0, 0, 255], ## 8
        [255, 85, 0], #9
        [0, 255, 85], #10
        [255, 0, 255], #11
        [255, 0, 0], #12
        [0, 175, 255], #13
        [255, 255, 0], #14
        [0, 0, 255], #15
        [0, 255, 0], #16
    ]

    # draw point
    for i in range(len(points)):
        x,y = points[i][0:2]
        x,y = int(x),int(y)
        if i==13 or i == 14:
            continue
        if box is not None and (x < box[0] or x > box[2] or y < box[1] or y > box[3]): # 在box外不绘制
            continue
        if score_thre is not None and scores is not None and scores[i] > score_thre:
            continue
        cv2.circle(canvas, (x, y), r, colors[i%17], thickness=-1)

    # draw line
    for i in range(len(connetions)):
        point1_idx,point2_idx = connetions[i][0:2]
        point1 = points[point1_idx]
        point2 = points[point2_idx]
        Y = [point2[0],point1[0]]
        X = [point2[1],point1[1]]
        if box is not None and (Y[0] < box[0] or Y[0] > box[2] or Y[1] < box[0] or Y[1] > box[2] or X[0] < box[1] or X[0] > box[3] or X[1] < box[1] or X[1] > box[3]): # 在box外不绘制
            continue
        
        if score_thre is not None and scores is not None and (scores[point1_idx] > score_thre or scores[point2_idx] > score_thre):
            continue

        mX = int(np.mean(X))
        mY = int(np.mean(Y))
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((mY, mX), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, connection_colors[i%17])
    return canvas

