import cv2
import numpy as np
import sys
import time

# 開啟網路攝影機
#cap = cv2.VideoCapture('D:/不規則抗體微流體凝集/各流道有凝集.avi')
cap = cv2.VideoCapture('D:/不規則抗體微流體凝集/有凝集.avi')
#cap = cv2.VideoCapture('D:/不規則抗體微流體凝集/無凝集.avi')

# 設定影像尺寸
width = 280
height = 60

# 設定擷取影像的尺寸大小
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# 初始化平均影像
def getBgndValue(*args):
    '''args= [frame_idx]'''
    if args[0]:
        idx = args[0]-1        
    else:
        idx = args[0]
    status, frame = cap.read(idx)
    #avg = cv2.blur(frame, (4, 4))
    avg = cv2.GaussianBlur(frame, (9, 9), 0)
    avg_float = np.float32(avg)
    
    #set up the begining frame index
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx+2)
    return avg, avg_float

frame_idx = 1

#set up the begining frame index
cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
while(cap.isOpened()):
    try:
            
        # get the frqme index  
        frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
        #print (frame_idx)
        
        # 讀取一幅影格
        ret, frame = cap.read()
        
        # 若讀取至影片結尾，則跳出
        if ret == False:
            break
        
        # 模糊處理
        #blur = cv2.blur(frame, (4, 4))
        blur = cv2.GaussianBlur(frame, (9, 9), 0)
        
        # get the coefficient of the frame         
        avg, avg_float = getBgndValue(frame_idx)

        # 計算目前影格與平均影像的差異值
        diff = cv2.absdiff(avg, blur)        
                
        '''if there's mismatch between previous frame and current one,
           skip this round'''
        if not diff.sum():
            continue
        
        # 將圖片轉為灰階
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # 篩選出變動程度大於門檻值的區域
        #ret, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        
        # 使用型態轉換函數去除雜訊
        kernel = np.ones((5, 15), np.uint8)
        thresh = cv2.morphologyEx(thresh,
                                  cv2.MORPH_OPEN,
                                  kernel,
                                  iterations=2)
        thresh = cv2.morphologyEx(thresh,
                                  cv2.MORPH_CLOSE,
                                  kernel,
                                  iterations=6)
        
        
        # 產生等高線 找輪廓
        cntImg, cnts, _ = cv2.findContours(thresh.copy(),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
        _sum = 0
        for elm in cnts:
            _sum += elm.sum() #_sum = _sum + elm.sum()
        #print (_sum)
        if _sum >= 25000: 
            cv2.imshow('frame', frame)
            continue
        
        for c in cnts:
            # 忽略太小的區域
            #print (c.sum(), cv2.contourArea(c), cv2.arcLength(c, True))
            if cv2.contourArea(c) <= 200 or cv2.arcLength(c, True) <= 30:
                cv2.imshow('frame', frame)
                continue
            '''
            if c.sum() <= layer1_size:
                if cv2.contourArea(c) <= 500:
                    continue
            else: 
                layer1_size += 500
                continue
            '''
            # 計算等高線的外框範圍
            #(x, y, w, h) = cv2.boundingRect(c)
            box = cv2.minAreaRect(c)    
            box = np.int0(cv2.boxPoints (box))
            
            # 畫出外框
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
            cv2.drawContours(frame, cnts, -1, (0, 0, 255), 2)
            
            # save frame as JPEG file
            cv2.imwrite("D:\\opencv_practice\\pic\\frame{}.jpg".format(frame_idx), frame)
            time.sleep(0.01)
            
        # 畫出等高線（除錯用）
        #cv2.drawContours(frame, cnts, -1, (0, 255, 255), 2)
        
        # 顯示偵測結果影像
        cv2.imshow('frame', frame)
        
        # press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # press 'p' to pause
        if cv2.waitKey(1) & 0xFF == ord('p'):
            cv2.waitKey(0)
        
        # 更新平均影像
        #cv2.accumulateWeighted(blur, avg_float, 0.01)
        #avg = cv2.convertScaleAbs(avg_float)
    except:
        type, message, traceback = sys.exc_info()
        print ('type:{}\nmsg:{}\ntraceback:{}'.format(type, message, traceback))
        cap.release()
        cv2.destroyAllWindows()
cap.release()
cv2.destroyAllWindows()
    