import numpy as np
import math
import cv2
import os
import json
from ...utils.box import BoundBox
from ...cython_utils.cy_yolo2_findboxes import box_constructor
from keras.models import load_model

IMG_HEIGHT = 300
IMG_WIDTH = 300
chk_dual_frames = []
two_frame_crop_image = []
cache = []
hit = [0]


def expit(x): #활성화 지수 함수 
    return 1. / (1. + np.exp(-x))

def _softmax(x): #활성화 함수 
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def findboxes(self, net_out): #상자 찾기 
    # meta
    meta = self.meta
    boxes = list()
    boxes=box_constructor(meta,net_out)
    return boxes

def postprocess(self, net_out, im, save = True):

    # print('\nFile: ', os.path.basename(im))
    boxes = self.findboxes(net_out) # 바운딩 박스

    # meta
    meta = self.meta
    threshold = meta['thresh'] #민감도 설정 변수 입력
    colors = meta['colors'] #인덱스 별로 다른 색상
    labels = meta['labels'] #라벨은 텍스트에서 레퍼런스
    if type(im) is not np.ndarray:
        imgcv = cv2.imread(im)
    else: imgcv = im
    h, w, _ = imgcv.shape #shape 찾기
    
    resultsForJSON = []
    crop_image_list = [] # 박스 이미지
    location_list = [] # 박스 위치
    real_keras_input = []
    
    for b in boxes:
        tmp_input = [] #임의 입력값

        boxResults = self.process_box(b, h, w, threshold)
        if boxResults is None:
            continue
        left, right, top, bot, mess, max_indx, confidence = boxResults
        thick = int((h + w) // 300)
        if self.FLAGS.json:
            resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
            continue
        
        tmp_input.append(int(left))
        tmp_input.append(int(right))
        tmp_input.append(int(top))
        tmp_input.append(int(bot))
        tmp_input.append(mess) #임시에 집어 넣기
        crop_img = imgcv[top:bot,left:right]
        
        crop_image_list.append(crop_img) #둘은 1:1 매칭 되는 형태임
        location_list.append(tmp_input)


    if len(chk_dual_frames) == 0: # 첫프레임
        chk_dual_frames.append(location_list)
        chk_dual_frames.append(location_list)
        two_frame_crop_image.append(crop_image_list)
        two_frame_crop_image.append(crop_image_list)
        # 1번 프레임은 무조건 인식해야함
        real_keras_input = crop_image_list
        prediction = predict(real_keras_input)
        for k, pred in enumerate(prediction):
            if pred < 0.5:
                chk_dual_frames[1][k][4] = 'aesop'
            else:
                chk_dual_frames[1][k][4] = 'kiehls' # 이때 앞뒤 값이 같이 바뀌는 이유는 리스트 append가 값을 참조하기 때문
            cache.append([chk_dual_frames[1][k], 1]) # 1번 프레임 객체들 모두 캐시에 입력
            hit[0] += 1


    elif len(chk_dual_frames) != 0: #2번 프레임
        chk_dual_frames[0] = chk_dual_frames[1]
        chk_dual_frames[1] = location_list
        two_frame_crop_image[0] = two_frame_crop_image[1]
        two_frame_crop_image[1] = crop_image_list #한 칸씩 앞으로 밀기
        #새 프레임이 들어오고

        #경우의 수 나누기
        if len(chk_dual_frames[0]) == len(chk_dual_frames[1]): # 1. 인식한 객체의 수가 같을때
            dist = is_same_dist(chk_dual_frames[0], chk_dual_frames[1])
            print('\n객체간 거리 :', dist)
            over_idx = overdist(dist)

            # 1.1 진또배기 같은 비슷해서 정확히 같은 객체가 위치만 조금 다른형태로 인식된경우
            if len(over_idx) == 0:
                #chk_dual_frames[0]에서 인식한 결과 그대로 가져다 쓰기
                #여기서 이름값 인덱스 4번인가 그거 그대로 베끼기
                print('\nsame\n')
                for i in range(len(chk_dual_frames[0])):
                    chk_dual_frames[1][i][4] = chk_dual_frames[0][i][4]
                hit[0] += len(chk_dual_frames[0])
                cacheManage(chk_dual_frames[1])

            # 1.2 인식한 객체의 수는 같은데 다른 객체를 일부 포함해서 일부만 거리가 다를때
            elif len(over_idx) > 0:
                print('\nnot same\n', over_idx)
                
                # 일단 이름 전부 붙이기
                for i in range(len(chk_dual_frames[0])):
                    if i in over_idx:
                        chk_dual_frames[1][i][4] = ''
                    else:
                        chk_dual_frames[1][i][4] = chk_dual_frames[0][i][4]
                hit[0] += len(chk_dual_frames[0]) - len(over_idx)

                real_keras_input=[]
                for x in over_idx:
                    real_keras_input.append(two_frame_crop_image[1][x])                
                #여기서 케라스로 들어가서 인식 후 결과 이름만 따오기
                #결과 이름 재조합 인덱스에 없는거 붙이기
                prediction = predict(real_keras_input)
                for i, x in enumerate(over_idx):
                    if chk_dual_frames[1][x][4] != '1':
                        sys.exit('!!! not same numbering got wrong !!!')
                    if prediction[i] < 0.5:
                        chk_dual_frames[1][x][4] = 'aesop'
                    else:
                        chk_dual_frames[1][x][4] = 'kiehls'
                cacheManage(chk_dual_frames[1])
        
        else: # 2. 인식한 객체의 수가 다를때
            #crop_image 그냥 인식 시키기
            #일단 완성 시키고 후에 유클리드 거리 기반 분류 코드 재 작성 ->고도화 포인트 1
            print('\ndifferent box number\n')
            real_keras_input = crop_image_list
            # if len(real_keras_input) == 0:
            #     print('\nzero box\n')
            #     chk_dual_frames[1] = [0]
            # else:
            prediction = predict(real_keras_input)
            for k, pred in enumerate(prediction):
                if pred < 0.5:
                    chk_dual_frames[1][k][4] = 'aesop'
                else:
                    chk_dual_frames[1][k][4] = 'kiehls'

    print('프레임 0  : ', chk_dual_frames[0],'\n프레임 1 : ', chk_dual_frames[1])
    # for i in range(len(cache)):
    #     print('캐시 ', i, ': ', cache[i][0], ', 히트: ', cache[i][1], '\n')
    print('히트 횟수: ', hit[0])
    print('\n')
    print('*' * 150)
    print('\n')

    # 박스 그리기
    for box in chk_dual_frames[1]:
        thick = int((h + w) // 300)
        cv2.rectangle(imgcv,
            (box[0], box[2]), (box[1], box[3]),
            colors[1], thick)
        
        cv2.putText(imgcv, box[4], (box[0], box[2] - 12),
            0, 1e-3 * h, colors[1], thick//3)

    if not save: return imgcv

    # 옵션: json으로 출력
    outfolder = os.path.join(self.FLAGS.imgdir, 'out')
    img_name = os.path.join(outfolder, os.path.basename(im))
    if self.FLAGS.json:
        textJSON = json.dumps(resultsForJSON)
        textFile = os.path.splitext(img_name)[0] + ".json"
        with open(textFile, 'w') as f:
            f.write(textJSON)
        return
    cv2.imwrite(img_name, imgcv)

    # print('프레임 0  : ', chk_dual_frames[0],'   프레임 1 : ', chk_dual_frames[1])
    # print('\n')
    # print('*' * 150)
    # print('\n')


def calculate_distance(a, b):
    dist = (((a[0]-b[0])**2 + (a[2]-b[2])**2)**0.5 + ((a[0]-b[0])**2 + (a[3]-b[3])**2)**0.5+((a[1]-b[1])**2 + (a[2]-b[2])**2)**0.5+((a[1]-b[1])**2 + (a[3]-b[3])**2)**0.5)/4
    return dist

def is_same_dist(a, b): # 두 이미지 박스 개수 같을 때 박스 간 거리
    dist = []
    for x in range(len(a)):
        dist.append(calculate_distance(a[x],b[x]))
    return dist

def overdist(a):
    over_num=[]
    for  x in range(len(a)):
        if a[x]>100:
            over_num.append(x)
    return over_num

def predict(img_list): # 리사이징, 예측 함수 (임시로 300 * 300 * 3 크기)
    prediction = []
    model = load_model('/Users/hyewon/PycharmProjects/toner-keras/model_vgg_0630.h5')
    print('Number: ', len(img_list))
    for img_arr in img_list:
        print('Shape: ', img_arr.shape)
        img_arr = cv2.resize(img_arr, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        x = img_arr.reshape([-1, IMG_HEIGHT, IMG_WIDTH, 3])
        output = model.predict(x)
        print('Output: ', output[0][0])
        prediction.append(output[0][0])

    return prediction

def cacheManage(objects): # 캐시 매니징
    for obj in objects:
        