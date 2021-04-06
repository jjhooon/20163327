from ctypes import * #ctypes는 파이썬에서 DLL 로딩하여 dll에서 제공하는 함수 호출을 가능하게 하는 모듈이다
import math #수학 관련 함수를 사용하기 위해 math 모듈을 import
import random #random 모듈 import

def sample(probs): #probs를 인자로 받은 sample 함수 정의한다.
    s = sum(probs) #probs를 모두 더해 s에 저장한다.
    probs = [a/s for a in probs] #probs의 각 요소를 s로 나누고, 리스트에 넣어 probs로 다시 지정해준다.
    r = random.uniform(0, 1) 
    for i in range(len(probs)): #probs의 크기만큼 반복, r값이 0보다 작으면 i값을 아니면 probs의 크기-1을 리턴한다.
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values): #ctype과 value를 인자로 받는 c_array 함수 정의한다.
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

# 아래의 BOX, DETECTION, IMAGE, METADATA class는 모두 Structure를 상속받은 class이다.

class BOX(Structure): # 각 grid cell은 2개의 Bounding box를 예측하고, bounding box는 예측값을 가진다.
    _fields_ = [("x", c_float), # x,y는 중심점의 좌표
                ("y", c_float),
                ("w", c_float), # w는 너비
                ("h", c_float)] # h는 높이

class DETECTION(Structure):
    _fields_ = [("bbox", BOX), #bounding box
                ("classes", c_int), 
                ("prob", POINTER(c_float)), #prob array의 확률
                ("mask", POINTER(c_float)), #mask 정보
                ("objectness", c_float), #객체를 포함하고 있을 가능성(objectness)
                ("sort_class", c_int)] #class 분류


class IMAGE(Structure): 
    _fields_ = [("w", c_int), #w는 너비
                ("h", c_int), #h는 높이
                ("c", c_int), #c는 confidence(신뢰도)
                ("data", POINTER(c_float))] #data 정보

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
#make하여 컴파일 하면 libdarknet.so라는 파일이 생성되고, CDLL을 이용하여 모듈을 로딩한다.
#Ubuntu terminal 창에서 nm 명령어를 통하여 아래의 내장함수들이 libdarknet.so 파일 안에 있는 것을 확인할 수 있었다.
#libdarknet.so 파일 안의 내장함수를 이용하여 변수 데이터 타입을 지정하였다. (line 53~119)
lib = CDLL("libdarknet.so", RTLD_GLOBAL) 
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im): #main문에서 지정될 net, met 변수를 인자로 받는 분류 함수 classify를 정의한다.
    out = predict_image(net, im) #libdarknet.so의 network_predict_image 함수를 받은 predict_img가 net과 image을 인자로 받는다.
    res = []
    for i in range(meta.classes): #coco.data에는 80개의 class가 있으므로 80번 반복한다. 
        res.append((meta.names[i], out[i])) #res 리스트에 coco dataset의 class이름과 out을 추가한다.(스택 형태)
    res = sorted(res, key=lambda x: -x[1]) #res의 1번째 index 원소를 기준으로 내림차순 정렬한다.
    return res #res 리턴

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    #main문의 net, meta와 image, threshold 초기값과 nms(non-maximum-suppression) 초기값을 가진 detect 함수를 정의한다.
    #nms의 IOU가 nms의 threshold 값보다 크면 동일한 물체를 detect했다고 판단하고 지운다.
    im = load_image(image, 0, 0) #image를 불러 im에 저장한다.
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im) # net과 im을 받아 이미지를 예측한다.
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum) #위에 적혀있는 datatype(line 71)에 맞게 변수를 넣어준다.
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms); #detect된 박스에 대해 non-maximum-suppression을 진행하는 반복문이다.

    res = []
    for j in range(num):
        for i in range(meta.classes): #coco.data에는 80개의 class가 있으므로 80번 반복한다. 
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1]) #res의 1번째 index 원소를 기준으로 내림차순 정렬한다.
    free_image(im)
    free_detections(dets, num)
    return res
    
if __name__ == "__main__":
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]
    net = load_net("cfg/tiny-yolo.cfg", "tiny-yolo.weights", 0) #tiny-yolo.cfg파일과 weights를 불러 net에 넣는다. #cfg파일에는 batch size, learing rate와 같은 network학습에 필요한 정보가 담겨있다.
    meta = load_meta("cfg/coco.data")  #coco data의 정보를 불러 meta에 넣는다.
    r = detect(net, meta, "data/dog.jpg") #앞서 선언한 net, meta 변수와 detect 함수를 이용하여 data/dog.jpg 파일에서 object detection을 한다.
    print r
    

