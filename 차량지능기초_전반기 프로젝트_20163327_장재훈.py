# 필요한 모듈 import
import cv2 
import mediapipe as mp
import math
import numpy as np 
import os 
import time
import torch

# Mediapipe의 face mesh 모델 가져오기
# face mesh 속성 설정
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Mediapipe의 face detection 모델 가져오기
# face detection 속성 설정
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    min_detection_confidence = 0.5)

# Drawing 스펙 설정
mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(color = (255,255,0), thickness=1, circle_radius=1)

# 눈과 입의 특징을 추출하는 함수 정의

def distance(p1, p2): # 두 point간 거리 구하는 함수, Euclidean distance
    return (((p1[:2] - p2[:2])**2).sum())**0.5

def eye_aspect_ratio(landmarks, eye): # 눈 길이(horizontal line)와 눈 너비(vertical line)의 비
    N1 = distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
    N2 = distance(landmarks[eye[2][0]], landmarks[eye[2][1]])
    N3 = distance(landmarks[eye[3][0]], landmarks[eye[3][1]])
    D = distance(landmarks[eye[0][0]], landmarks[eye[0][1]])
    return (N1 + N2 + N3) / (3 * D)

def eye_feature(landmarks): # 양쪽 눈의 eye_aspect_ratio
    return (eye_aspect_ratio(landmarks, left_eye) + eye_aspect_ratio(landmarks, right_eye))/2

def mouth_feature(landmarks):# eye_aspect_ratio와 마찬가지로 입의 길이와 입 너비의 비를 구하는 함수
    N1 = distance(landmarks[mouth[1][0]], landmarks[mouth[1][1]])
    N2 = distance(landmarks[mouth[2][0]], landmarks[mouth[2][1]])
    N3 = distance(landmarks[mouth[3][0]], landmarks[mouth[3][1]])
    D = distance(landmarks[mouth[0][0]], landmarks[mouth[0][1]])
    return (N1 + N2 + N3)/(3*D)

# FaceMesh mediapipe model의 face landmark를 이용하여 pupil circularity(동공 원형도) 계산
def pupil_circularity(landmarks, eye):
    perimeter = distance(landmarks[eye[0][0]], landmarks[eye[1][0]]) +             distance(landmarks[eye[1][0]], landmarks[eye[2][0]]) +             distance(landmarks[eye[2][0]], landmarks[eye[3][0]]) +             distance(landmarks[eye[3][0]], landmarks[eye[0][1]]) +             distance(landmarks[eye[0][1]], landmarks[eye[3][1]]) +             distance(landmarks[eye[3][1]], landmarks[eye[2][1]]) +             distance(landmarks[eye[2][1]], landmarks[eye[1][1]]) +             distance(landmarks[eye[1][1]], landmarks[eye[0][0]])
    area = math.pi * ((distance(landmarks[eye[1][0]], landmarks[eye[3][1]]) * 0.5) ** 2)
    return (4*math.pi*area)/(perimeter**2)

# 양쪽 눈의 pupil_circularity를 더하여 pupil feature 구하기
def pupil_feature(landmarks): 
    return (pupil_circularity(landmarks, left_eye) + pupil_circularity(landmarks, right_eye))/2

# image를 받아 eye, mouth, pupil, eye와 mouth의 합, 총 4가지 feature와 drawing image까지 5가지 return하는 함수
def run_face_mp(image):
    
    # image를 수평축을 기준으로 뒤집고 BGR image를 RGB image로 변환한다.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # face mesh와 face detection process 진행
    results = face_mesh.process(image)
    results2 = face_detection.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Image에 face detection 박스 표시
    if results2.detections:
        for detection in results2.detections:
            mp_drawing.draw_detection(image, detection)
        cv2.putText(image,'There is Driver',(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3,cv2.LINE_4,False)
    else:
        cv2.putText(image,'There is no Driver',(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,cv2.LINE_4,False)
    
    # image의 face mesh landmark를 이용하여 feature 값을 추출하고, face mesh를 그린다.
    if results.multi_face_landmarks:
        landmarks_positions = []
        # 이미지에 얼굴만 있다고 가정한다.
        for _, data_point in enumerate(results.multi_face_landmarks[0].landmark):
            landmarks_positions.append([data_point.x, data_point.y, data_point.z]) # normalize된 ladmark position 추가
        landmarks_positions = np.array(landmarks_positions)
        landmarks_positions[:, 0] *= image.shape[1]
        landmarks_positions[:, 1] *= image.shape[0]

        # Imgage에 Face mesh 그리기
        for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
                
        # 위에서 정의했던 함수들을 이용하여 feature를 추출한다.
        ear = eye_feature(landmarks_positions)
        mar = mouth_feature(landmarks_positions)
        puc = pupil_feature(landmarks_positions)
        moe = mar/ear
    else:
        ear = -1000
        mar = -1000
        puc = -1000
        moe = -1000

    return ear, mar, puc, moe, image

# input data로 20 frame의 feature list를 받는다.
# 이를 바탕으로 운전자의 상태(정상, 졸음운전) 판단한다.
def get_classification(input_data):
    model_input = []
    model_input.append(input_data[:5])
    model_input.append(input_data[3:8])
    model_input.append(input_data[6:11])
    model_input.append(input_data[9:14])
    model_input.append(input_data[12:17])
    model_input.append(input_data[15:])
    model_input = torch.FloatTensor(np.array(model_input))
    preds = torch.sigmoid(model(model_input)).gt(0.5).int().data.numpy()
    return int(preds.sum() >= 5)

# 중립 position(기준점)에서 feature를 얻기 위하여 calibrate함수를 정의한다.
# default frame은 25로 설정하고, eye, mouth, pupil, eye+mouth 총 4가지의 normalization feature를 return 한다.
def calibrate(calib_frame_count=25):
    
    # Normalize feature value를 담기 위한 빈 리스트 선언
    ears = []
    mars = []
    pucs = []
    moes = []

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
            
        # 앞서 만들었던 run_face_mp 함수를 이용하여 feature value를 얻고, 이를 기준값으로 삼는다.
        ear, mar,puc, moe, image = run_face_mp(image)
        if ear != -1000:
            ears.append(ear)
            mars.append(mar)
            pucs.append(puc)
            moes.append(moe)

        cv2.putText(image, "Calibration", (0,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3, cv2.LINE_4, False)
        cv2.imshow('Calibration', image)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
        if len(ears) >= calib_frame_count: # ear feature의 크기가 25보다 크면 중지하고, 상태 진단을 시작한다.
            break

    cv2.destroyAllWindows()
    cap.release()
    
    # 추출한 각 feature를 행렬 형태로 변환하고, normalize하여 리스트 형태로 반환한다.
    ears = np.array(ears)
    mars = np.array(mars)
    pucs = np.array(pucs)
    moes = np.array(moes)
    return [ears.mean(), ears.std()], [mars.mean(), mars.std()], [pucs.mean(), pucs.std()], [moes.mean(), moes.std()]

# 운전자의 상태를 모니터링하는 infer 함수 정의
# calibration을 통해 얻은 normalization feature를 인자로 받는다.
def infer(ears_norm, mars_norm, pucs_norm, moes_norm):
    ear_main = 0
    mar_main = 0
    puc_main = 0
    moe_main = 0
    decay = 0.9 # feature 값의 noise를 부드럽게 하기 위하여 decay 사용

    label = None

    input_data = []
    frame_before_run = 0

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        ear, mar, puc, moe, image = run_face_mp(image)
        if ear != -1000:
            ear = (ear - ears_norm[0])/ears_norm[1]
            mar = (mar - mars_norm[0])/mars_norm[1]
            puc = (puc - pucs_norm[0])/pucs_norm[1]
            moe = (moe - moes_norm[0])/moes_norm[1]
            if ear_main == -1000:
                ear_main = ear
                mar_main = mar
                puc_main = puc
                moe_main = moe
            else:
                ear_main = ear_main*decay + (1-decay)*ear
                mar_main = mar_main*decay + (1-decay)*mar
                puc_main = puc_main*decay + (1-decay)*puc
                moe_main = moe_main*decay + (1-decay)*moe
        else:
            ear_main = -1000
            mar_main = -1000
            puc_main = -1000
            moe_main = -1000
        
        if len(input_data) == 20:
            input_data.pop(0)
        input_data.append([ear_main, mar_main, puc_main, moe_main])

        frame_before_run += 1
        if frame_before_run >= 15 and len(input_data) == 20:
            frame_before_run = 0
            label = get_classification(input_data)
            print ('got label ', label)
        
        # label이 0이면 정상 상태, label이 1이면 졸음 운전
        if label is not None:
            if label == 0:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.putText(image, "%s" %(states[label]), (0,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_4, False)
            
        # window 창으로 결과 출력
        cv2.imshow('Driver Status Monitoring', image)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
    
    cv2.destroyAllWindows()
    cap.release()

right_eye = [[33, 133], [160, 144], [159, 145], [158, 153]] # right eye landmark 좌표
left_eye = [[263, 362], [387, 373], [386, 374], [385, 380]] # left eye landmark 좌표
mouth = [[61, 291], [39, 181], [0, 17], [269, 405]] # mouth landmark 좌표
states = ['Nice driving', 'Drowsy. Wake up!!'] # 운전자의 상태 label

# LSTM 모델을 이용하여 운전자 상태 예측
# 5장의 frame을 거쳐 예측을 하고, 예측 상태를 return한다.
model_lstm_path = 'Downloads\clf_lstm_jit6.pth'
model = torch.jit.load(model_lstm_path)
model.eval()

# calibrate 함수를 통해 normalize feature 획득
print ('Starting calibration. Please be in neutral state')
time.sleep(1)
ears_norm, mars_norm, pucs_norm, moes_norm = calibrate()

# normalize feature를 이용하여 운전자 상태 모니터링
print ('Start monitoring')
time.sleep(1)
infer(ears_norm, mars_norm, pucs_norm, moes_norm)

# mediapipe의 face mesh와 face detection을 끝낸다.
face_mesh.close()
face_detection.close()




