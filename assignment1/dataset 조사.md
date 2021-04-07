### 1. 자율주행 인지에 관련된 3종 이상의 공개 Data Set 조사, 정리하기 ###

자율주행 인지 시스템은 LiDAR, Radar, Camera 같은 센서를 이용하여 주행 환경을 인식한다, 이처럼 다양한 센서를 이용하여 얻은 데이터는 자율주행 자동차의 인지를 위한 딥러닝 모델의 학습 데이터로 이용된다. 많은 공개 Data Set 중 대표적인 몇 가지를 알아보자.

#### 1) KITTI Dataset ####

KITTI(Karlsruhe Institute of Technology and Toyota Technological Institue)는 자율주행에 사용되는 가장 인기 있는 dataset 중 하나이다. 이 dataset은 stereo, optical/scene flow, visual odometry, 2D/3D obeject detection & tracking, segmentation과 같은 다양한 컴퓨터 비전 연구를 위해 만들어졌다. 이를 위해, KITTI dataset은 GPS/IMU 센서, Velodyne 사의 Laser scanner, 두 개의 고해상도 color/grayscale 비디오카메라를 장착한 자율주행 플랫폼 'Annieway'를 이용하여 독일 Karlsruhe의 도시, 시골, 고속도로 환경을 주행하며 만들어졌다. 이미지 하나 당 최대 15대의 자동차와 30명의 보행자를 확인할 수 있다. KITTI는 모든 데이터를 원본 형태로 제공하는 것 외에도, 각각의 task에 대한 benchmark를 추출하고 각 benchmark에 대한 평가 사이트를 제공한다.

![image](https://user-images.githubusercontent.com/81551992/113826397-12812a00-97bd-11eb-884b-8fbfe6dfbfd1.png)

<그림 1> 자율주행 플랫폼 'Annieway'

제공되는 Raw data는 10Hz로 수집 및 synchronize 되었고, 6가지 카테고리(도심, 거주 지역, 도로, 캠퍼스, 사람, 캘리브레이션)로 분류할 수 있다. Raw dataset의 종류는 3D object tracklet labels(차량, 트럭, 보행자 등), Calibration, 3D GPS/IMU data, 3D Velodyne point clouds, 원본/전처리 된 grayscale stereo, 원본/전처리 된 color stereo로 구성되어 있다. Raw dataset외에도 앞서 언급하였던 다양한 컴퓨터 비전 task들에 대한 dataset 또한 다운받을 수 있다. 

![image](https://user-images.githubusercontent.com/81551992/113826746-7c99cf00-97bd-11eb-8383-d2121483e7c1.png) ![image](https://user-images.githubusercontent.com/81551992/113826758-802d5600-97bd-11eb-9055-be6e907523b2.png)
![image](https://user-images.githubusercontent.com/81551992/113826770-83284680-97bd-11eb-99db-49553adc12a4.png) ![image](https://user-images.githubusercontent.com/81551992/113826782-86233700-97bd-11eb-9e15-9663e0fc2436.png)
![image](https://user-images.githubusercontent.com/81551992/113826785-87546400-97bd-11eb-8c90-c6f00cdb08ff.png) ![image](https://user-images.githubusercontent.com/81551992/113826799-8a4f5480-97bd-11eb-9b51-2cd42ad2a6a2.png)

<그림2> 6가지 카테고리(도심, 거주 지역, 도로, 캠퍼스, 사람, 캘리브레이션)의 raw data

KITTI Vision Benchmark Suite 사이트에는 자율주행 자동차를 위한 다양한 task에 대한 benchmark들이 올라와있다. 예를 들어, detection, tracking, depth, odometry, sceneflow, stereo, segmentation이 있다. 이 중 몇가지 task에 대해 살펴보면 다음과 같다. 

① Visual Odometry

Odometry benchmark는 22개의 연속된 stereo로 구성되어 있다. 이 중, 11개는 training을 위한 ground_truth 경로로 구성되어 있고, 나머지는 평가를 위하여 ground_truth가 제공되어 있지 않다. 이 benchmark의 경우, laser 기반의 SLAM이나 LiDAR 정보를 결합한 알고리즘을 이용하여 monocular나 stereo visual odometry를 제공할 수 있다. 하지만 한가지 제한점은 모든 방법이 완전히 automatic하며 parameter set이 모든 과정에 동일하게 적용되는 것이다.

![image](https://user-images.githubusercontent.com/81551992/113826942-b36fe500-97bd-11eb-9c52-ad773127ffe9.png)

<그림 3> Visual Odometry

② 3D Object Detection

3D Object detection benchmark는 80.256 labeled objects뿐만 아니라, 7481개의 training image와 7518개의 test images로 구성되어 있다. 평가를 위하여 precision - recall 곡선을 계산하였고, AP 지표를 이용하여 method의 순위를 매겼다. 또한, 2D object detection에서 사용하였던 PASCAL 기준을 사용하여 3D object detection의 performance를 평가하였다. 

![image](https://user-images.githubusercontent.com/81551992/113827003-c4205b00-97bd-11eb-9257-1f7b97daf369.png)

<그림 4> 3D object detection

③ Scene Flow

Stereo flow 2015 / flow 2015 / scene flow 2015 benchmarks는 4가지의 색상으로 구성된 200장의 training scenes와 200장의 test scenes로 구성되어 있다. 이것들은 ground_truth 값이 반자종 과정을 통하여 얻어진 dynamic scenes으로 이뤄져 있다. 평가는 전체 200장의 test image의 ground_truth pixel과 bad pixel의 비율(percentage)로 계산된다. 이 benchmark에서는 disparity나 flow end-point error가 3px보다 작거나 5%보다 작을 시, pixel이 올바르게 추정되었다고 간주한다.

![image](https://user-images.githubusercontent.com/81551992/113827039-d00c1d00-97bd-11eb-8c6a-00e95a87fb34.png)

하지만, KITTI dataset은 독일 Karlsruhe라는 한정적인 지역과 맑은 날씨에서만 수집되었기 때문에 예외인 상황(교통 문화가 다른 국가, 비 오는 날씨)에서는 한계를 지닌다.

#### 2) BDD100K ####

BDD는 Berkeley Deep Drive의 약자로 BDD100K는 10만 개의 비디오로 구성되어 있는 가장 큰 자율주행 인지 dataset이다. 각 비디오는 40초 길이, 720p의 높은 해상도, 초당 30프레임으로 취득되었다. BDD100K는 미국의 New York, SanFranciso, Berkeley와 같은 다양한 지역에서 수집되었고 거주 지역, 도시 거리, 고속도로와 같은 다양한 주행 환경을 담고 있다. 뿐만 아니라, 주간/야간의 다양한 시간에서 흐린 날씨, 맑은 날씨, 비 오는 날씨, 안개와 같은 다양한 날씨의 영상이 기록되어 있다. 이 비디오들은 7만 개의 training set, 1만 개의 validation set, 2만 개의 testing set으로 나눠졌고, 각 비디오의 10th second 프레임에는 이미지 작업을 위한 라벨이 달려있다. 
![image](https://user-images.githubusercontent.com/81551992/113827144-eca85500-97bd-11eb-9413-d0dca1f0bcd5.png)

<그림 6> BDD100K

![image](https://user-images.githubusercontent.com/81551992/113827229-034eac00-97be-11eb-8e28-9e67cf3db984.png)

<그림 7> 날씨, 장면, 시간에 따른 dataset의 분포

BDD100K의 benchmark는 image tagging, lane detection, driveable area segmentation, road object detection, semantic segmentation, instance segmentation, multi-object tracking, domain adaption, imitation learning의 10가지 task들로 구성되어 있다. 이 중 몇 가지 task에 대해 자세히 살펴보자.

① Lane Marking

BDD100K는 차선 내의 차량을 어떻게 지시하는지에 따라 두 가지로 lane marking을 하였다. 차량의 운전 방향을 의미하는 수직 차선은 적색으로 표시하였고, 차량이 멈춰야하는 차선을 의미하는 수평 차선은 청색으로 표시하였다. 

![image](https://user-images.githubusercontent.com/81551992/113827255-0d70aa80-97be-11eb-9c81-a3bc43df8dc3.png)

<그림 8> Lane Marking

② Road Object Detection

BDD100K는 버스, 신호등, 교통 표지판, 자전거, 트럭, 오토바이, 자동차, 기차, 사람, 라이더를 위한 10만 개의 키프레임에 2D Bounding Box와 라벨이 달려있다. 이전의 다른 dataset에 비해 보행자에 대한 정보가 많기 때문에 보행자 검출 및 회피를 위하여 BDD100K를 활용할 수 있다.

③ Driveable Area

차선이 명확하지 않은 경우도 있기 때문에 차선만으로 차량의 주행 여부를 판단하기는 불충분하다. 이에, 운전 가능 지역을 구분할 수 있는 것이 좋다, 운전 가능 지역은 directly driveable area와 alternatively drive area로 구분할 수 있다. Directly driveable area는 현재 주행 중인 경로로 적색으로 표시하고, alternatively drive area는 대체 주행이 가능한 경로로 청색으로 표시한다. 

![image](https://user-images.githubusercontent.com/81551992/113827274-1497b880-97be-11eb-9197-8653c1df7b26.png)

<그림 9> Driveable Area
④ Semantic Instance Segmentation

전체 dataset으로부터 무작위 샘플링 된 1만 개의 각각의 비디오로부터 얻어진 이미지에 대하여 픽셀 레벨의 세분화(fine-grained)된 라벨을 제공한다. 각각의 픽셀에는 이미지의 객체 라벨의 instance 번호를 나타내는 식별자와 라벨이 주어진다. 많은 클래스들은 instance로 나눠질 수 없기 때문에 클래스 라벨의 작은 부분 집합에만 instance 식별자가 할당된다. 전체 라벨 셋은 각 이미지의 라벨 픽셀 수를 최대화할 뿐만 아니라, 도로 환경에서의 객체의 다양성을 포착하는 선택된 40개의 객체 클래스로 구성된다.

![image](https://user-images.githubusercontent.com/81551992/113827305-1eb9b700-97be-11eb-91fe-c8cd9decbf46.png)

<그림 10> Instance Segmentation

#### 3) nuScenes ####

nuScenes dataset은 Motional에서 만든 대규모 자율주행 dataset이다. 우선, Motianal은 무인 차량이 안전하고, 믿을 수 있고, 현실성 있도록 다양한 개발과 기술을 제공하고 있는 회사이다. 이들은 dataset을 대중적으로 공개하여 Computer Vision과 자율주행 연구를 지원하고 있다.

![image](https://user-images.githubusercontent.com/81551992/113827349-2b3e0f80-97be-11eb-985c-daee17977583.png)

<그림 11> nuScenes

nuScenes dataset은 주행 및 교통 상황이 까다로운 Boston과 Singapore 두 도시에서 수집된 1000개의 주행 장면, 140만개의 Camera image, Radar sweeps, 39만개의 LiDAR sweeps, 140만개의 object bounding box로 구성되어 있다. 20초 길이의 각 장면은 다양한 교통 상황, 주행 전략, 돌발 상황을 담고 있다. 이처럼 다양하고 복잡한 nuScenes는 장면에 존재하는 수십 개의 객체들이 있는 도시 환경에서 안전한 자율주행을 가능하게 하는 기술 개발에 도움을 줄 것이다. 또한, 서로 다른 나라에서 데이터를 수집하였기 때문에 특정 환경(다른 위치, 날씨, 교통 표지판 등)이 아닌 Computer Vision 및 자율주행 연구에서의 일반화를 가능하게 한다. Object detection, tracking과 같은 Computer Vision task에 이용하기 위하여, 전체 dataset에 3D Bounding box와 함께 23개의 object class의 라벨을 달았다. 추가적으로, active, pose같은 특성들도 라벨을 달았다. 이러한 nuScenes datset은 KITTI dataset으로부터 영감을 받았다. KITTI dataset과 비교했을 때, nuScenes는 KITTI보다 7배 더 많은 object 주석을 포함하고 있다. 또한, BDD/Cityscapes/Apolloscapes와 같은 카메라 기반의 dataset과는 달리 nuScenes는 모든 센서를 다루는 것을 목표로 한다. 이에, 1개의 LiDAR, 5개의 RADAR, 6개의 Camea, IMU, GPS센서를 사용하였다.
2020년 6월, Motional은 nuScenes-lidarseg와 nuImages가 추가된 nuScenes의 확장판을 출시하였다. nuScenes-lidarseg는 32개의 가능한 semantic labels을 이용하여 nuScenes의 keyframe으로부터 얻어진 각각의 lidar point에 라벨링을 하였다. 그 결과, nuScenes-lidarseg은 4만개의 point cloud와 1000개의 장면에 걸쳐 14억개의 라벨링 point를 포함하고 있다. 구체적으로, nuScenes-lidarseg은 foreground class(보행자, 차량, 사이클리스트 등)과 background class(도로 표면, 자연, 빌딩 등)을 포함하고 있다. 또한, nuImages는 93000개의 2d annotated image를 이용하여 이전 nuScenes의 dataset을 보완하였다.

![image](https://user-images.githubusercontent.com/81551992/113827386-35f8a480-97be-11eb-961e-b844b87d1b48.png)

<그림 12> nuScenes lidarseg

nuScenes를 이용한 task는 detection&tracking, prediction, lidar segmentation 크게 네 가지로 나눌 수 있다. 각 task에 대해 간단히 설명하면 다음과 같다.

① Detection & tracking

nuScenes을 이용한 3D obeject detection task이다. 이 task의 목표는 각 set의 특성 및 속도 vector의 추정뿐만 아니라 서로 다른 10개의 카테고리에 3D bounding box를 두는 것이다. nuScenes의 23개의 class 중, 비슷한 것은 묶고 거의 없는 class에 대해서는 버림으로서 10개의 클래스를 이용한다. 예를 들어, 오토바이/트럭/버스/자동차는 vehicle이라는 하나의 클래스로 묶는 것이다. Detection 결과들은 2Hz keyfame 단위로 평가되며 train/validation/test set 모두 json 형태로 저장된다. Task의 평가 metrics로는 mAP, TP 성능 지표를 활용한다. Tracking은 detection에서 자연스레 이어지는 과정이다. 잘 알려진 detection 알고리즘으로부터 얻어진 object를 시간 따라 tracking하는 것이다. camera, lidar, radar센서를 이용하여 3D multi object tracking을 진행하고, online tracking을 실시한다. 즉, 미래의 센서 data는 활용하지 못하고 과거와 현재 데이터만을 이용하여 진행하는 task이다. 다양한 평가 metrics가 있지만, challenge의 우승자는 ‘AMOTA’를 사용하였다. ‘AMOTA’는 average multi object tracking accuracy의 약자로 각각 다른 recall thresholds에서 false positives, missed targets, identity switches 세 가지 error를 조합한 ‘MOTA’를 평균 내는 방식이다. 

② Prediction

nuScenes prediction task의 목표는 nuScenes dataset에서 object의 미래 경로를 예측하는 것이다. Task의 평가 metrics로는 세 가지를 사용한다. 첫째, 예측된 경로와 ground_truth 상의 point들 사이의 L2(유클리디안) 거리를 평균 내는 ‘minADE_k’이다. 둘째, 예측된 경로와 ground_truth의 최종 목표 지점 사이의 거리를 이용하는 ‘minFDE_k'이다. 마지막으로, 예측된 경로와 grount_truth의 L2 거리의 최댓값이 2m보다 크다면 예측에 실패했다고 정의한다.

③ Lidar segmentation

nuScenes lidar segmentation은 앞서 언급하였던 nuScenes-lidarseg를 이용한다. 이 task의 목표는 point clouds set의 모든 point에 대하여 카테고리를 예측하는 것이다. 카테고리는 nuScnes-lidarseg의 32개 sematic label(class)중에서 detection 할 때와 유사한 방식으로 구분하여 10개의 foreground class와 6개의 background class로 총 16개로 구성되어 있다. 결과는 point cloud의 point label을 담고 있는 bin형식의 파일과 json 형식의 파일로 구성된다. Task의 평가 metrics로는 mean intersection-over-union(mIOU)를 사용한다. 

이처럼, 다양한 센서 데이터와 task를 가진 nuScenes dataset은 꾸준히 이용되고 있다. 2019년 3월, nuScenes 공개 이후 수많은 사람이 이를 이용하여 기술 개발을 진행해왔으며 250개 이상의 논문에서 dataset을 이용하였다. 그 후 nuScenes-lidarseg을 공개하여 부족하다고 평가 받던 LiDAR 부분을 보강하면서 높은 레벨의 자율주행 상용화를 앞당기는데 큰 역할을 할 것으로 예상된다. 
