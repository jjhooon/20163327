### 3. 2의 정리한 코드 중 하나 실행해서 결과 확인 ###

#### 1) 코드 실행 ####

2번에서 언급하였던 YOLOv3 모델을 가져와 Linux 환경에서 구현해보았다. 가지고 있는 컴퓨터가 Window 기반의 컴퓨터이기에 듀얼 부팅을 하여 Ubuntu 환경에서 모델을 돌렸다. Input으로 넣을 이미지는 운전하면서 직접 찍었던 두 장의 사진이며, 전방의 object를 얼마나 정확하게 인지하는지 확인할 수 있었다. 구현 코드 및 환경설정은 다음과 같다.

YOLOv3을 모델을 가져와 구현을 하기 전에 먼저 GPU, Opencv, cudnn 설치가 필요하다. 하지만, Intel의 내장 graphic 카드를 사용하기에 별도의 GPU 설치 및 cudnn설치를 진행하지 않았고 기존에 Ubuntu 환경에서의 Opencv를 사용한 경험이 있어 라이브러리가 이미 설치되어 있는 상태였다. Ubuntu 터미널 창을 열어 입력한 커맨드 위주의 설명은 다음과 같다.

![image](https://user-images.githubusercontent.com/81551992/113757951-3a857480-974e-11eb-9449-07632074e0a6.png)

우선, 'cd opencv/opencv-3.2.0/build'를 입력하여 directory를 변경한다. 그 후 , build 위치에다 git을 설치하기 위하여 'sudo apt install git'을 입력하고, 잘 설치된 것을 확인할 수 있다. git의 설치가 완료되면 YOLOv3의 저자인 Joseph Redmon의 github에서 darknet을 clone('git clone')한다. 이후의 과정을 진행하기 위하여 'cd darknet'을 입력하여 방금 설치했던 darknet으로 directory를 이동한다. 앞서 GPU, Opencv, cudnn의 설치를 언급하였는데 이것들에 대한 환경설정을 위하여 'vi Makefile‘을 입력하였다. ’vi Makefile‘을 입력하였을 때 열리는 창은 아래와 같다.

![image](https://user-images.githubusercontent.com/81551992/113757998-46713680-974e-11eb-8dd1-859a74bd8197.png)

위에서부터 세 줄을 보면 초기에는 GPU, Opencv, cudnn의 값이 모두 0으로 설정되어 있다. 이 때, Opencv만 설치하여 사용하기 때문에 값을 1로 바꿔주고, 나머지의 값은 0으로 setting후 shift+z+z를 눌러 값을 저장하고 다시 터미널 창으로 돌아간다. 그 후 'make' 명령을 통하여 컴파일을 실행한다.

![image](https://user-images.githubusercontent.com/81551992/113758025-4e30db00-974e-11eb-832a-8cc436d86bf5.png)

위 사진의 첫째 줄은 YOLOv3의 가중치를 받는 명령이다. ‘wget yolov3.weight가 있는 링크‘를 입력하여 darknet 폴더에 yolov3.weight를 다운받는다. 다운로드가 완료되면 input image(person_detect.jpg)와 다운 받은 yolov3.weight, detect cfg 파일을 이용하여 학습을 진행한다. 실행시키면 convolution neural network와 filter size가 모두 기록되면서 detection이 진행되고 있는 것을 확인할 수 있다.

![image](https://user-images.githubusercontent.com/81551992/113758044-5721ac80-974e-11eb-97aa-ea17ee1909d2.png)

Detect를 하는데 31.45초의 시간이 걸렸고, 여러 대의 자동차와 사람이 검출된 것을 알 수 있다. 

![image](https://user-images.githubusercontent.com/81551992/113758067-5e48ba80-974e-11eb-8eed-fde840ddd1b2.png)

앞서 언급하였던 방식과 동일하게 car_detect.jpg 파일에 대해서 detection을 진행한다. 그 결과, 신호등, 트럭 한 대, 여러 대의 자동차가 검출된 것을 확인할 수 있다.

#### 2) 코드 실행 결과 ####

![image](https://user-images.githubusercontent.com/81551992/113758120-73254e00-974e-11eb-8d6b-fcb1c40cf8d3.png)![image](https://user-images.githubusercontent.com/81551992/113758134-77ea0200-974e-11eb-8e9e-c6dc18abea68.png)

좌측의 사진은 person_detect 사진을 input으로 받아 object detection을 진행한 결과이다. 횡단보도를 건너고 있는 보행자가 모두 검출되고 있는 것을 확인할 수 있고, 멀리 있는 자동차들까지도 검출되었다. 하지만, 전방의 신호등이 검출되지 않은 점에서는 아쉬운 결과이다.
우측의 사진은 car_detect 사진을 input으로 받아 object detection을 진행한 결과이다. 전방의 차량과 트럭뿐만 아니라, 반대 차선의 차량과 신호등의 검출까지 성공한 것을 확인할 수 있다. 하지만, 우측 끝에 신호 대기 중인 차량에 대해서는 검출을 하지 못하였다.
