# 2019-2 Intelligent Robot Crash Lab
- 기부로봇 만들기 프로젝트

## 주요기능
- 사람을 발견할 시 일정거리(약 1m) 이내로 접근
- Naeyo의 화면이 기본(무표정)에서 송금 가능한 QR코드로 전환되며 기부를 유도
- 송금 완료 후 사용자가 Tony의 머리에 있는 카메라를 향해 엄지를 보여주면 Tony가 같이 "엄지척"을 보여줌. 동시에 Naeyo의 표정이 웃는 얼굴로 바뀜

## 시연영상
![feedback](https://user-images.githubusercontent.com/42140395/71198615-81402880-22d7-11ea-9502-249ffb819ea8.gif)
![targetting](https://user-images.githubusercontent.com/42140395/71198561-666db400-22d7-11ea-8ecb-fc430a6bbb65.gif)

## Requirements
- TX2 ROS Package 환경에 의해 아래의 개발환경을 맞추어야 원활하게 사용 가능합니다
- python 2.7
- opencv 3.4.0.14
- Keras >= 2.0.2
- Theano 0.9.0
- h5py 2.5.0

## rqt_graph
<img width="1289" alt="tony-and-naeyo-rqt" src="https://user-images.githubusercontent.com/42140395/71197846-d9762b00-22d5-11ea-805a-b9f747e6b6db.png">


## Screenshot
![tony-and-naeyo](https://user-images.githubusercontent.com/42140395/71197994-307c0000-22d6-11ea-9f9f-f9657936d130.jpeg)
![tony-and-naeyo2](https://user-images.githubusercontent.com/42140395/71198094-6325f880-22d6-11ea-8b5d-424a6b16bd64.jpeg)
![all](https://user-images.githubusercontent.com/42140395/71198156-8d77b600-22d6-11ea-9df8-c5f5a2226ed1.jpeg)


## Role
|  <center>Name</center> |  <center>Main Role</center> | 
|:--------|:--------:|
|**유호연** | <center>Team Leader, 구동모터제어, Infra(Real Sense Cam, Network)</center> |
|**배종학** | <center>OpenCV, Thumbs-up Detection(Keras)</center> |
|**김인영** | <center>ROS Software Architecture Design, Human Detection(Darknet) </center> |
|**정은수** | <center>Robot Hardware 설계, TONY Gesture</center> |
