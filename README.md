# motion_detection
using decision tree for motion detection

## model 구성
sklearn library를 활용하여 가속도 센서의 데이터를 바탕으로 의사결정트리를 학습

data pre-processing은 다음 논문을 참고함

Yang, J. (2009, October). Toward physical activity diary: motion recognition using simple acceleration features with mobile phones. In Proceedings of the 1st international workshop on Interactive multimedia for consumer electronics (pp. 1-10).

## data 구성
Sensor Kinetics Pro를 이용하여 falling/non-falling 경우의 가속도 데이터베이스 구성
