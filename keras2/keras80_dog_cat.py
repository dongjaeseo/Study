# 이미지는 
# data/image/vgg/ 에 4개를 넣으시오
# 개 고양이 라이언 슈트
# 욜케 넣어놓을것
# 파일명 :
# dog1.jpg cat1.jpg lion1.jpg suit1.jpg

from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array # 이미지를 넘파이 배열로 바꿔준다!
import numpy as np

img_dog = load_img('../data/image/vgg/dog1.jpg', target_size = (224,224)) # 사이즈도 지정가능
img_cat = load_img('../data/image/vgg/cat1.jpg', target_size = (224,224))
img_ryan = load_img('../data/image/vgg/ryan1.jpg', target_size = (224,224))
img_suit = load_img('../data/image/vgg/suit1.jpg', target_size = (224,224))

# plt.imshow(img_cat) # 이미지 잘 불러왔나 확인!
# plt.show()

# print(img_suit) # 로드 이미지 했을땐 케라스로 임포트 해서 케라스 형식이다! 
# <PIL.Image.Image image mode=RGB size=224x224 at 0x1CD822F85B0>
# 그래서 image_to_array 를 사용해 배열로 바꿔준다

arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_ryan = img_to_array(img_ryan)
arr_suit = img_to_array(img_suit)

# print(arr_dog) # [254. 254. 255.]]]
# print(type(arr_suit)) <class 'numpy.ndarray'>
# print(arr_dog.shape) (224, 224, 3)

# 이렇게 이미지를 불러오면 RGB 형태이다
# 근데 VGG16 에 넣을때는 BGR 형태로 넣어야한다
# RGB -> BGR
from tensorflow.keras.applications.vgg16 import preprocess_input
# 알아서 vgg16 에 맞춰 전처리를 해준다
arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_ryan = preprocess_input(arr_ryan)
arr_suit = preprocess_input(arr_suit)

# print(arr_dog)
# print(arr_dog.shape) (224, 224, 3)
# 쉐이프는 동일, R 과 G 의 위치만 바뀌었다

arr_input = np.stack([arr_dog, arr_cat, arr_ryan, arr_suit]) # np.stack 은 배열을 합쳐주는 역할
# print(arr_input.shape) (4, 224, 224, 3) ## 순서대로 dog, cat, ryan, suit 가 합쳐진 배열


#2. 모델구성   ## 훈련 시킬게 아니라 모델을 그대로 쓸것이다
model = VGG16()
results = model.predict(arr_input)

# print(results)
# print('results.shape : ', results.shape)
'''
[[1.4992932e-09 4.9452953e-10 5.1459503e-11 ... 4.8396742e-10     
  1.6936048e-07 1.1220930e-06]
 [4.3583753e-07 1.2146568e-06 4.4076779e-07 ... 5.3686909e-07
  3.3494583e-05 3.7730621e-05]
 [9.2846369e-07 4.9175487e-06 1.0787576e-06 ... 8.0677171e-07
  3.9812530e-05 3.4896409e-04]
 [3.1493435e-08 9.4961639e-10 2.1516704e-09 ... 1.1728181e-10
  2.4805676e-08 2.1393834e-07]]
results.shape :  (4, 1000)    << 1000 은 imagenet 에서 분류할수 있는 카테고리 수
'''

# 이걸 확인 어떻게 할까? 확인하는것 또한 제공해주겠지
# 이미지 결과 확인
from tensorflow.keras.applications.vgg16 import decode_predictions # 해독한다! > 예측한걸 해석한다

decode_results = decode_predictions(results)
print('===========================================')
print('results[0] : ', decode_results[0])
print('===========================================')
print('results[1] : ', decode_results[1])
print('===========================================')
print('results[2] : ', decode_results[2])
print('===========================================')
print('results[3] : ', decode_results[3])
print('===========================================')

'''
===========================================
results[0] :  [('n02099601', 'golden_retriever', 0.97542906), ('n02099712', 'Labrador_retriever', 0.017516011), ('n02104029', 'kuvasz', 0.0017046107), ('n02101556', 'clumber', 0.0012573326), ('n02102480', 'Sussex_spaniel', 0.0012475004)]
===========================================
results[1] :  [('n02124075', 'Egyptian_cat', 0.43415174), ('n02123045', 'tabby', 0.4329717), ('n02123159', 'tiger_cat', 0.097306974), ('n02127052', 'lynx', 0.011449572), ('n02123394', 'Persian_cat', 0.0026808705)]
===========================================
results[2] :  [('n04548280', 'wall_clock', 0.21537575), ('n03532672', 'hook', 0.14212109), ('n04579432', 'whistle', 0.069210045), ('n02951585', 'can_opener', 0.065023765), 
('n03109150', 'corkscrew', 0.059872627)]
===========================================
results[3] :  [('n04350905', 'suit', 0.8362251), ('n04591157', 'Windsor_tie', 0.1381057), ('n02883205', 'bow_tie', 0.012294901), ('n03680355', 'Loafer', 0.005307802), ('n10148035', 'groom', 0.0044345083)]
===========================================
'''
# 강아지 - 골든 리트리버일 확률이 0.9754 >> 97.54 퍼센트!
# 고양이 - Egyptian cat 일 확률이 43.41 퍼센트!
# 라이언은 이미지넷에 있는 1000개 카테고리에 들어있지 않기때문에 측정이 정확치 않다!
# 라이언 - 벽시계일 확률이 21.53 퍼센트!
# 슈트 - 슈트일 확률이 83.62 퍼센트!

# 쭝요한껀 너녜뜰이 쏜으로 뺶날하는껐뽀따 이꼐 짤또ㅒ
# 쒸프트 뭐야 