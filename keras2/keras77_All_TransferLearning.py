from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1

model = ResNet152V2()
model.trainbale = False

model.summary()
print(len(model.weights))
print(len(model.trainable_weights))

# 모델별로 파라미터수 정리!!

# VGG16
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# 32
# 32

# VGG19
# Total params: 143,667,240
# Trainable params: 143,667,240
# Non-trainable params: 0
# 38
# 38

# Xception
# Total params: 22,910,480
# Trainable params: 22,855,952
# Non-trainable params: 54,528
# 236
# 156

# ResNet101
# Total params: 44,707,176
# Trainable params: 44,601,832
# Non-trainable params: 105,344
# 626
# 418

# ResNet101V2
# Total params: 44,675,560
# Trainable params: 44,577,896
# Non-trainable params: 97,664
# 544
# 344

# ResNet152V2
# Total params: 60,380,648
# Trainable params: 60,236,904
# Non-trainable params: 143,744
# 816
# 514