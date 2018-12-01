import numpy as np
from sklearn import preprocessing

data = np.array([[3, -1.5 , 2, -5.4], [0, 4, -0.3, 2.1], [1, 3.3, -1.9, -4.3]])

# 标记编码
# 定义一个标记编码器
label_encoder = preprocessing.LabelEncoder()
# 创建一些标记
input_classes = ['audi', 'ford', 'audi', 'toyota', 'ford', 'bwm']
# 标记编码
label_encoder.fit(input_classes)
print('\nClass maping:')
for i, item in enumerate(label_encoder.classes_):
    print(item,'-->',i)

labels = ['toyota', 'ford', 'audi']
encoded_labels = label_encoder.transform(labels)
print('\nLables = ',labels)
print('Encoded labels = ',list(encoded_labels))

encoded_labels = [2, 1, 0, 3, 1]
decoded_labels = label_encoder.inverse_transform(encoded_labels)
print('\nEncoded labels = ',encoded_labels)
print('Decoded labels = ',list(decoded_labels))

# one-hot
encoder = preprocessing.OneHotEncoder()
encoder.fit([[0, 2, 1, 12], [1, 3, 5, 3],[2, 3, 2, 12],[1, 2, 4, 3]])
encoder_vector = encoder.transform([[2, 3, 5 ,3]]).toarray()
print('\nEncoder vector = ',encoder_vector)

# 二值化
data_binaried = preprocessing.Binarizer(threshold=1.4).transform(data)
print('\n Binarized data = ', data_binaried)

# 均值移除
data_standardized = preprocessing.scale(data)
print('mean = ',data_standardized.mean(axis=0))
print('Std deviation = ', data_standardized.std(axis=0))

# 范围缩放
data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled = data_scaler.fit_transform(data)
print('\nMin max scaled data = ',data_scaled)

# 归一化
data_normalized = preprocessing.normalize(data,norm='l1')
print('\nL1 normalized data = ',data_normalized)