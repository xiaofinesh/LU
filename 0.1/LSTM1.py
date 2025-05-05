import numpy as np
import pandas as pd

# 加载数据集
# 将 '实时舱报表数据-20250328.csv' 替换为实际的数据集文件
data = pd.read_csv('实时舱报表数据-20250328.csv')

# 打印预处理前的数据集
print("原始数据集：")
print(data.head())
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 加载数据集
# 将 'data.csv' 替换为实际的数据集文件
data = pd.read_csv('data.csv')

# 数据预处理
# 假设数据集包含 'features' 和 'label' 列
features = data.iloc[:, :-1].values  # 提取特征列
labels = data.iloc[:, -1].values    # 提取标签列

# 特征归一化
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

# 调整数据形状以适配 LSTM 输入 (样本数, 时间步长, 特征数)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# 构建 LSTM 模型
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),  # LSTM 层，50 个单元
    Dense(1, activation='sigmoid')  # 输出层，假设是二分类问题
])

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f"测试集准确率: {accuracy:.2f}")

# 保存模型
model.save('lstm_leak_detection_model.h5')