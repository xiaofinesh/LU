import pandas as pd
import tensorflow as tf
from keras import layers, models
# 从 sklearn 库中导入用于划分训练集和测试集的函数
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 1. 加载数据集
df = pd.read_csv("Processed_Data1.csv", encoding='gbk')  # 使用 gbk 编码以支持中文

# 2. 分离特征和标签
X = df.drop(columns=["时间", "TRUE/FALSE"])  # 假设“是否漏液”是标签列，"时间"列不用作为特征
y = df["TRUE/FALSE"].astype(int)  # 将“是否漏液”列转换为整数（0/1）

# 3. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. 构建 MLP 模型
model = models.Sequential([
    layers.Dense(64, activation='relu', input_dim=X_train.shape[1]),  # 第一层
    layers.Dense(32, activation='relu'),  # 第二层
    layers.Dense(1, activation='sigmoid')  # 输出层，二分类问题使用 sigmoid 激活函数
])

# 6. 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 7. 训练模型
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 8. 评估模型
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# 9. 可选：使用模型进行预测
predictions = model.predict(X_test)

