import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# 设置随机种子以便复现
np.random.seed(42)
tf.random.set_seed(42)

# 1. 加载数据
df = pd.read_csv("Processed_Data1.csv", encoding='gbk')
X = df.drop(columns=["时间", "TRUE/FALSE"])
y = df["TRUE/FALSE"].astype(int)

# 2. 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_array = y.values

# 3. 设置 KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
acc_scores = []
fold_num = 1

# 4. 交叉验证
for train_idx, val_idx in kf.split(X_scaled):
    print(f"\n🔁 Fold {fold_num}:")

    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y_array[train_idx], y_array[val_idx]

    # 模型构建
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_dim=X_train.shape[1]),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 训练模型（记录训练历史）
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=50, batch_size=32,
                        verbose=0)

    # 模型评估
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    acc_scores.append(acc)
    print(f"Fold {fold_num} Accuracy: {acc:.4f}")

    # 绘制训练过程图
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'Fold {fold_num} - Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'fold{fold_num}_accuracy_curve.png')
    plt.close()

    # 预测 + 混淆矩阵
    y_pred = model.predict(X_val)
    y_pred_label = (y_pred > 0.5).astype(int)

    cm = confusion_matrix(y_val, y_pred_label)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Fold {fold_num} - Confusion Matrix')
    plt.savefig(f'fold{fold_num}_confusion_matrix.png')
    plt.close()

    # 最后一折保存模型
    if fold_num == 5:
        model.save("leakage_detection_model_fold5.h5")
        print("✅ 模型已保存为 'leakage_detection_model_fold5.h5'")

    fold_num += 1

# 5. 输出总评估结果
print(f"\n📊 5折交叉验证 - 平均准确率: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
