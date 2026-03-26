import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    cohen_kappa_score
)

# 读取数据
train_path = "./dataset/train_data.csv"
test_path  = "./dataset/test_data3.csv"
train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

# 特征工程
def feature_engineering(df):
    df = df.copy()

    # 比值特征
    df["GR_RD"] = df["GR"] / (df["RD"] + 1e-6)
    df["K_GR"] = df["K"] / (df["GR"] + 1e-6)
    df["AC_GR"] = df["AC"] / (df["GR"] + 1)
    df["RD_GR"] = df["RD"] / (df["GR"] + 1)
    df["K_AC"] = df["K"] / (df["AC"] + 1)
    df["CAL_RD"] = df["CAL"] / (df["RD"] + 1)
    # 乘积特征
    df["AC_CAL"] = df["AC"] * df["CAL"]
    df["GR_SP"] = df["GR"] * df["SP"]
    df["RD_SP"] = df["RD"] * df["SP"]
    df["AC_SP"] = df["AC"] * df["SP"]
    # 对数特征
    df["log_RD"] = np.log1p(df["RD"])
    df["log_AC"] = np.log1p(df["AC"])
    df["log_GR"] = np.log1p(df["GR"])
    df["log_K"] = np.log1p(df["K"])
    df["log_CAL"] = np.log1p(df["CAL"])
    # 平方特征
    df["GR_sq"] = df["GR"] ** 2
    df["RD_sq"] = df["RD"] ** 2
    df["AC_sq"] = df["AC"] ** 2

    return df

# 应用特征工程
train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

# 构建特征与标签
target = "Core Lithology"
features = [col for col in train_df.columns if col != target]

# 选定特征类型
basic_features = ["AC", "CAL", "GR", "K", "RD", "SP"] # 原始特征
ratio_features = ["GR_RD", "K_GR", "AC_GR", "RD_GR", "K_AC", "CAL_RD"] # 比值特征
product_features = ["AC_CAL", "GR_SP", "RD_SP", "AC_SP"] # 乘积特征
log_features = ["log_GR", "log_RD", "log_AC", "log_K", "log_CAL"] # 对数特征
square_features = ["GR_sq", "RD_sq", "AC_sq"] # 平方特征

# 选择制定特征，并赋值给specific_features
specific_features = basic_features+log_features+ratio_features+product_features+square_features
# specific_features = basic_features+log_features
le = LabelEncoder()
X_train = train_df[specific_features].values # 获得训练集特征列
y_train = le.fit_transform(train_df["Core Lithology"].values) # 获得训练集标签列

X_test = test_df[specific_features].values # 获得测试集特征列
y_test = le.transform(test_df["Core Lithology"].values) # 获得测试集标签列

# 特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

xgb_flat = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=7,
        n_estimators=500,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42
    )
xgb_flat.fit(X_train, y_train)

# 对测试集进行预测
proba = xgb_flat.predict_proba(X_test)
# y_pred_raw = np.argmax(proba, axis=1)
# y_pred = y_pred_raw
# 地质约束函数
def apply_geological_constraint(
    proba,
    clastic_set,
    carbonate_set,
    carbonate_threshold=0.6
):
    y_pred = np.zeros(len(proba), dtype=int)

    for i in range(len(proba)):
        pred_class = np.argmax(proba[i])

        # 如果预测为碳酸盐岩
        if pred_class in carbonate_set:
            carbonate_prob = proba[i, carbonate_set].sum()

            # 不满足地质可信度
            if carbonate_prob < carbonate_threshold:
                # 强制在碎屑岩中重新选择
                y_pred[i] = clastic_set[
                    np.argmax(proba[i, clastic_set])
                ]
            else:
                y_pred[i] = pred_class
        else:
            y_pred[i] = pred_class

    return y_pred

# 地质族划分（基于岩性成因）
clastic_set = [0, 1, 2, 3, 6]   # 碎屑岩
carbonate_set = [4, 5]         # 碳酸盐岩

# 应用地质约束
y_pred = apply_geological_constraint(
    proba,
    clastic_set,
    carbonate_set,
    carbonate_threshold=0.8
)
print("Macro-F1   :", f1_score(y_test, y_pred, average="macro"))
print("Weighted-F1:", f1_score(y_test, y_pred, average="weighted"))
print("Cohen Kappa   :", cohen_kappa_score(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))
print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
