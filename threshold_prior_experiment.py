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
test_path  = "./dataset/test_data1.csv"
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

le = LabelEncoder()
X_train = train_df[features].values # 获得训练集特征列
y_train = le.fit_transform(train_df["Core Lithology"].values) # 获得训练集标签列

X_test = test_df[features].values # 获得测试集特征列
y_test = le.transform(test_df["Core Lithology"].values) # 获得测试集标签列

# 特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ----------- 定义层次标签映射 -----------
# 碎屑岩（5 类）
clastic_global = [0, 1, 2, 3, 6]
clastic_map = {g: i for i, g in enumerate(clastic_global)}
clastic_inv_map = {i: g for g, i in clastic_map.items()}

# 碳酸盐岩（2 类）
carbonate_global = [4, 5]  # 碳酸盐族
carbonate_map = {g: i for i, g in enumerate(carbonate_global)}
carbonate_inv_map = {i: g for g, i in carbonate_map.items()}

# -----------第一层，岩性族，做2分类（0：碎屑岩 1：碳酸盐岩）------------
carbonate_classes = [4, 5]  # 白云石、石灰岩
# 找到训练集岩性列哪些为碳酸盐岩，哪些不是，并将是碳酸盐岩族的岩性设置为1，不是的设置为0
y_family_train = np.isin(y_train, carbonate_classes).astype(int)
xgb_family = xgb.XGBClassifier(
    objective="binary:logistic",
    n_estimators=700,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=1,
    eval_metric="logloss",
    random_state=42
)

xgb_family.fit(X_train, y_family_train)

# -----------训练碎屑岩子模型------------
# 找到训练集中所有的碎屑岩和非碎屑岩
mask_clastic = np.isin(y_train, clastic_global)
# 把所有的碎屑岩从训练集中取出来
X_clastic = X_train[mask_clastic]
# 对五类碎屑岩进行重新编码
y_clastic = np.array([clastic_map[y] for y in y_train[mask_clastic]])

xgb_clastic = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=len(clastic_global),
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)

xgb_clastic.fit(X_clastic, y_clastic)


# -----------训练碳酸盐岩子模型------------
# 找到训练集中所有的碳酸盐岩和非碳酸盐岩
mask_carbonate = np.isin(y_train, carbonate_global)
# 把所有的碳酸盐岩从训练集中取出来
X_carbonate = X_train[mask_carbonate]
# 对两类碳酸盐岩进行重新编码
y_carbonate = np.array([carbonate_map[y] for y in y_train[mask_carbonate]])

xgb_carbonate = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=2,
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)

xgb_carbonate.fit(X_carbonate, y_carbonate)


# -----------预测融合------------
# 获取二分类（碳酸/碎屑）中为1的数据（碳酸）
family_prob = xgb_family.predict_proba(X_test)[:, 1]
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    family_pred = family_prob > t
    y_pred = np.zeros(len(X_test), dtype=int)

    idx0 = np.where(family_pred == 0)[0] # 碎屑岩
    idx1 = np.where(family_pred == 1)[0] # 碳酸盐岩

    if len(idx0) > 0: # 碎屑岩细分
        y0 = np.argmax(xgb_clastic.predict_proba(X_test[idx0]), axis=1)
        y_pred[idx0] = [clastic_inv_map[int(y)] for y in y0]

    if len(idx1) > 0: # 碳酸盐岩细分
        y1 = np.argmax(xgb_carbonate.predict_proba(X_test[idx1]), axis=1)
        y_pred[idx1] = [carbonate_inv_map[int(y)] for y in y1]

    # -----------评估------------
    print("阈值先验结果",t)
    print("Macro-F1      :", f1_score(y_test, y_pred, average="macro"))
    print("Weighted-F1   :", f1_score(y_test, y_pred, average="weighted"))
    print("Cohen Kappa   :", cohen_kappa_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))
