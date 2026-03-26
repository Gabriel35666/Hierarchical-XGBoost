import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    cohen_kappa_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import time



# 读取数据
train_path = "./dataset/train_data.csv"
test_path  = "./dataset/test_data2.csv"
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
# specific_features = basic_features
le = LabelEncoder()
X_train = train_df[specific_features].values # 获得训练集特征列
y_train = le.fit_transform(train_df["Core Lithology"].values) # 获得训练集标签列

X_test = test_df[specific_features].values # 获得测试集特征列
y_test = le.transform(test_df["Core Lithology"].values) # 获得测试集标签列

# 特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

time_records = []
# 定义一个统一的训练与评估函数（只输出关键指标 + 详细报告）
def train_and_evaluate(model, name, X_train, y_train, X_test, y_test):
    print(f"\n=== Training {name} ===")
    # -------- Training Time --------
    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_train

    # -------- Inference Time --------
    start_test = time.perf_counter()
    y_pred_model = model.predict(X_test)
    infer_time = time.perf_counter() - start_test

    # 单位样本预测时间
    avg_pred_time_per_sample=infer_time / len(X_test)

    macro = f1_score(y_test, y_pred_model, average="macro")
    weighted = f1_score(y_test, y_pred_model, average="weighted")
    kappa = cohen_kappa_score(y_test, y_pred_model)

    print(f"\n{name} Results")
    print("Macro-F1      :", round(macro, 4))
    # print("Weighted-F1   :", round(weighted, 4))
    # print("Cohen Kappa   :", round(kappa, 4))
    # print("\nClassification Report:\n")
    # print(classification_report(y_test, y_pred_model, digits=4))
    # print("Confusion Matrix:\n")
    # print(confusion_matrix(y_test, y_pred_model))
    print("Train Time (s):", round(train_time, 4))
    print("Infer Time (s):", round(infer_time, 4))
    print("单位样本预测时间 (s):", avg_pred_time_per_sample)

    # # 保存结果
    # df_test = pd.read_csv("./dataset/test_data3.csv")
    # df_predict = df_test.copy()
    # df_predict['Predicted_Lithology'] = y_pred_model+1
    # filename = f"./results/{name.replace(' ', '_').replace('(', '').replace(')', '')}_井3预测结果.csv"
    # df_predict.to_csv(filename)
    # print(f"预测结果已保存至：{filename}")

    return y_pred_model

# 计算类别数
num_classes = len(np.unique(y_train))
# 1. Flat XGBoost（单层多分类，直接对比你分层结构的最常见baseline）
flat_xgb = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=num_classes,
    n_estimators=500,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42,
)
train_and_evaluate(flat_xgb, "Flat XGBoost", X_train, y_train, X_test, y_test)

# # 2. Random Forest（测井岩性识别中最经典、最常用的模型之一）
# rf = RandomForestClassifier(
#     n_estimators=500,
#     max_depth=None,
#     min_samples_split=2,
#     min_samples_leaf=1,
#     random_state=42,
#     n_jobs=-1
# )
# train_and_evaluate(rf, "Random Forest", X_train, y_train, X_test, y_test)
#
# # 3. Support Vector Machine（RBF核，早年测井岩性识别主流模型）
# # 注意：如果样本量较大（>1万），SVM训练会比较慢，可注释掉或改用LinearSVC
# svc = SVC(
#     kernel="rbf",
#     C=5.0,  # 经验值，对测井数据通常有效
#     gamma="scale",
#     random_state=42
# )
# train_and_evaluate(svc, "SVM (RBF)", X_train, y_train, X_test, y_test)
#
# # 4. Multi-Layer Perceptron（神经网络，近年来在测井岩性识别中越来越常见）
# mlp = MLPClassifier(
#     hidden_layer_sizes=(256, 128, 64),
#     activation="relu",
#     solver="adam",
#     alpha=0.0001,
#     learning_rate="adaptive",
#     max_iter=1000,
#     early_stopping=True,
#     validation_fraction=0.1,
#     n_iter_no_change=20,
#     random_state=42
# )
# train_and_evaluate(mlp, "MLP Neural Network", X_train, y_train, X_test, y_test)

# 5. LightGBM
flat_lgb = lgb.LGBMClassifier(
    num_class=num_classes,
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=100,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
train_and_evaluate(flat_lgb, "LightGBM", X_train, y_train, X_test, y_test)

# # 6. K-Nearest Neighbors （KNN）
# knn = KNeighborsClassifier(
#     n_neighbors=7,           # 常见经验值，可根据数据调整
#     weights='distance',
#     n_jobs=-1
# )
# train_and_evaluate(knn, "KNN", X_train, y_train, X_test, y_test)
