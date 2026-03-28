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
    n_estimators=500, # 500
    max_depth=5,
    learning_rate=0.08, # 0.08
    subsample=0.8,
    colsample_bytree=0.8,
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
    n_estimators=700, # 700
    max_depth=5, # 5
    learning_rate=0.02, # 0.02
    subsample=1, # 1
    colsample_bytree=1, # 1
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
    n_estimators=100,
    max_depth=3, # 3
    learning_rate=0.6, # 0.6
    subsample=0.5, # 0.5
    colsample_bytree=0.5, # 0.5
    eval_metric="mlogloss",
    random_state=42
)

xgb_carbonate.fit(X_carbonate, y_carbonate)


# -----------预测融合------------
# 获取二分类（碳酸/碎屑）中为1的数据（碳酸）
family_prob = xgb_family.predict_proba(X_test)[:, 1]
family_pred = family_prob > 0.2  # 已验证：0.2为最优阈值

y_pred = np.zeros(len(X_test), dtype=int) # 初始化最终预测结果数组

# 碎屑岩
idx_clastic = np.where(family_pred == 0)[0] # 找到二分类中被预测为0（碎屑）的索引
proba = xgb_clastic.predict_proba(X_test[idx_clastic]) # 用碎屑岩模型进行预测，得到预测概率
y_local = np.argmax(proba, axis=1) # 取出概率最大的类别索引
y_pred[idx_clastic] = [clastic_inv_map[int(y)] for y in y_local] # 映射会最终标签


# 碳酸盐岩
idx_carbonate = np.where(family_pred == 1)[0] # 找到二分类中被预测为1（碳酸）的索引
proba = xgb_carbonate.predict_proba(X_test[idx_carbonate])
y_local = np.argmax(proba, axis=1)   # 一定是 0 / 1
y_pred[idx_carbonate] = [carbonate_inv_map[int(y)] for y in y_local]

# # 将预测结果保存到新文件中
# df_test = pd.read_csv(test_path)
# df_predict = df_test.copy()
# df_predict['Predicted_Lithology'] = y_pred+1
# df_predict.to_csv("./results/HXGBoost_井2预测结果.csv")
#
# -----------评估------------
print("Hierarchical XGBoost Results\n")
print("Macro-F1      :", f1_score(y_test, y_pred, average="macro"))
# print("Weighted-F1   :", f1_score(y_test, y_pred, average="weighted"))
# print("Cohen Kappa   :", cohen_kappa_score(y_test, y_pred))
# print("\nClassification Report:\n")
# print(classification_report(y_test, y_pred, digits=4))
# print("Confusion Matrix:\n")
# print(confusion_matrix(y_test, y_pred))

# # =========================
# # 特征名称 & 类型映射
# # =========================
# feature_names = specific_features
#
# def get_feature_type(name):
#     if name in basic_features:
#         return "Original"
#     elif name in ratio_features:
#         return "Ratio"
#     elif name in product_features:
#         return "Product"
#     elif name in log_features:
#         return "Log"
#     elif name in square_features:
#         return "Square"
#     else:
#         return "Other"
#
# # 提取特征重要性
# def extract_xgb_importance(model, feature_names, model_name):
#     booster = model.get_booster()
#     score = booster.get_score(importance_type="gain")
#
#     df = pd.DataFrame({
#         "Feature_Index": list(score.keys()), # 获取特征索引
#         "Importance": list(score.values())
#     })
#
#     # 将特征索引映射为特征名
#     df["Feature"] = df["Feature_Index"].apply(
#         lambda x: feature_names[int(x[1:])]
#     )
#     # 获取每个特征的特征类型
#     df["Feature_Type"] = df["Feature"].apply(get_feature_type)
#     df["Model"] = model_name
#     # 按重要性降序排列
#     df = df.sort_values("Importance", ascending=False).reset_index(drop=True)
#
#     # 归一化（便于多模型 / 多井比较）
#     df["Importance_norm"] = df["Importance"] / df["Importance"].sum()
#
#     return df
#
# # 岩性族级二分类器
# family_importance_df = extract_xgb_importance(
#     xgb_family,
#     feature_names,
#     model_name="Lithology_Family_Classifier"
# )
#
# # 碎屑岩族子模型
# clastic_importance_df = extract_xgb_importance(
#     xgb_clastic,
#     feature_names,
#     model_name="Clastic_Submodel"
# )
#
# # 碳酸盐岩族子模型
# carbonate_importance_df = extract_xgb_importance(
#     xgb_carbonate,
#     feature_names,
#     model_name="Carbonate_Submodel"
# )
#
# # 保存特征重要性表
# all_importance_df = pd.concat(
#     [family_importance_df, clastic_importance_df, carbonate_importance_df],
#     axis=0
# )
#
# # 三种模型的特征重要性
# all_importance_df.to_csv(
#     "./results/HXGBoost_FeatureImportance_Well2.csv",
#     index=False
# )
# # 按照特征类型汇总贡献
# type_summary = (
#     all_importance_df
#     .groupby(["Model", "Feature_Type"])["Importance_norm"]
#     .sum()
#     .reset_index()
# )
#
# # 特征类型汇总贡献
# type_summary.to_csv(
#     "./results/HXGBoost_FeatureType_Contribution_Well2.csv",
#     index=False
# )
