import pandas as pd
import numpy as np
import statsmodels.api as sm

# 创建模拟数据，模拟过采样情况
np.random.seed(123)
n = 1000

# 生成自变量
X = np.random.randn(n, 2)
# 真实模型参数
true_intercept = -3.35
true_coefs = [2.0, 1.5]

# 计算真实概率
logit = true_intercept + X @ true_coefs
true_prob = 1 / (1 + np.exp(-logit))

# 生成响应变量
y = np.random.binomial(1, true_prob)

# 创建DataFrame
df = pd.DataFrame({'y': y, 'x1': X[:, 0], 'x2': X[:, 1]})

# 模拟过采样：保留所有y=1，但只保留部分y=0
df_event = df[df['y'] == 1]
df_nonevent = df[df['y'] == 0].sample(frac=0.2, random_state=456)  # 仅保留20%的非事件
df_oversampled = pd.concat([df_event, df_nonevent])

# 计算总体和样本事件概率
p1 = df['y'].mean()  # 总体事件概率
r1 = df_oversampled['y'].mean()  # 过采样后样本事件概率

print(f"总体事件概率: {p1:.4f}")
print(f"样本事件概率: {r1:.4f}")

# 计算权重
df_oversampled['weight'] = np.where(
    df_oversampled['y'] == 1, 
    p1 / r1, 
    (1 - p1) / (1 - r1)
)

# 加权逻辑回归
X = sm.add_constant(df_oversampled[['x1', 'x2']])
y = df_oversampled['y']
df_oversampled['weight'] = 1
weights = df_oversampled['weight']

#%%

model_weighted = sm.GLM(y, X, 
                        family=sm.families.Binomial(),
                        freq_weights=weights).fit()

model_weighted = sm.GLM(y, X, 
                        family=sm.families.Binomial(),
                        freq_weights=weights).fit(method='nm',maxiter=1000)

model = sm.Logit(y,X).fit()

print("\n=== 加权调整模型结果 ===")
print(f"截距: {model_weighted.params['const']:.4f}")
print(f"x1系数: {model_weighted.params['x1']:.4f}")
print(f"x2系数: {model_weighted.params['x2']:.4f}")

#%% Offset part

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# 创建示例数据
np.random.seed(123)
n = 200
df = pd.DataFrame({
    'x1': np.random.randn(n),
    'x2': np.random.randn(n),
    'x3': np.random.randn(n),
    'y': np.random.binomial(1, 0.5, n)
})

# method 1
# 1. 创建偏移变量（对应SAS中的Restrict变量）
df['Restrict'] = df['x1'] + 2 * df['x2']

# 2. 使用公式语法，offset()固定系数为1，且不包含x1和x2在模型中
model = smf.glm('y ~ x3 + offset(Restrict)', 
                data=df, 
                family=sm.families.Binomial()).fit()

print(model.summary())
print(f"\n固定系数：x1=1, x2=2")
print(f"估计的x3系数: {model.params['x3']:.4f}")


# method 2
# 创建偏移项
df['offset_term'] = df['x1'] + 2 * df['x2']

# 构建设计矩阵 - 只包含x3，不包含x1和x2
X = sm.add_constant(df[['x3']])  # 只包含常数项和x3
y = df['y']

# 拟合模型，指定offset参数
model = sm.GLM(y, X, 
               family=sm.families.Binomial(),
               offset=df['offset_term']).fit()

print(model.summary())
print(f"\n偏移量形状: {df['offset_term'].shape}")
print(f"设计矩阵列: {X.columns.tolist()}")

#%%

model_weighted = sm.GLM(y, X, 
                        family=sm.families.Binomial(),
                        freq_weights=weights).fit(method='newton',maxiter=1000)

model = sm.GLM(y, X, 
               family=sm.families.Binomial(),
               offset=df['offset_term']).fit(method='newton',maxiter=1000)


#%%

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 生成模拟数据
np.random.seed(123)
n = 1000

# 真实参数
true_beta = np.array([-2.0, 1.5, -0.8, 0.6])

# 生成自变量
X = np.column_stack([
    np.ones(n),
    np.random.randn(n),
    np.random.randn(n),
    np.random.randn(n)
])

# 计算真实概率
linear_predictor = X @ true_beta
true_prob = 1 / (1 + np.exp(-linear_predictor))

# 生成响应变量
y = np.random.binomial(1, true_prob)

# 创建DataFrame
df = pd.DataFrame(X, columns=['const', 'x1', 'x2', 'x3'])
df['y'] = y

# 1. 创建权重（示例：过采样校正权重）
p1 = 0.10  # 总体事件概率
r1 = df['y'].mean()  # 样本事件概率
print(f"总体事件概率: {p1:.4f}")
print(f"样本事件概率: {r1:.4f}")

# 计算过采样校正权重
weights = np.where(df['y'] == 1, p1 / r1, (1 - p1) / (1 - r1))
df['weights'] = weights

# 2. 创建偏移量（示例：固定某些变量的系数）
# 假设我们想固定x1系数为1，x2系数为0.5
df['offset_term'] = df['x1'] * 1.0 + df['x2'] * 0.5

# 3. 同时使用权重和偏移量的模型
print("\n=== 同时使用权重和偏移量的模型 ===")
model_combined = sm.GLM(df['y'], df[['const', 'x3']],  # 注意：x1和x2已通过偏移量固定
                       family=sm.families.Binomial(),
                       freq_weights=df['weights'],
                       offset=df['offset_term']).fit(method='newton', maxiter=1000)

print(model_combined.summary())
print(f"\n固定系数: x1 = 1.0, x2 = 0.5")
print(f"估计的截距: {model_combined.params['const']:.4f}")
print(f"估计的x3系数: {model_combined.params['x3']:.4f}")

# 4. 对比不同模型
models = {}

# 无权重无偏移量的基准模型
print("\n=== 基准模型（无权重无偏移量） ===")
model_baseline = sm.GLM(df['y'], df[['const', 'x1', 'x2', 'x3']],
                       family=sm.families.Binomial()).fit(method='newton', maxiter=1000)
models['baseline'] = model_baseline
print(f"截距: {model_baseline.params['const']:.4f}")
print(f"x1系数: {model_baseline.params['x1']:.4f}")
print(f"x2系数: {model_baseline.params['x2']:.4f}")
print(f"x3系数: {model_baseline.params['x3']:.4f}")

# 仅权重的模型
print("\n=== 仅权重模型 ===")
model_weighted_only = sm.GLM(df['y'], df[['const', 'x1', 'x2', 'x3']],
                            family=sm.families.Binomial(),
                            freq_weights=df['weights']).fit(method='newton', maxiter=1000)
models['weighted_only'] = model_weighted_only
print(f"截距: {model_weighted_only.params['const']:.4f}")
print(f"x1系数: {model_weighted_only.params['x1']:.4f}")
print(f"x2系数: {model_weighted_only.params['x2']:.4f}")
print(f"x3系数: {model_weighted_only.params['x3']:.4f}")

# 仅偏移量的模型
print("\n=== 仅偏移量模型 ===")
# 注意：这里使用完整的X，因为偏移量固定了x1和x2的系数
model_offset_only = sm.GLM(df['y'], df[['const', 'x3']],
                          family=sm.families.Binomial(),
                          offset=df['offset_term']).fit(method='newton', maxiter=1000)
models['offset_only'] = model_offset_only
print(f"截距: {model_offset_only.params['const']:.4f}")
print(f"x3系数: {model_offset_only.params['x3']:.4f}")

# 5. 计算预测概率并比较
df['pred_baseline'] = model_baseline.predict(df[['const', 'x1', 'x2', 'x3']])
df['pred_weighted_only'] = model_weighted_only.predict(df[['const', 'x1', 'x2', 'x3']])
df['pred_offset_only'] = model_offset_only.predict(df[['const', 'x3']])
df['pred_combined'] = model_combined.predict(df[['const', 'x3']])

# 6. 可视化比较
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. 参数估计比较
ax = axes[0, 0]
model_names = ['基准模型', '仅权重', '仅偏移量', '权重+偏移量']
intercepts = [
    model_baseline.params['const'],
    model_weighted_only.params['const'],
    model_offset_only.params['const'],
    model_combined.params['const']
]
x3_coefs = [
    model_baseline.params['x3'],
    model_weighted_only.params['x3'],
    model_offset_only.params['x3'],
    model_combined.params['x3']
]

x = np.arange(len(model_names))
width = 0.35
ax.bar(x - width/2, intercepts, width, label='截距', color='skyblue', alpha=0.7)
ax.bar(x + width/2, x3_coefs, width, label='x3系数', color='lightcoral', alpha=0.7)
ax.axhline(y=true_beta[0], color='blue', linestyle='--', alpha=0.5, label='真实截距')
ax.axhline(y=true_beta[3], color='red', linestyle='--', alpha=0.5, label='真实x3系数')
ax.set_xlabel('模型类型')
ax.set_ylabel('系数值')
ax.set_title('参数估计比较')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 预测概率分布
ax = axes[0, 1]
bins = np.linspace(0, 1, 21)
ax.hist(df['pred_baseline'], bins=bins, alpha=0.5, label='基准模型', density=True)
ax.hist(df['pred_weighted_only'], bins=bins, alpha=0.5, label='仅权重', density=True)
ax.hist(df['pred_offset_only'], bins=bins, alpha=0.5, label='仅偏移量', density=True)
ax.hist(df['pred_combined'], bins=bins, alpha=0.5, label='权重+偏移量', density=True)
ax.set_xlabel('预测概率')
ax.set_ylabel('密度')
ax.set_title('预测概率分布')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. 与真实概率的相关性
ax = axes[0, 2]
models_to_compare = ['pred_baseline', 'pred_weighted_only', 'pred_offset_only', 'pred_combined']
colors = ['blue', 'green', 'red', 'purple']
for i, pred_col in enumerate(models_to_compare):
    ax.scatter(true_prob, df[pred_col], alpha=0.3, s=10, color=colors[i], 
               label=pred_col.replace('pred_', ''))
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='完美预测')
ax.set_xlabel('真实概率')
ax.set_ylabel('预测概率')
ax.set_title('预测校准')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. 平均预测概率
ax = axes[1, 0]
mean_probs = [df[col].mean() for col in models_to_compare]
ax.bar(model_names, mean_probs, alpha=0.7)
ax.axhline(y=true_prob.mean(), color='red', linestyle='--', label='真实平均概率')
ax.set_xlabel('模型类型')
ax.set_ylabel('平均预测概率')
ax.set_title('平均预测概率比较')
ax.set_xticklabels(model_names, rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)

# 在柱状图上添加数值
for i, v in enumerate(mean_probs):
    ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

# 5. 误差分布
ax = axes[1, 1]
errors = {}
for pred_col in models_to_compare:
    error_name = pred_col.replace('pred_', 'error_')
    df[error_name] = df[pred_col] - true_prob
    errors[error_name] = df[error_name]

for i, (error_name, error_vals) in enumerate(errors.items()):
    ax.hist(error_vals, bins=30, alpha=0.5, density=True, 
            label=error_name.replace('error_', ''))

ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('预测误差 (预测-真实)')
ax.set_ylabel('密度')
ax.set_title('误差分布')
ax.legend()
ax.grid(True, alpha=0.3)

# 6. 性能指标
ax = axes[1, 2]
ax.axis('off')  # 关闭坐标轴，用于显示表格

# 计算性能指标
from sklearn.metrics import mean_squared_error, roc_auc_score

performance_data = []
for pred_col in models_to_compare:
    mse = mean_squared_error(true_prob, df[pred_col])
    auc = roc_auc_score(y, df[pred_col])
    mean_abs_error = np.mean(np.abs(df[pred_col] - true_prob))
    
    performance_data.append([
        pred_col.replace('pred_', ''),
        f'{mse:.4f}',
        f'{auc:.4f}',
        f'{mean_abs_error:.4f}'
    ])

# 创建表格
table = ax.table(cellText=performance_data,
                 colLabels=['模型', 'MSE', 'AUC', 'MAE'],
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
ax.set_title('模型性能指标', fontsize=12, pad=20)

plt.tight_layout()
plt.show()

# 7. 打印权重和偏移量的统计信息
print("\n" + "="*50)
print("权重和偏移量的统计信息:")
print("="*50)
print(f"权重 - 均值: {df['weights'].mean():.4f}, 标准差: {df['weights'].std():.4f}")
print(f"权重 - 最小值: {df['weights'].min():.4f}, 最大值: {df['weights'].max():.4f}")
print(f"偏移量 - 均值: {df['offset_term'].mean():.4f}, 标准差: {df['offset_term'].std():.4f}")
print(f"偏移量 - 最小值: {df['offset_term'].min():.4f}, 最大值: {df['offset_term'].max():.4f}")

# 检查权重和偏移量的相关性
correlation = np.corrcoef(df['weights'], df['offset_term'])[0, 1]
print(f"权重与偏移量的相关性: {correlation:.4f}")

#%%

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Generate simulated data
np.random.seed(123)
n = 1000

# True parameters
true_beta = np.array([-2.0, 1.5, -0.8, 0.6])

# Generate independent variables
X = np.column_stack([
    np.ones(n),
    np.random.randn(n),
    np.random.randn(n),
    np.random.randn(n)
])

# Calculate true probabilities
linear_predictor = X @ true_beta
true_prob = 1 / (1 + np.exp(-linear_predictor))

# Generate response variable
y = np.random.binomial(1, true_prob)

# Create DataFrame
df = pd.DataFrame(X, columns=['const', 'x1', 'x2', 'x3'])
df['y'] = y

# 1. Create weights (example: oversampling correction weights)
p1 = 0.10  # Population event probability
r1 = df['y'].mean()  # Sample event probability
print(f"Population event probability: {p1:.4f}")
print(f"Sample event probability: {r1:.4f}")

# Calculate oversampling correction weights
weights = np.where(df['y'] == 1, p1 / r1, (1 - p1) / (1 - r1))
df['weights'] = weights

# 2. Create offset (example: fixing coefficients of certain variables)
# Suppose we want to fix x1 coefficient as 1, x2 coefficient as 0.5
df['offset_term'] = df['x1'] * 1.0 + df['x2'] * 0.5

# 3. Model using both weights and offset
print("\n=== Model Using Both Weights and Offset ===")
model_combined = sm.GLM(df['y'], df[['const', 'x3']],  # Note: x1 and x2 are fixed via offset
                       family=sm.families.Binomial(),
                       freq_weights=df['weights'],
                       offset=df['offset_term']).fit(method='newton', maxiter=1000)

print(model_combined.summary())
print(f"\nFixed coefficients: x1 = 1.0, x2 = 0.5")
print(f"Estimated intercept: {model_combined.params['const']:.4f}")
print(f"Estimated x3 coefficient: {model_combined.params['x3']:.4f}")

# 4. Compare different models
models = {}

# Baseline model (no weights, no offset)
print("\n=== Baseline Model (No Weights, No Offset) ===")
model_baseline = sm.GLM(df['y'], df[['const', 'x1', 'x2', 'x3']],
                       family=sm.families.Binomial()).fit(method='newton', maxiter=1000)
models['baseline'] = model_baseline
print(f"Intercept: {model_baseline.params['const']:.4f}")
print(f"x1 coefficient: {model_baseline.params['x1']:.4f}")
print(f"x2 coefficient: {model_baseline.params['x2']:.4f}")
print(f"x3 coefficient: {model_baseline.params['x3']:.4f}")

# Weight-only model
print("\n=== Weight-Only Model ===")
model_weighted_only = sm.GLM(df['y'], df[['const', 'x1', 'x2', 'x3']],
                            family=sm.families.Binomial(),
                            freq_weights=df['weights']).fit(method='newton', maxiter=1000)
models['weighted_only'] = model_weighted_only
print(f"Intercept: {model_weighted_only.params['const']:.4f}")
print(f"x1 coefficient: {model_weighted_only.params['x1']:.4f}")
print(f"x2 coefficient: {model_weighted_only.params['x2']:.4f}")
print(f"x3 coefficient: {model_weighted_only.params['x3']:.4f}")

# Offset-only model
print("\n=== Offset-Only Model ===")
# Note: Using full X because offset fixes x1 and x2 coefficients
model_offset_only = sm.GLM(df['y'], df[['const', 'x3']],
                          family=sm.families.Binomial(),
                          offset=df['offset_term']).fit(method='newton', maxiter=1000)
models['offset_only'] = model_offset_only
print(f"Intercept: {model_offset_only.params['const']:.4f}")
print(f"x3 coefficient: {model_offset_only.params['x3']:.4f}")

# 5. Calculate predicted probabilities and compare
df['pred_baseline'] = model_baseline.predict(df[['const', 'x1', 'x2', 'x3']])
df['pred_weighted_only'] = model_weighted_only.predict(df[['const', 'x1', 'x2', 'x3']])
df['pred_offset_only'] = model_offset_only.predict(df[['const', 'x3']])
df['pred_combined'] = model_combined.predict(df[['const', 'x3']])

# 6. Visual comparison
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Parameter estimation comparison
ax = axes[0, 0]
model_names = ['Baseline', 'Weight-Only', 'Offset-Only', 'Weight+Offset']
intercepts = [
    model_baseline.params['const'],
    model_weighted_only.params['const'],
    model_offset_only.params['const'],
    model_combined.params['const']
]
x3_coefs = [
    model_baseline.params['x3'],
    model_weighted_only.params['x3'],
    model_offset_only.params['x3'],
    model_combined.params['x3']
]

x = np.arange(len(model_names))
width = 0.35
ax.bar(x - width/2, intercepts, width, label='Intercept', color='skyblue', alpha=0.7)
ax.bar(x + width/2, x3_coefs, width, label='x3 Coefficient', color='lightcoral', alpha=0.7)
ax.axhline(y=true_beta[0], color='blue', linestyle='--', alpha=0.5, label='True Intercept')
ax.axhline(y=true_beta[3], color='red', linestyle='--', alpha=0.5, label='True x3 Coefficient')
ax.set_xlabel('Model Type')
ax.set_ylabel('Coefficient Value')
ax.set_title('Parameter Estimation Comparison')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Predicted probability distribution
ax = axes[0, 1]
bins = np.linspace(0, 1, 21)
ax.hist(df['pred_baseline'], bins=bins, alpha=0.5, label='Baseline', density=True)
ax.hist(df['pred_weighted_only'], bins=bins, alpha=0.5, label='Weight-Only', density=True)
ax.hist(df['pred_offset_only'], bins=bins, alpha=0.5, label='Offset-Only', density=True)
ax.hist(df['pred_combined'], bins=bins, alpha=0.5, label='Weight+Offset', density=True)
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Density')
ax.set_title('Predicted Probability Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Correlation with true probabilities
ax = axes[0, 2]
models_to_compare = ['pred_baseline', 'pred_weighted_only', 'pred_offset_only', 'pred_combined']
colors = ['blue', 'green', 'red', 'purple']
for i, pred_col in enumerate(models_to_compare):
    ax.scatter(true_prob, df[pred_col], alpha=0.3, s=10, color=colors[i], 
               label=pred_col.replace('pred_', ''))
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Prediction')
ax.set_xlabel('True Probability')
ax.set_ylabel('Predicted Probability')
ax.set_title('Prediction Calibration')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Mean predicted probability
ax = axes[1, 0]
mean_probs = [df[col].mean() for col in models_to_compare]
ax.bar(model_names, mean_probs, alpha=0.7)
ax.axhline(y=true_prob.mean(), color='red', linestyle='--', label='True Mean Probability')
ax.set_xlabel('Model Type')
ax.set_ylabel('Mean Predicted Probability')
ax.set_title('Mean Predicted Probability Comparison')
ax.set_xticklabels(model_names, rotation=45)
ax.legend()
ax.grid(True, alpha=0.3)

# Add values on bars
for i, v in enumerate(mean_probs):
    ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

# 5. Error distribution
ax = axes[1, 1]
errors = {}
for pred_col in models_to_compare:
    error_name = pred_col.replace('pred_', 'error_')
    df[error_name] = df[pred_col] - true_prob
    errors[error_name] = df[error_name]

for i, (error_name, error_vals) in enumerate(errors.items()):
    ax.hist(error_vals, bins=30, alpha=0.5, density=True, 
            label=error_name.replace('error_', ''))

ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('Prediction Error (Predicted - True)')
ax.set_ylabel('Density')
ax.set_title('Error Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 6. Performance metrics
ax = axes[1, 2]
ax.axis('off')  # Turn off axes for displaying table

# Calculate performance metrics
from sklearn.metrics import mean_squared_error, roc_auc_score

performance_data = []
for pred_col in models_to_compare:
    mse = mean_squared_error(true_prob, df[pred_col])
    auc = roc_auc_score(y, df[pred_col])
    mean_abs_error = np.mean(np.abs(df[pred_col] - true_prob))
    
    performance_data.append([
        pred_col.replace('pred_', ''),
        f'{mse:.4f}',
        f'{auc:.4f}',
        f'{mean_abs_error:.4f}'
    ])

# Create table
table = ax.table(cellText=performance_data,
                 colLabels=['Model', 'MSE', 'AUC', 'MAE'],
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
ax.set_title('Model Performance Metrics', fontsize=12, pad=20)

plt.tight_layout()
plt.show()

# 7. Print statistics of weights and offsets
print("\n" + "="*50)
print("Statistics of Weights and Offsets:")
print("="*50)
print(f"Weights - Mean: {df['weights'].mean():.4f}, Std: {df['weights'].std():.4f}")
print(f"Weights - Min: {df['weights'].min():.4f}, Max: {df['weights'].max():.4f}")
print(f"Offset - Mean: {df['offset_term'].mean():.4f}, Std: {df['offset_term'].std():.4f}")
print(f"Offset - Min: {df['offset_term'].min():.4f}, Max: {df['offset_term'].max():.4f}")

# Check correlation between weights and offsets
correlation = np.corrcoef(df['weights'], df['offset_term'])[0, 1]
print(f"Correlation between weights and offset: {correlation:.4f}")







