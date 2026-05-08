# Hankel-SVD-SSA框架内音频降噪改进方法分析

## 概述

本文档分析在Hankel-SVD-SSA框架内改善音频降噪效果的可能方法（除了后处理筛选）。当前系统已实现：
- Hankel矩阵嵌入 + 多声道联合SVD分解
- 多种截断策略：固定秩、能量阈值、Wiener软加权
- W-correlation分组（已验证效果有限）
- 反对角线平均化重构

**核心问题**：SVD分量中信号和噪声混合，后处理筛选无法有效分离。

---

## 1. 改变分解本身使信号和噪声更好分离

### 1.1 自适应Hankel窗口长度

**原理**：当前使用固定的窗口长度L，但音频信号的局部特性（频率内容、平稳性）随时间变化。自适应选择L可以使Hankel矩阵更好地匹配局部信号结构。

**实现方式**：
```python
def adaptive_window_length(signal_segment, min_L=64, max_L=1024):
    """基于信号特性选择最优窗口长度"""
    # 方法1：基于局部平稳性
    # 计算信号段的自相关函数，选择自相关衰减到0.5的延迟作为L
    
    # 方法2：基于频谱平坦度（SFM）
    # SFM接近1（噪声）→ 短窗口
    # SFM接近0（谐波）→ 长窗口
    
    # 方法3：基于瞬时频率方差
    # 方差大 → 短窗口（捕获快速变化）
    # 方差小 → 长窗口（捕获稳定结构）
    
    return optimal_L
```

**优势**：
- 更好地匹配局部信号特性
- 短窗口捕获瞬态，长窗口捕获稳态
- 可能改善信号噪声分离

**挑战**：
- 增加计算复杂度
- 需要可靠的局部特性估计方法
- 帧间窗口长度变化可能导致OLA不连续

### 1.2 多分辨率Hankel矩阵（MRSSA）

**原理**：使用多个不同窗口长度的Hankel矩阵，对每个分辨率进行独立SVD分解，然后融合结果。

**实现方式**：
```python
def multiresolution_ssa(signal, L_list=[64, 256, 1024]):
    """多分辨率SSA"""
    results = []
    for L in L_list:
        H = hankel_embed(signal, L)
        U, S, Vh = scipy.linalg.svd(H)
        # 对每个分辨率进行Wiener软加权
        weighted = wiener_weighting(U, S, Vh)
        reconstructed = diagonal_average(weighted)
        results.append(reconstructed)
    
    # 融合策略：
    # 1. 加权平均（基于每个分辨率的SNR估计）
    # 2. 选择性融合（选择每个时间点最强的分辨率）
    # 3. 频域融合（不同频率使用不同分辨率）
    
    return fused_result
```

**优势**：
- 短窗口：捕获瞬态和高频细节
- 长窗口：捕获低频结构和长期相关性
- 融合可以互补优势

**挑战**：
- 计算成本增加（多个SVD）
- 融合策略需要仔细设计
- 可能引入相位不一致

### 1.3 非均匀嵌入

**原理**：当前使用均匀嵌入（连续样本），但可以使用非均匀采样模式，如对数间隔或基于信号特性的自适应间隔。

**实现方式**：
```python
def nonuniform_embedding(signal, L, mode='log'):
    """非均匀Hankel嵌入"""
    if mode == 'log':
        # 对数间隔：更好地捕获低频特性
        indices = np.logspace(0, np.log10(len(signal)), L, dtype=int)
    elif mode == 'adaptive':
        # 基于信号局部频率的自适应间隔
        # 高频区域密集采样，低频区域稀疏采样
        pass
    
    H = np.array([signal[indices + i] for i in range(len(signal) - max(indices))])
    return H
```

**优势**：
- 可能更好地匹配音频信号的频谱特性
- 对数间隔可能更适合音乐信号

**挑战**：
- 破坏Hankel矩阵的Toeplitz结构
- 反对角线平均化需要修改
- 数学理论基础需要验证

### 1.4 预处理增强

**原理**：在Hankel嵌入前对信号进行预处理，增强信号结构或抑制噪声。

**方法**：
1. **预加重**：高通滤波增强高频，补偿高频噪声抑制
2. **频谱加权**：基于人耳听觉特性加权（A加权、C加权）
3. **自适应增益控制**：归一化局部能量，使SVD更稳定

```python
def preprocess_for_hankel(signal, sr, method='preemphasis'):
    """Hankel嵌入前的预处理"""
    if method == 'preemphasis':
        # 预加重滤波器
        alpha = 0.97
        filtered = np.append(signal[0], signal[1:] - alpha * signal[:-1])
        return filtered
    elif method == 'spectral_weighting':
        # A加权（简化版）
        # 实际应使用完整的A加权滤波器
        pass
```

**优势**：
- 简单易实现
- 可能改善SVD的信号噪声分离

**挑战**：
- 需要仔细选择预处理参数
- 可能引入失真

---

## 2. SVD阶段改善分离效果的其他数学方法

### 2.1 鲁棒PCA（Robust Principal Component Analysis）

**原理**：将矩阵分解为低秩部分（L，信号）+稀疏部分（S，噪声），而不是单一的SVD分解。

**数学模型**：
$$\min_{L,S} \|L\|_* + \lambda\|S\|_1 \quad \text{s.t.} \quad X = L + S$$

其中$\|L\|_*$是核范数（奇异值之和），$\|S\|_1$是L1范数。

**实现方式**：
```python
def robust_pca(X, lambda_param=None, max_iter=100):
    """鲁棒PCA via ADMM"""
    if lambda_param is None:
        lambda_param = 1.0 / np.sqrt(max(X.shape))
    
    L = np.zeros_like(X)
    S = np.zeros_like(X)
    Y = np.zeros_like(X)
    
    mu = 1.25 * np.linalg.norm(X)  # 惩罚参数
    rho = 1.6  # ADMM松弛参数
    
    for i in range(max_iter):
        # 更新L：软阈值奇异值
        U, sigma, Vt = np.linalg.svd(X - S + Y/mu, full_matrices=False)
        sigma_thresh = np.maximum(sigma - 1/mu, 0)
        L = U @ np.diag(sigma_thresh) @ Vt
        
        # 更新S：软阈值
        S = np.maximum(np.abs(X - L + Y/mu) - lambda_param/mu, 0) * np.sign(X - L + Y/mu)
        
        # 更新Y
        Y = Y + mu * (X - L - S)
        
        # 检查收敛
        if np.linalg.norm(X - L - S) < 1e-6:
            break
    
    return L, S
```

**优势**：
- 明确分离低秩（信号）和稀疏（噪声）
- 理论基础扎实
- 可能比Wiener软加权更好分离信号噪声

**挑战**：
- 计算成本高（迭代优化）
- 需要调整lambda参数
- 可能不适合实时处理

### 2.2 加权SVD（Weighted SVD）

**原理**：对不同奇异值应用不同权重，基于先验知识或自适应估计。

**实现方式**：
```python
def weighted_svd(H, weights=None):
    """加权SVD"""
    U, S, Vh = np.linalg.svd(H, full_matrices=False)
    
    if weights is None:
        # 基于奇异值衰减率的自适应权重
        # 快速衰减 → 信号 → 高权重
        # 缓慢衰减 → 噪声 → 低权重
        decay_rate = np.diff(np.log(S + 1e-10))
        weights = np.exp(-decay_rate.mean() * np.arange(len(S)))
        weights = np.concatenate([[1], weights])  # 第一个分量权重最大
    
    weighted_S = S * weights
    return U, weighted_S, Vh
```

**优势**：
- 可以融入先验知识
- 计算简单（在SVD基础上）

**挑战**：
- 权重选择需要经验或自适应方法

### 2.3 稀疏SVD

**原理**：约束奇异向量的稀疏性，使信号分量更集中，噪声分量更分散。

**数学模型**：
$$\min_{U,V} \|X - U \Sigma V^T\|_F^2 + \lambda(\|U\|_1 + \|V\|_1)$$

**实现方式**：
```python
def sparse_svd(H, sparsity_penalty=0.1, max_iter=50):
    """稀疏SVD via迭代软阈值"""
    # 初始化
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    
    for i in range(max_iter):
        # 更新U：软阈值
        U_new = np.maximum(np.abs(U) - sparsity_penalty, 0) * np.sign(U)
        # 正交化
        U_new, _ = np.linalg.qr(U_new)
        
        # 更新V：软阈值
        Vt_new = np.maximum(np.abs(Vt) - sparsity_penalty, 0) * np.sign(Vt)
        Vt_new, _ = np.linalg.qr(Vt_new.T)
        Vt_new = Vt_new.T
        
        # 更新S
        S_new = np.diag(U_new.T @ H @ Vt_new.T)
        
        U, S, Vt = U_new, S_new, Vt_new
    
    return U, S, Vt
```

**优势**：
- 稀疏奇异向量可能更好地分离信号噪声
- 信号通常集中在少数分量，噪声分散

**挑战**：
- 计算复杂度高
- 稀疏性约束可能引入失真

### 2.4 基于信息论准则的截断

**原理**：使用AIC（赤池信息准则）或BIC（贝叶斯信息准则）选择最优截断秩，而不是简单的能量阈值。

**实现方式**：
```python
def aic_truncation(S, m, n):
    """基于AIC的截断秩选择"""
    # S: 奇异值
    # m, n: 矩阵维度
    
    k_range = np.arange(1, len(S) + 1)
    aic_values = []
    
    for k in k_range:
        # 信号能量
        signal_energy = np.sum(S[:k]**2)
        # 噪声方差估计
        noise_var = np.mean(S[k:]**2) if k < len(S) else 0
        # AIC
        # AIC = n*log(noise_var) + 2*k
        # 其中n是自由参数个数
        n_params = k * (m + n - k)  # 奇异向量的自由度
        aic = (m * n) * np.log(noise_var + 1e-10) + 2 * n_params
        aic_values.append(aic)
    
    optimal_k = k_range[np.argmin(aic_values)]
    return optimal_k
```

**优势**：
- 统计上最优的截断选择
- 不需要手动设置能量阈值

**挑战**：
- 需要可靠的噪声方差估计
- 计算成本略高

### 2.5 贝叶斯SVD

**原理**：使用贝叶斯框架估计信号和噪声的后验分布，得到更稳健的估计。

**实现方式**：
```python
def bayesian_svd(H, noise_var_prior=1.0):
    """贝叶斯SVD（简化版）"""
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    
    # 假设先验：信号奇异值服从指数分布，噪声服从高斯分布
    # 后验估计：收缩奇异值
    
    # James-Stein收缩估计器
    n, p = H.shape
    sigma2 = noise_var_prior
    
    # 收缩因子
    shrinkage = 1 - (sigma2 * min(n, p)) / (S**2 + 1e-10)
    shrinkage = np.maximum(shrinkage, 0)
    
    S_shrunk = S * shrinkage
    return U, S_shrunk, Vt
```

**优势**：
- 统计上更稳健
- 可以融入先验知识

**挑战**：
- 需要选择合适的先验分布
- 计算可能复杂

---

## 3. 迭代SSA的具体实现方式

### 3.1 基本迭代SSA

**原理**：多次应用SSA，每次对残差进行处理，逐步提取剩余信号成分。

**实现方式**：
```python
def iterative_ssa(signal, L, n_iterations=3, threshold_strategy='energy'):
    """迭代SSA降噪"""
    current_signal = signal.copy()
    total_removed = np.zeros_like(signal)
    
    for i in range(n_iterations):
        # 1. Hankel嵌入
        H = hankel_embed(current_signal, L)
        
        # 2. SVD分解
        U, S, Vt = np.linalg.svd(H, full_matrices=False)
        
        # 3. 截断（可以使用自适应阈值）
        if threshold_strategy == 'energy':
            # 能量阈值随迭代递减
            energy_threshold = 0.9 - 0.1 * i  # 0.9, 0.8, 0.7
            k = select_k_by_energy(S, energy_threshold)
        elif threshold_strategy == 'wiener':
            # Wiener权重
            weights = wiener_weights(S, noise_fraction=0.1 + 0.05*i)
            S_truncated = S * weights
            k = len(S)  # 保留所有分量，但加权
        
        # 4. 重构
        if threshold_strategy == 'energy':
            H_denoised = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        else:
            H_denoised = U @ np.diag(S_truncated) @ Vt
        
        signal_denoised = diagonal_average(H_denoised)
        
        # 5. 计算残差
        residual = current_signal - signal_denoised
        total_removed += residual
        
        # 6. 更新当前信号（用于下一次迭代）
        current_signal = signal_denoised
        
        # 7. 可选：检查收敛
        if np.linalg.norm(residual) < 1e-6 * np.linalg.norm(signal):
            break
    
    return current_signal, total_removed
```

**优势**：
- 逐步提取剩余信号成分
- 可能捕获被第一次SSA遗漏的弱信号

**挑战**：
- 可能过度处理，引入失真
- 需要设计收敛准则

### 3.2 残差反馈迭代SSA

**原理**：将残差反馈到输入端，与原始信号混合后再次处理。

**实现方式**：
```python
def residual_feedback_ssa(signal, L, n_iterations=3, feedback_gain=0.5):
    """残差反馈迭代SSA"""
    current_input = signal.copy()
    output = np.zeros_like(signal)
    
    for i in range(n_iterations):
        # 1. SSA处理
        H = hankel_embed(current_input, L)
        U, S, Vt = np.linalg.svd(H, full_matrices=False)
        
        # Wiener软加权
        weights = wiener_weights(S, noise_fraction=0.1)
        S_weighted = S * weights
        H_denoised = U @ np.diag(S_weighted) @ Vt
        
        signal_denoised = diagonal_average(H_denoised)
        
        # 2. 计算残差
        residual = current_input - signal_denoised
        
        # 3. 更新输出
        output += signal_denoised
        
        # 4. 残差反馈：将残差与原始信号混合
        # 这样下一次迭代可以处理残差中的信号成分
        current_input = signal + feedback_gain * residual
        
        # 5. 可选：递减反馈增益
        feedback_gain *= 0.8
    
    return output
```

**优势**：
- 残差中的信号成分可以被再次提取
- 可能改善弱信号的恢复

**挑战**：
- 可能放大噪声
- 需要仔细调整反馈增益

### 3.3 自适应阈值迭代SSA

**原理**：每次迭代根据当前残差特性自适应调整阈值。

**实现方式**：
```python
def adaptive_threshold_iterative_ssa(signal, L, n_iterations=3):
    """自适应阈值迭代SSA"""
    current_signal = signal.copy()
    output = np.zeros_like(signal)
    
    for i in range(n_iterations):
        # 1. 估计当前信号的SNR
        snr_estimate = estimate_snr(current_signal)
        
        # 2. 根据SNR选择阈值策略
        if snr_estimate > 20:  # 高SNR
            # 使用较激进的阈值
            energy_threshold = 0.7
            noise_fraction = 0.2
        elif snr_estimate > 10:  # 中等SNR
            energy_threshold = 0.8
            noise_fraction = 0.15
        else:  # 低SNR
            # 使用保守的阈值
            energy_threshold = 0.9
            noise_fraction = 0.1
        
        # 3. SSA处理
        H = hankel_embed(current_signal, L)
        U, S, Vt = np.linalg.svd(H, full_matrices=False)
        
        # 4. 自适应截断
        if snr_estimate > 15:
            # 高SNR：使用能量阈值
            k = select_k_by_energy(S, energy_threshold)
            H_denoised = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        else:
            # 低SNR：使用Wiener软加权
            weights = wiener_weights(S, noise_fraction)
            S_weighted = S * weights
            H_denoised = U @ np.diag(S_weighted) @ Vt
        
        signal_denoised = diagonal_average(H_denoised)
        
        # 5. 更新
        output += signal_denoised
        current_signal = signal - output  # 残差
    
    return output
```

**优势**：
- 根据信号特性自适应调整
- 可能比固定阈值更稳健

**挑战**：
- 需要可靠的SNR估计方法
- 计算复杂度增加

### 3.4 多通道迭代SSA（立体声版本）

**原理**：对立体声信号进行迭代SSA，保持左右声道相位一致性。

**实现方式**：
```python
def multichannel_iterative_ssa(stereo_signal, L, n_iterations=3):
    """多通道迭代SSA（保持相位一致性）"""
    left, right = stereo_signal[:, 0], stereo_signal[:, 1]
    
    current_left = left.copy()
    current_right = right.copy()
    output_left = np.zeros_like(left)
    output_right = np.zeros_like(right)
    
    for i in range(n_iterations):
        # 1. 联合Hankel嵌入（保持相位关系）
        H_left = hankel_embed(current_left, L)
        H_right = hankel_embed(current_right, L)
        H_joint = np.vstack([H_left, H_right])  # 联合矩阵
        
        # 2. 联合SVD分解
        U, S, Vt = np.linalg.svd(H_joint, full_matrices=False)
        
        # 3. 自适应截断
        energy_threshold = 0.9 - 0.05 * i
        k = select_k_by_energy(S, energy_threshold)
        
        # 4. 重构联合矩阵
        H_joint_denoised = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        
        # 5. 分离左右声道
        H_left_denoised = H_joint_denoised[:H_left.shape[0], :]
        H_right_denoised = H_joint_denoised[H_left.shape[0]:, :]
        
        # 6. 反对角线平均化
        left_denoised = diagonal_average(H_left_denoised)
        right_denoised = diagonal_average(H_right_denoised)
        
        # 7. 更新
        output_left += left_denoised
        output_right += right_denoised
        current_left = left - output_left
        current_right = right - output_right
    
    return np.column_stack([output_left, output_right])
```

**优势**：
- 保持立体声相位一致性
- 联合分解可能比独立处理更好

**挑战**：
- 计算成本增加（联合矩阵更大）
- 需要处理声道间能量不平衡

---

## 4. 综合改进方案

### 4.1 推荐的改进路径

**短期改进（低风险，易实现）**：
1. **自适应窗口长度**：基于局部SFM选择L
2. **基于AIC/BIC的截断**：替代固定能量阈值
3. **加权SVD**：基于奇异值衰减率的自适应权重

**中期改进（中等风险，需要实验验证）**：
1. **多分辨率SSA**：使用3个不同L，加权融合
2. **迭代SSA（3次）**：残差反馈，自适应阈值
3. **鲁棒PCA**：替代标准SVD，分离低秩+稀疏

**长期改进（高风险，需要深入研究）**：
1. **稀疏SVD**：约束奇异向量稀疏性
2. **贝叶斯SVD**：基于后验分布的估计
3. **非均匀嵌入**：对数间隔或自适应间隔

### 4.2 实验验证方案

**测试数据**：
- Beethoven Op.27 No.2 Adagio（当前测试数据）
- 添加更多测试样本：不同乐器、不同噪声类型

**评估指标**：
- SNR（信噪比）
- PESQ/STOI（语音质量指标）
- 主观听感测试（A/B对比）
- 计算时间

**实验设计**：
1. 单独测试每个改进方法
2. 组合测试（如：自适应L + 迭代SSA）
3. 对比当前基线（Wiener软加权）

---

## 5. 实现建议

### 5.1 代码结构

建议在`src/core/strategies/`目录下添加新的策略模块：

```
src/core/strategies/
├── truncation.py          # 现有
├── grouping.py            # 现有
├── adaptive_window.py     # 新增：自适应窗口长度
├── multiresolution.py     # 新增：多分辨率SSA
├── iterative_ssa.py       # 新增：迭代SSA
├── robust_pca.py          # 新增：鲁棒PCA
└── bayesian.py            # 新增：贝叶斯方法
```

### 5.2 参数调优

**自适应窗口长度**：
- min_L = 64, max_L = 1024
- 基于SFM的阈值：SFM > 0.5 → 短窗口，SFM < 0.2 → 长窗口

**迭代SSA**：
- n_iterations = 3（默认）
- feedback_gain = 0.5（默认），递减因子0.8
- 收敛准则：残差能量 < 1e-6 * 原始信号能量

**鲁棒PCA**：
- lambda_param = 1/sqrt(max(m,n))
- max_iter = 100
- 收敛容差 = 1e-6

---

## 6. 预期效果

**自适应窗口长度**：
- 预期SNR提升：2-5dB
- 计算成本：略增（需要局部特性估计）

**多分辨率SSA**：
- 预期SNR提升：3-8dB
- 计算成本：3倍（3个分辨率）

**迭代SSA**：
- 预期SNR提升：5-10dB（3次迭代）
- 计算成本：3倍（3次迭代）

**鲁棒PCA**：
- 预期SNR提升：5-15dB
- 计算成本：10-50倍（迭代优化）

---

## 7. 风险与限制

1. **计算成本**：许多方法显著增加计算时间
2. **参数敏感性**：需要仔细调参，可能过拟合
3. **相位一致性**：多声道处理需要保持相位关系
4. **实时性**：迭代方法不适合实时处理
5. **理论验证**：部分方法缺乏严格的数学证明

---

## 8. 下一步行动

1. **实现自适应窗口长度**（1-2天）
2. **实现基于AIC的截断**（1天）
3. **测试迭代SSA（基本版本）**（2-3天）
4. **实验验证**（3-5天）
5. **文档更新**（1天）

**优先级**：自适应窗口长度 > 迭代SSA > 鲁棒PCA > 多分辨率SSA