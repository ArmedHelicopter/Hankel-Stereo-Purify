# 术语简释（Glossary）

面向快速查阅；**定义与验收口径**以 [`prd.md`](prd.md)、[`software_design.md`](software_design.md) 为准。

| 术语 | 简释 |
|------|------|
| **MSSA** | 多通道奇异谱分析（Multichannel Singular Spectrum Analysis）：对多路（如立体声）联合构造轨迹矩阵再做奇异值分解与子空间截断的一类方法。 |
| **Hankel 嵌入** | 将一维序列排成 Hankel 结构矩阵（反斜对角元素相等），把时序问题转为矩阵代数（模块 A）。 |
| **联合块矩阵** | 左右声道 Hankel 矩阵水平拼接得到的块矩阵（记作 X_total），用于立体声相干约束（模块 B，**F-03**）。 |
| **截断 SVD** | 保留前 k 个奇异方向或按能量阈值截断，抑制噪声子空间（模块 C，**F-04**）。 |
| **对角平均** | 将低秩近似矩阵沿反斜对角线平均回一维序列（模块 D）。 |
| **W-correlation** | 基于加权相关对奇异分量分组/聚类的工具；实现与阈值见 `strategies/grouping` 与配置。 |
| **OLA** | 重叠相加（Overlap-Add）：分帧处理后在输出端加窗叠接以消除帧边界不连续（**F-02**）。 |
| **`process_frame`** | 单帧上 A→B→C→D 的编排入口；整文件由门面层分帧循环调用（见 software_design §3）。 |
| **Facade / 门面** | 如 `AudioPurifier`：路径校验、构造单帧去噪函数、组合 OLA 与 I/O（见 software_design §3.3）。 |
| **NF-01 / NF-02** | 峰值内存约束；零拷贝 Hankel 构造与数值基准对齐（见 prd §4）。 |
