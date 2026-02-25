# Region-Reg-DPO：面向视频补全的分区正则化直接偏好优化

## 完整数学推导（终版）

---

## 一、问题设定与符号定义

### 1.1 三区 Mask 分解

将视频帧空间分为三个不相交区域：

- **洞内区域** R_hole，对应二值 mask $M_h$（需要生成的核心区域）
- **边界带** R_bd，对应二值 mask $M_b$（由 $M_h$ 膨胀 $k$ 像素再减去 $M_h$ 得到）
- **上下文区域** R_ctx，对应二值 mask $M_c$（未遮挡的原始像素区域）

满足空间完整划分：

$$M_h + M_b + M_c = 1$$

边界带生成：

$$M_b = \text{Dilate}(M_h, k) - M_h, \quad \text{建议 } k = 5 \sim 15 \text{ 像素}$$

### 1.2 偏好对定义

训练时通过 `create_random_shape_with_random_motion` 生成随机 mask，于是自然地：

- **正样本（win）**：原始未遮挡视频帧（Ground Truth），对应噪声目标 $\epsilon^w$
- **负样本（lose）**：模型补全后的结果，对应噪声目标 $\epsilon^l$

### 1.3 两组独立的区域权重

Region-Reg-DPO 使用两组独立的区域权重，它们**分别作用于损失函数的不同位置**：

| 权重 | 所在位置 | 物理含义 | 是否受 DGR 调制 |
|------|---------|---------|-------------|
| $\alpha_r$ | DPO 损失（$\sigma$ 函数**内部**） | 区域对偏好学习的贡献 | **是**（随 DGR 衰减） |
| $\rho_r$ | SFT 损失（$\sigma$ 函数**外部**） | 区域对正样本锚定的强度 | **否**（常数，永不衰减） |

**为什么不冗余**：$\alpha_r$ 乘以 DGR 后会随训练衰减直至归零，$\rho_r$ 始终不变。两者在训练不同阶段扮演不同角色（详见第六节）。

---

## 二、区域加权 DPO 损失

### 2.1 区域加权 KL 差分

简记 $\epsilon_\theta^w = \epsilon_\theta(x_t^w, t)$，$\epsilon_\theta^l = \epsilon_\theta(x_t^l, t)$，参考模型同理。

对于 win 样本：

$$\Delta_{\text{region}}^{w} = \sum_{r \in \{h,b,c\}} \alpha_r \Big( \|M_r \odot (\epsilon^w - \epsilon_\theta^w)\|_2^2 - \|M_r \odot (\epsilon^w - \epsilon_{\text{ref}}^w)\|_2^2 \Big)$$

对于 lose 样本：

$$\Delta_{\text{region}}^{l} = \sum_{r \in \{h,b,c\}} \alpha_r \Big( \|M_r \odot (\epsilon^l - \epsilon_\theta^l)\|_2^2 - \|M_r \odot (\epsilon^l - \epsilon_{\text{ref}}^l)\|_2^2 \Big)$$

### 2.2 Region-DPO 损失

令 $\beta' = \beta T \omega(\lambda_t)$，定义：

$$s_{\text{region}}(\theta) = \Delta_{\text{region}}^{w} - \Delta_{\text{region}}^{l}$$

$$S = -\beta' \cdot s_{\text{region}}(\theta)$$

$$\mathcal{L}_{\text{Region-DPO}}(\theta) = -\mathbb{E}\big[\log \sigma(S)\big]$$

---

## 三、梯度推导（5 步）

### Step 1：对 $\log \sigma$ 求导

$$\nabla_\theta \mathcal{L} = -\mathbb{E}\big[(1 - \sigma(S)) \cdot \nabla_\theta S\big]$$

> 依据：$\frac{d}{dx}\log\sigma(x) = 1 - \sigma(x)$，链式法则。

### Step 2：$S$ 对 $\theta$ 的梯度

$$\nabla_\theta S = -\beta' \cdot \nabla_\theta s_{\text{region}}(\theta)$$

参考模型冻结，故 $\epsilon_{\text{ref}}$ 对 $\theta$ 无梯度：

$$\nabla_\theta s_{\text{region}} = \sum_r \alpha_r \Big[\nabla_\theta \|M_r \odot (\epsilon^w - \epsilon_\theta^w)\|^2 - \nabla_\theta \|M_r \odot (\epsilon^l - \epsilon_\theta^l)\|^2\Big]$$

### Step 3：Mask 加权 L2 范数的梯度

对任意二值 mask $M_r$（$M_{r,i} \in \{0, 1\}$，因此 $M_{r,i}^2 = M_{r,i}$）：

$$\nabla_\theta \|M_r \odot (\epsilon - \epsilon_\theta)\|^2 = \nabla_\theta \sum_i M_{r,i} (\epsilon_i - \epsilon_{\theta,i})^2 = -2 \sum_i M_{r,i}(\epsilon_i - \epsilon_{\theta,i}) \nabla_\theta \epsilon_{\theta,i}$$

写成向量形式：

$$\boxed{\nabla_\theta \|M_r \odot (\epsilon - \epsilon_\theta)\|^2 = -2\big(M_r \odot (\epsilon - \epsilon_\theta)\big)^T \nabla_\theta \epsilon_\theta}$$

### Step 4：代入正负样本

$$\nabla_\theta s_{\text{region}} = \sum_r \alpha_r \Big[-2\big(M_r \odot (\epsilon^w - \epsilon_\theta^w)\big)^T \nabla_\theta \epsilon_\theta^w + 2\big(M_r \odot (\epsilon^l - \epsilon_\theta^l)\big)^T \nabla_\theta \epsilon_\theta^l\Big]$$

### Step 5：合并完整梯度

定义**区域 DPO 梯度比率**：

$$\text{DGR}_{\text{region}} = \beta'(1 - \sigma(S)) = \frac{\beta'}{1 + e^{S}} = \frac{\beta'}{1 + e^{-\beta' \cdot s_{\text{region}}(\theta)}}$$

最终梯度：

$$\boxed{\nabla_\theta \mathcal{L}_{\text{Region-DPO}} = -\mathbb{E}\bigg[2 \cdot \text{DGR}_{\text{region}} \cdot \sum_r \alpha_r \Big\{\big(M_r \odot (\epsilon^w - \epsilon_\theta^w)\big)^T \nabla_\theta \epsilon_\theta^w - \big(M_r \odot (\epsilon^l - \epsilon_\theta^l)\big)^T \nabla_\theta \epsilon_\theta^l\Big\}\bigg]}$$

**梯度结构**：

- 区域 $r$ 正样本系数：$2 \cdot \alpha_r \cdot \text{DGR}_{\text{region}}$
- 区域 $r$ 负样本系数：$2 \cdot \alpha_r \cdot \text{DGR}_{\text{region}}$
- 正负样本**共享同一个 DGR**

---

## 四、缺陷分析

### 缺陷 1（继承自原始 DPO）：DGR 快速衰减 → 梯度消失

训练使 $s_{\text{region}}(\theta)$ 趋向负值，DGR 指数级衰减。

**数值验证**（$\beta' = 1000$）：

| $s_{\text{region}}$ | $e^{-\beta' \cdot s}$ | $\text{DGR} = \frac{1000}{1 + e^{-\beta' \cdot s}}$ | 相对初始值 |
|-----|-----|-----|-----|
| 0 | $e^0 = 1$ | $1000 / 2 = 500$ | 100% |
| −0.005 | $e^5 = 148.41$ | $1000 / 149.41 = 6.69$ | 1.34% |
| −0.01 | $e^{10} = 22026.47$ | $1000 / 22027.47 = 0.0454$ | 0.009% |

> 验证 $s = -0.005$：$-\beta' \cdot s = -1000 \times (-0.005) = 5$。$e^5 = 148.413$。$1000 / (1 + 148.413) = 1000 / 149.413 = 6.693$。$6.693 / 500 = 1.34\%$。✓
>
> 验证 $s = -0.01$：$-\beta' \cdot s = 10$。$e^{10} = 22026.466$。$1000 / 22027.466 = 0.04540$。$0.04540 / 500 = 0.00908\%$。✓

**核心问题**：所有区域共享同一个 DGR。$s_{\text{region}}$ 仅变化 0.01，DGR 从 500 降至 0.045（衰减 11000 倍）。$\alpha_r$ 的权重差异在此衰减面前无意义。

### 缺陷 2（继承自原始 DPO）：无绝对分布约束 → "作弊路径"

记 $A_r^w = \|M_r \odot (\epsilon^w - \epsilon_\theta^w)\|^2$，$A_r^l = \|M_r \odot (\epsilon^l - \epsilon_\theta^l)\|^2$。

DPO 只约束 $\sum_r \alpha_r (A_r^w - A_r^l)$ 的差值，不约束各 $A_r$ 的绝对大小：

| 路径 | $A_r^w$ | $A_r^l$ | 差值 $A_r^w - A_r^l$ | 模型状态 |
|------|---------|---------|-----|------|
| 理想路径 | 10 → 5 | 10 → 15 | −10 | 健康 |
| 作弊路径 | 10 → 500 | 10 → 510 | −10 | 已崩溃 |

两条路径的差值相同，DPO 损失无法区分。高维空间中误差自然倾向膨胀（体积集中现象），模型更容易走作弊路径。

### 缺陷 3（新发现）：区域优化需求不对称被共享 DGR 掩盖

| 区域 | 优化目标 | 面积 | 收敛速度 | 对持续梯度的需求 |
|------|---------|------|---------|------------|
| 洞内 $M_h$ | 生成合理新内容 | 大 | 快（信号强） | 中等 |
| 边界 $M_b$ | 无缝融合过渡 | 小 | **慢**（窄带，信号弱） | **最高** |
| 上下文 $M_c$ | 保持不变 | 大 | 不需要 | 最低 |

**矛盾**：洞内区域面积大、信号强 → 最先学到偏好差异 → 驱动 $s_{\text{region}}$ 快速变负 → DGR 全局衰减 → 边界区域尚未充分优化就失去梯度。

**边界是最需要持续优化的区域，却最容易被洞内的快速收敛拖垮。**

---

## 五、梯度修复

### 5.1 设计原则

遵循 Reg-DPO 方法论：先改梯度，再反推损失。

目标：为正样本在每个区域添加**不随 DGR 衰减的常数项** $\rho_r > 0$。

**关键约束**：$\rho_r$ 必须是不依赖 DGR 的常数。若令 $\rho_r = f(\text{DGR})$，则当 DGR → 0 时 $\rho_r$ 也归零，SFT 保护失效，修复无意义。

### 5.2 修改后的梯度

$$\boxed{\nabla_\theta \mathcal{L}_{\text{修改}} = -\mathbb{E}\bigg[\sum_r 2\Big\{(\alpha_r \cdot \text{DGR} + \rho_r) \cdot \big(M_r \odot (\epsilon^w - \epsilon_\theta^w)\big)^T \nabla_\theta \epsilon_\theta^w - \alpha_r \cdot \text{DGR} \cdot \big(M_r \odot (\epsilon^l - \epsilon_\theta^l)\big)^T \nabla_\theta \epsilon_\theta^l\Big\}\bigg]}$$

各区域的有效梯度系数：

| 区域 $r$ | 正样本系数 | 负样本系数 | DGR → 0 时正样本系数 |
|---------|---------|---------|----------------|
| 洞内 | $\alpha_h \cdot \text{DGR} + \rho_h$ | $\alpha_h \cdot \text{DGR}$ | $\rho_h > 0$ |
| 边界 | $\alpha_b \cdot \text{DGR} + \rho_b$ | $\alpha_b \cdot \text{DGR}$ | $\rho_b > 0$（**最大**） |
| 上下文 | $\alpha_c \cdot \text{DGR} + \rho_c$ | $\alpha_c \cdot \text{DGR}$ | $\rho_c \approx 0$ |

---

## 六、反推损失函数

### 6.1 拆解梯度

将修改后的梯度拆为两部分。

**Part 1**（原始 Region-DPO 梯度）：

$$-\mathbb{E}\bigg[2 \cdot \text{DGR} \cdot \sum_r \alpha_r \Big\{\big(M_r \odot (\epsilon^w - \epsilon_\theta^w)\big)^T \nabla_\theta \epsilon_\theta^w - \big(M_r \odot (\epsilon^l - \epsilon_\theta^l)\big)^T \nabla_\theta \epsilon_\theta^l\Big\}\bigg]$$

这正是 $\nabla_\theta \mathcal{L}_{\text{Region-DPO}}$。✓

**Part 2**（额外项）：

$$-\mathbb{E}\bigg[\sum_r 2\rho_r \cdot \big(M_r \odot (\epsilon^w - \epsilon_\theta^w)\big)^T \nabla_\theta \epsilon_\theta^w\bigg]$$

### 6.2 识别 Part 2

由 Step 3：

$$\nabla_\theta \|M_r \odot (\epsilon^w - \epsilon_\theta^w)\|^2 = -2\big(M_r \odot (\epsilon^w - \epsilon_\theta^w)\big)^T \nabla_\theta \epsilon_\theta^w$$

因此：

$$-2\rho_r \cdot \big(M_r \odot (\epsilon^w - \epsilon_\theta^w)\big)^T \nabla_\theta \epsilon_\theta^w = \rho_r \cdot \nabla_\theta \|M_r \odot (\epsilon^w - \epsilon_\theta^w)\|^2$$

> 验证：$\rho_r \cdot [-2(M_r \odot (\epsilon^w - \epsilon_\theta^w))^T \nabla_\theta \epsilon_\theta^w] = -2\rho_r \cdot (M_r \odot (\epsilon^w - \epsilon_\theta^w))^T \nabla_\theta \epsilon_\theta^w$。左边 = 右边。✓

所以 Part 2 = $\nabla_\theta \Big[\sum_r \rho_r \cdot \|M_r \odot (\epsilon^w - \epsilon_\theta^w)\|^2\Big]$，即**分区加权 SFT 损失的梯度**。

### 6.3 最终损失函数

$$\boxed{\mathcal{L}_{\text{Region-Reg-DPO}}(\theta) = \underbrace{-\mathbb{E}\Big[\log \sigma\big(-\beta' \cdot s_{\text{region}}(\theta)\big)\Big]}_{\text{Region-DPO 损失}} + \underbrace{\mathbb{E}\bigg[\sum_{r \in \{h,b,c\}} \rho_r \cdot \|M_r \odot (\epsilon^w - \epsilon_\theta(x_t^w, t))\|^2\bigg]}_{\text{分区 SFT 正则化}}}$$

其中：

$$s_{\text{region}}(\theta) = \sum_r \alpha_r \Big[\big(\|M_r \odot (\epsilon^w - \epsilon_\theta^w)\|^2 - \|M_r \odot (\epsilon^w - \epsilon_{\text{ref}}^w)\|^2\big) - \big(\|M_r \odot (\epsilon^l - \epsilon_\theta^l)\|^2 - \|M_r \odot (\epsilon^l - \epsilon_{\text{ref}}^l)\|^2\big)\Big]$$

---

## 七、$\alpha_r$ 与 $\rho_r$ 的不冗余性

### 7.1 位置不同

- $\alpha_r$ 在 $\sigma$ 函数内部，通过 $s_{\text{region}}$ 影响 DGR，进而影响整个 DPO 梯度
- $\rho_r$ 在 $\sigma$ 函数外部，直接加在正样本梯度系数上

### 7.2 动态行为不同

训练过程中正样本在区域 $r$ 的总梯度系数为 $\alpha_r \cdot \text{DGR} + \rho_r$。

| 训练时期 | $\text{DGR}$（$\beta'=1000$） | $\alpha_r \cdot \text{DGR}$（$\alpha_r=1$） | $\rho_r$（$=100$） | 总系数 | 主导 |
|---------|-----|-----|-----|-----|-----|
| 初始（$s \approx 0$） | 500 | 500 | 100 | 600 | DPO 偏好学习 |
| 早期（$s = -0.005$） | 6.69 | 6.69 | 100 | 107 | 过渡 |
| 中后期（$s = -0.01$） | 0.045 | 0.045 | 100 | 100 | **SFT 锚定** |

> 验证初始：$\alpha_r \cdot \text{DGR} = 1 \times 500 = 500$，总系数 $= 500 + 100 = 600$，DPO 占比 $500/600 = 83\%$。✓
>
> 验证中后期：$\alpha_r \cdot \text{DGR} = 1 \times 0.045 = 0.045$，总系数 $\approx 100$，SFT 占比 $\approx 100\%$。✓

**若 $\rho_r$ 也依赖 DGR**（如错误设计 $\rho_r = \gamma_r \cdot \text{DGR}$）：

| 训练时期 | $\alpha_r \cdot \text{DGR}$ | $\gamma_r \cdot \text{DGR}$（$\gamma_r=1$） | 总系数 | DGR → 0 时 |
|---------|-----|-----|-----|-----|
| 初始 | 500 | 500 | 1000 | — |
| 中后期 | 0.045 | 0.045 | 0.09 | **全部归零，保护失效** |

两者合并等价于 $(\alpha_r + \gamma_r) \cdot \text{DGR}$，只是换了一个 DPO 权重，SFT 项并不存在。

---

## 八、三重稳定机制

### 机制 1：分区持续梯度信号

当 DGR → 0 时：

| 区域 | 正样本系数 | 负样本系数 | 效果 |
|------|---------|---------|------|
| 洞内 | → $\rho_h$ | → 0 | 持续学习重建 |
| 边界 | → $\rho_b$（**最大**） | → 0 | **最强锚定，保护融合边界** |
| 上下文 | → $\rho_c \approx 0$ | → 0 | 不干扰 |

### 机制 2：阻断"作弊路径"

SFT 项 $\rho_r \cdot \|M_r \odot (\epsilon^w - \epsilon_\theta^w)\|^2$ 直接惩罚 $A_r^w$ 的膨胀：

| 路径 | $A_r^w$ | SFT 惩罚 | 能否通过 |
|------|---------|----------|-------|
| 理想路径 | 5 | $5\rho_r$ | ✓ |
| 作弊路径 | 500 | $500\rho_r$（100 倍惩罚） | ✗ |

一旦 $A_r^w$ 被锚定，模型只能走理想路径：$A_r^w$ ↓，$A_r^l$ ↑。

### 机制 3：分区偏移控制

| 区域 | 允许的偏移 | $\rho_r$ 强度 | 原因 |
|------|---------|------------|------|
| 洞内 | 适度（向更真实方向） | 中等 | 需要生成自由度 |
| 边界 | **几乎不允许** | **最大** | 必须与原始像素无缝衔接 |
| 上下文 | 不需要 | ≈ 0 | 由 Blending 步骤保证 |

---

## 九、与原始 Reg-DPO 的统一性

**命题**：当 $\alpha_h = \alpha_b = \alpha_c = 1$ 且 $\rho_h = \rho_b = \rho_c = \rho$ 时，Region-Reg-DPO 退化为标准 Reg-DPO。

**验证 DPO 项**：

三区不重叠且完整划分（$M_{h,i} + M_{b,i} + M_{c,i} = 1$ 对所有像素 $i$），因此：

$$\sum_r 1 \cdot \|M_r \odot (\epsilon - \epsilon_\theta)\|^2 = \sum_r \sum_i M_{r,i}(\epsilon_i - \epsilon_{\theta,i})^2 = \sum_i (\epsilon_i - \epsilon_{\theta,i})^2 = \|\epsilon - \epsilon_\theta\|^2 \quad \checkmark$$

> 验证：每个像素 $i$ 恰好属于一个区域，$M_{h,i} + M_{b,i} + M_{c,i} = 1$，所以 $\sum_r M_{r,i} = 1$。✓

$s_{\text{region}}$ 退化为标准 $s(\theta)$。✓

**验证 SFT 项**：

$$\sum_r \rho \cdot \|M_r \odot (\epsilon^w - \epsilon_\theta)\|^2 = \rho \cdot \sum_r \|M_r \odot (\epsilon^w - \epsilon_\theta)\|^2 = \rho \cdot \|\epsilon^w - \epsilon_\theta\|^2 \quad \checkmark$$

**Region-Reg-DPO 是 Reg-DPO 的严格泛化。** ✓

---

## 十、参数推荐

### 10.1 两组区域权重

| 参数 | 推荐值 | 作用位置 | 说明 |
|------|-------|---------|------|
| $\alpha_h$ | 1.0 | DPO（$\sigma$ 内） | 洞内偏好学习基准 |
| $\alpha_b$ | 1.5 ~ 2.0 | DPO（$\sigma$ 内） | 边界偏好学习优先 |
| $\alpha_c$ | 0.05 ~ 0.1 | DPO（$\sigma$ 内） | 上下文几乎不参与偏好 |
| $\rho_h$ | 50 ~ 150 | SFT（$\sigma$ 外） | 洞内中等锚定 |
| $\rho_b$ | 100 ~ 250 | SFT（$\sigma$ 外） | 边界最强锚定 |
| $\rho_c$ | 0 ~ 10 | SFT（$\sigma$ 外） | 上下文由 Blending 保护 |

> $\rho_r$ 的量级参考：训练初始时 DGR = $\beta'/2$。当 $\beta'=1000$ 时 DGR = 500，将 $\rho_b$ 设为 100~250 意味着 SFT 在初始时占正样本系数的 17%~33%，在 DGR 衰减后占 100%。

### 10.2 其他超参数

| 参数 | 推荐值 | 说明 |
|------|-------|------|
| $\beta$ | 500 ~ 1000 | 偏好锐度，与 Reg-DPO 一致 |
| 微调策略 | LoRA (rank=16, $\alpha$=32) | 仅微调主 UNet，冻结 BrushNet |
| 学习率 | 1e-6 ~ 5e-6 | Reg-DPO 验证的稳定范围，需结合 $\beta'$ 调整 |
| 膨胀核 $k$ | 5 ~ 15 像素 | 边界带宽度 |
| 训练帧数 | 22 帧 | 与 DiffuEraser 推理 clip 长度一致 |

**关于学习率**：Stage 3 内 DPO 项与 SFT 项共享同一个 lr。两者的相对强度由 $\alpha_r \cdot \text{DGR}$ 和 $\rho_r$ 的比值决定，不存在"为两项设不同学习率"的问题。

### 10.3 调参路线

| 轮次 | 配置 | 观察目标 |
|------|------|---------|
| 第一轮 | $\alpha_r$ 分区，$\rho_r = 0$ | Region-DPO 是否稳定（DGR 曲线） |
| 第二轮 | 加入统一 $\rho = 100$ | 训练是否稳定，补全质量 |
| 第三轮 | 分区 $\rho$（$\rho_b=200, \rho_h=100, \rho_c=0$） | 边界质量是否改善 |
| 第四轮 | 微调 $\alpha$ 和 $\rho$ 比例 | 寻找最优配置 |

---

## 十一、完整损失展开式（可直接实现）

$$\mathcal{L}(\theta) = -\mathbb{E}\Bigg[\log \sigma\bigg(-\beta' \sum_r \alpha_r \Big[\big(\|M_r \odot (\epsilon^w - \epsilon_\theta^w)\|^2 - \|M_r \odot (\epsilon^w - \epsilon_{\text{ref}}^w)\|^2\big) - \big(\|M_r \odot (\epsilon^l - \epsilon_\theta^l)\|^2 - \|M_r \odot (\epsilon^l - \epsilon_{\text{ref}}^l)\|^2\big)\Big]\bigg)\Bigg]$$

$$+ \quad \mathbb{E}\bigg[\sum_r \rho_r \cdot \|M_r \odot (\epsilon^w - \epsilon_\theta(x_t^w, t))\|^2\bigg]$$

其中：

- $\beta' = \beta T \omega(\lambda_t)$
- $r \in \{h, b, c\}$，对应洞内、边界、上下文
- $M_h + M_b + M_c = 1$（三区不重叠，完整划分）
- $\alpha_r$：DPO 区域权重（在 $\sigma$ 内，随 DGR 衰减）
- $\rho_r$：SFT 区域权重（在 $\sigma$ 外，常数不衰减）

---

## 十二、总结

### 演进路径

| 方法 | DPO 项 | SFT 项 | DGR → 0 时正样本梯度 |
|------|--------|--------|-----------------|
| 标准 DPO | 全像素 | 无 | → 0（全部消失） |
| Reg-DPO | 全像素 | 全局 $r$（常数） | → $r > 0$（统一保护） |
| Region-DPO | $\alpha_r$ 加权 | 无 | → 0（仍然消失） |
| **Region-Reg-DPO** | $\alpha_r$ 加权 | **$\rho_r$（常数，分区独立）** | → **$\rho_r > 0$（分区独立保护）** |

### 核心贡献

1. 完成了从 Region-DPO 到 Region-Reg-DPO 的完整 5 步梯度推导和损失反推
2. 发现了第三缺陷：区域间优化需求不对称被共享 DGR 掩盖（洞内快速收敛拖垮边界）
3. 提出 $\alpha_r$ / $\rho_r$ 双权重体系：$\alpha_r$ 控制偏好学习（随 DGR 衰减），$\rho_r$ 控制锚定保护（永不衰减），两者位于损失函数不同位置，不冗余
4. 证明了 Region-Reg-DPO 是 Reg-DPO 的严格泛化
