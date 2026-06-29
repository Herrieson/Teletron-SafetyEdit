# 安全感知图像编辑的隐式条件蒸馏方案

## 1. 目标

本方案目标是构建一个看起来像独立模型的图像内容安全自动编辑系统。系统输入一张图片，自动识别其中可能影响内容安全的区域或对象，并输出一张安全编辑后的图片。如果输入图片没有内容安全问题，输出应尽量与原图保持一致。

目标效果：

```text
输入: 原始图片
输出: 安全编辑后的图片
```

典型编辑包括：

- 将不安全对象替换为合理安全对象。
- 擦除不安全对象并补全背景。
- 对局部区域进行安全重绘。
- 对无风险图片保持不变。

本方案不以重新训练视觉理解模型或图像生成模型为目标，而是复用已有开源 VLM 和图像/视频编辑模型的能力，只训练它们之间的中间组件。

## 2. 核心思想

现有能力可以拆成两类：

- VLM 具备图像理解、内容安全判断、风险区域描述和编辑意图生成能力。
- 图像/视频编辑模型具备根据条件指令对图像进行编辑、修复、替换和生成的能力。

直接系统集成的教师流程是：

```text
图片 -> VLM -> 文本编辑提示词 -> 图像/视频编辑模型 -> 修改后的图片
```

本方案希望把上述显式 prompt 通道蒸馏成隐式连续条件通道：

```text
图片 -> VLM hidden states -> 中间组件 -> 图像/视频编辑模型 -> 修改后的图片
```

也就是说，训练目标不是让中间组件学习安全识别能力或图像生成能力，而是让中间组件学习：

```text
VLM 隐状态空间 -> 图像/视频编辑模型条件空间
```

这可以称为 `prompt-channel distillation` 或 `latent condition distillation`。

## 3. 模型边界

对外应封装成一个独立模型：

```python
edited_image = SafetyEditModel(image)
```

内部可以包含多个模块，但这些模块应被统一组织、统一 forward、统一 checkpoint 和统一推理入口管理。

推荐命名边界：

```text
SafetyEditModel
├── semantic_branch        # 冻结 VLM 相关模块
├── visual_branch          # 冻结图像/视频编辑模型的视觉编码模块
├── condition_bridge       # 可训练中间组件
├── edit_backbone          # 冻结图像/视频编辑主干
└── visual_decoder         # 冻结图像/视频解码器
```

架构图中应避免描述为 `Qwen -> WAN` 这种串联形式，而应描述为：

```text
                  Input Image
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
 Semantic Safety Branch      Visual Latent Encoder
        │                           │
        ▼                           │
 Safety Hidden States              │
        │                           │
        ▼                           │
 Condition Bridge                  │
        │                           │
        ├─────────────┐             │
        ▼             ▼             ▼
 Edit Condition    Edit Gate    Image Latents
        │             │             │
        └─────────────┴─────────────┘
                      │
                      ▼
           Safety-aware Edit Backbone
                      │
                      ▼
               Visual Decoder
                      │
                      ▼
             Safe Edited Image
```

## 4. 教师路径

教师路径使用已有的 prompt-based 流程自动生成伪监督数据。

```text
输入图片 x
    │
    ▼
Frozen VLM
    │
    ▼
teacher_prompt = VLM.decode(x)
    │
    ▼
teacher_condition = FrozenEditorTextEncoder(teacher_prompt)
    │
    ▼
y_teacher = FrozenEditor(x, teacher_condition)
```

教师路径可以输出：

- `teacher_prompt`: VLM 生成的编辑指令。
- `teacher_condition`: 编辑模型文本编码器输出的条件 embedding。
- `y_teacher`: 教师编辑后的图片。
- `teacher_mask`: 可选，风险区域 mask。
- `safe_flag`: 是否无需编辑。
- `verifier_score`: 可选，编辑后安全性和质量分数。

教师路径只在训练数据生成或蒸馏训练阶段使用。推理阶段不需要显式生成 prompt。

## 5. 学生路径

学生路径保留 VLM 和编辑模型，但不走文本 prompt 解码和文本 prompt 编码，而是由中间组件直接生成编辑模型可用的条件 embedding。

```text
输入图片 x
    │
    ├── Frozen VLM -> H_vlm
    │                   │
    │                   ▼
    │             ConditionBridge
    │                   │
    │                   ▼
    │              e_student
    │
    └── Frozen Visual Encoder -> z_x

e_student + z_x
    │
    ▼
Frozen Edit Backbone
    │
    ▼
Frozen Visual Decoder
    │
    ▼
y_student
```

形式化表达：

```text
H_vlm = F_vlm(x)
e_student = B(H_vlm)
z_x = E_visual(x)
y_student = D_visual(G_edit(z_x, e_student))
```

其中：

- `F_vlm` 冻结。
- `E_visual` 冻结。
- `G_edit` 冻结。
- `D_visual` 冻结。
- `B` 是唯一主要训练对象。

## 6. 训练对象

主要训练模块：

```text
condition_bridge
```

可选训练模块：

```text
mask_head
noop_gate
condition_resampler
```

冻结模块：

```text
VLM visual encoder
VLM language/model body
VLM text generation head
Editor text encoder
Editor visual encoder
Editor diffusion / DiT / edit backbone
Editor visual decoder
```

如果后续发现只训练中间组件表达能力不足，可作为第二阶段扩展：

- 对编辑模型 cross-attention 加 LoRA。
- 对 VLM 输出投影层加 LoRA。
- 对编辑模型少量条件层微调。

但初始方案应坚持不微调大模型。

## 7. 中间组件设计

中间组件的输入是 VLM hidden states，输出是编辑模型原本 condition 接口需要的 embedding。

输入输出形状示例：

```text
H_vlm:      [B, N_vlm, D_vlm]
e_prompt:  [B, N_txt, D_txt]
e_student: [B, N_txt, D_txt]
```

推荐结构：

```text
VLM hidden states
    │
    ▼
Learnable query tokens
    │
    ▼
Cross-attention / Perceiver Resampler
    │
    ▼
Small Transformer / MLP blocks
    │
    ▼
Linear projection
    │
    ▼
Editor condition embedding
```

伪代码：

```python
class ConditionBridge(nn.Module):
    def __init__(self, num_queries, vlm_dim, mid_dim, editor_dim, editor_tokens):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, mid_dim))
        self.vlm_proj = nn.Linear(vlm_dim, mid_dim)
        self.resampler = CrossAttentionResampler(mid_dim)
        self.blocks = SmallTransformer(mid_dim)
        self.out = nn.Linear(mid_dim, editor_dim)
        self.editor_tokens = editor_tokens

    def forward(self, h_vlm):
        h = self.vlm_proj(h_vlm)
        q = self.queries.expand(h.shape[0], -1, -1)
        c = self.resampler(q, h)
        c = self.blocks(c)
        e = self.out(c)
        return e[:, :self.editor_tokens]
```

如果编辑模型需要额外 mask 或 gate，中间组件可以输出：

```text
{
  "condition": e_student,
  "mask": edit_mask,
  "gate": edit_gate
}
```

## 8. No-op 机制

系统必须支持无风险图片不变化。

推荐增加 `noop_gate`：

```text
gate = NoopHead(H_vlm)
```

推理时：

```text
if gate < threshold:
    return input_image
else:
    return edited_image
```

也可以使用 soft gate：

```text
y_final = gate * y_edit + (1 - gate) * x
```

训练时必须包含大量安全图片：

```text
safe image:
  teacher_prompt = "no edit needed"
  y_teacher = x
  mask = all zeros
  gate target = 0
```

否则模型容易产生过度编辑。

## 9. 蒸馏数据生成

不需要人工标注图片，但需要无标注图片集作为输入分布。

数据生成流程：

```text
for image x in unlabeled_images:
    H_vlm = FrozenVLM(x)
    teacher_prompt = FrozenVLM.decode(H_vlm)
    e_prompt = FrozenEditorTextEncoder(teacher_prompt)
    y_teacher = FrozenEditor(x, e_prompt)

    verifier_result = Verify(x, teacher_prompt, y_teacher)
    if verifier_result.pass:
        save sample
```

建议保存字段：

```json
{
  "image_path": "path/to/input.jpg",
  "teacher_prompt": "replace the unsafe object with a safe alternative",
  "teacher_condition_path": "path/to/condition.pt",
  "teacher_output_path": "path/to/edited.jpg",
  "teacher_mask_path": "path/to/mask.png",
  "safe_flag": false,
  "verifier_score": 0.93,
  "metadata": {
    "teacher_vlm": "qwen-vl-like-model",
    "teacher_editor": "wan-like-editor",
    "created_at": "YYYY-MM-DD"
  }
}
```

如果存储空间允许，建议保存 `teacher_prompt` 和 `teacher_output`，便于人工抽检和失败分析。`teacher_condition` 可以保存为 tensor 文件，减少训练时重复运行文本编码器。

## 10. 自动过滤

由于教师路径会犯错，伪标签需要过滤。

建议过滤器：

- 安全复检：输出图是否仍有内容安全风险。
- 图像质量检测：是否有明显破损、扭曲、伪影。
- 局部性检测：非编辑区域是否被大幅修改。
- 语义一致性检测：主体、场景和整体构图是否基本保留。
- No-op 检测：安全图片是否被误改。

可接受样本：

```text
teacher_output 安全
teacher_output 图像质量合格
teacher_output 与输入图在非风险区域相似
teacher_prompt 与编辑结果一致
```

不接受样本：

```text
风险未移除
替换物不合理
背景大面积错误
安全图片被明显修改
VLM 误判导致不必要编辑
```

## 11. 损失函数

第一阶段建议只训练 condition 对齐：

```text
L_cond = ||B(H_vlm) - e_prompt||^2
```

或：

```text
L_cond = 1 - cosine(B(H_vlm), e_prompt)
```

第二阶段接入冻结编辑模型，加入输出蒸馏：

```text
L_img = distance(y_student, y_teacher)
```

推荐组合：

```text
L_img = L1(y_student, y_teacher)
      + LPIPS(y_student, y_teacher)
      + latent_loss(E_visual(y_student), E_visual(y_teacher))
```

如果有 mask：

```text
L_outside = ||(1 - mask) * (y_student - x)||_1
```

如果有 no-op gate：

```text
L_gate = BCE(gate, edit_needed)
```

对于安全图片：

```text
L_noop = ||y_student - x||_1
```

总损失：

```text
L = lambda_cond * L_cond
  + lambda_img * L_img
  + lambda_outside * L_outside
  + lambda_gate * L_gate
  + lambda_noop * L_noop
```

推荐训练顺序：

```text
Stage 1: L_cond
Stage 2: L_cond + L_img
Stage 3: L_cond + L_img + L_outside + L_gate + L_noop
```

## 12. 推理流程

最终推理不走教师 prompt 路径。

```text
输入图片 x
    │
    ├── Frozen VLM -> H_vlm
    │                   │
    │                   ▼
    │             ConditionBridge
    │                   │
    │                   ├── e_student
    │                   └── gate
    │
    └── Frozen Visual Encoder -> z_x

if gate < threshold:
    return x
else:
    y = FrozenEditor(z_x, e_student)
    return y
```

推理对外表现为：

```python
model = SafetyEditModel.from_pretrained(path)
edited = model(image)
```

可选返回：

```python
{
    "image": edited,
    "edit_score": gate,
    "mask": mask,
    "debug": {
        "used_edit": True
    }
}
```

正式产品路径可以只返回图片，调试路径返回附加信息。

## 13. 与 prompt-based 教师方案的关系

教师方案：

```text
图片 -> VLM -> 文本 prompt -> 编辑模型 -> 图片
```

学生方案：

```text
图片 -> VLM hidden states -> Bridge -> 编辑模型 -> 图片
```

蒸馏后不再依赖：

- 显式编辑 prompt。
- prompt engineering。
- 文本 prompt 解析。
- 推理阶段的教师路径。

但保留：

- VLM 的图像理解能力。
- 编辑模型的图像生成能力。
- 一个统一模型 forward。

## 14. 与普通两模型拼接的区别

普通拼接系统：

```text
用户或系统先调用 VLM
解析文本结果
再调用编辑模型
```

本方案：

```text
一个模型 forward 内完成:
图片 -> semantic branch -> condition bridge -> edit backbone -> 图片
```

关键区别：

- 中间传递连续隐向量，而不是自然语言 prompt。
- 中间组件由蒸馏训练得到，而不是规则拼接。
- 推理入口统一。
- checkpoint 和配置统一。
- 教师路径只在训练阶段使用。

## 15. 在 Teletron 中的落地方向

Teletron 当前已有视频生成训练、VAE/encoder、Wan/TeleAI 模型适配、分布式训练等能力。本方案可以作为新的安全编辑任务加入。

建议新增任务：

```text
safety_edit
safety_inpaint
safe_image_edit
```

建议新增模块：

```text
teletron/models/safety_edit/
├── safety_edit_model.py
├── condition_bridge.py
├── noop_gate.py
└── distill_losses.py
```

建议新增数据集：

```text
teletron/datasets/safety_edit_dataset.py
```

样本字段：

```python
{
    "image": input_image,
    "teacher_condition": e_prompt,
    "teacher_output": y_teacher,
    "teacher_mask": mask,
    "safe_flag": safe_flag,
}
```

建议新增训练脚本：

```text
examples/teleai/pretrain_safety_edit_bridge.py
```

第一阶段训练可以不运行完整编辑模型，只训练：

```text
Bridge(H_vlm) -> teacher_condition
```

第二阶段再接入冻结编辑模型做：

```text
FrozenEditor(image, Bridge(H_vlm)) -> teacher_output
```

## 16. 推荐实施步骤

### 16.1 MVP

先实现教师 pipeline：

```text
图片 -> VLM -> prompt -> 编辑模型 -> 输出图
```

产出伪标签数据。

### 16.2 条件蒸馏

冻结所有大模型，只训练：

```text
ConditionBridge
```

目标：

```text
Bridge(VLM_hidden) ≈ EditorTextEncoder(teacher_prompt)
```

### 16.3 输出蒸馏

加入冻结编辑模型，训练：

```text
FrozenEditor(image, Bridge(VLM_hidden)) ≈ teacher_output
```

### 16.4 No-op 和局部保持

加入安全图片、gate、mask 和非编辑区域保持损失。

### 16.5 统一封装

封装为：

```python
SafetyEditModel(image) -> image
```

训练和推理都不暴露内部 teacher prompt 路径。

## 17. 风险与边界

### 17.1 教师错误会被复制

教师 VLM 或编辑模型错误会形成错误伪标签。必须使用自动 verifier 和抽检机制过滤。

### 17.2 只训练中间组件能力有上限

中间组件只能把 VLM 信息映射到编辑模型已有的条件空间，无法让冻结编辑模型学会全新编辑能力。

### 17.3 安全图过度编辑

如果缺少 no-op 样本和 gate，模型会倾向于总是编辑。必须加入大量安全图片。

### 17.4 中间表示不一定天然对齐

VLM hidden states 与编辑模型 text condition embedding 空间差异大，需要 resampler 或小型 transformer，而不是简单线性层。

### 17.5 图像局部性可能不足

如果编辑模型不支持 mask，仅靠 condition embedding 可能导致大范围改图。建议尽早支持 mask 或 latent blending。

## 18. 方案总结

本方案构建一个安全感知图像编辑模型，输入图片后自动输出安全编辑版本。训练阶段使用已有 prompt-based 流程作为教师：

```text
图片 -> VLM -> prompt -> 编辑模型 -> teacher output
```

学生模型不显式生成 prompt，而是训练一个中间组件：

```text
VLM hidden states -> editor condition embedding
```

训练时冻结 VLM 和图像/视频编辑模型，只训练中间组件、可选 mask head 和 no-op gate。蒸馏目标不是学习视觉理解或图像生成能力，而是学习两个冻结大模型之间的隐式条件接口。

最终系统对外表现为一个独立模型：

```text
图片 -> SafetyEditModel -> 安全编辑后的图片
```

该方案兼顾工程可行性、训练成本和统一模型形态，是从现有 VLM 与图像/视频模型能力出发构建安全编辑模型的可行路径。
