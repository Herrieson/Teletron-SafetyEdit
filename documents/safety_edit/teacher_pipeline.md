# Safety Edit Teacher Pipeline

## 目标

教师流水线用于生成安全编辑蒸馏训练需要的伪监督数据。它本身不训练模型，而是编排本地冻结模型：

```text
input image
    -> local VLM safety/edit planner
    -> teacher prompt
    -> local editor text encoder
    -> teacher condition
    -> local image/video editor
    -> teacher output
    -> verifier/filter
    -> manifest + assets
```

第一阶段学生训练只需要：

```text
vlm_hidden -> ConditionBridge -> teacher_condition
```

因此教师流水线除了保存编辑后的图片，也要保存 `vlm_hidden` 和 `teacher_condition`。

## 模块边界

教师流水线代码位于：

```text
teletron/safety_edit/teacher_pipeline/
├── adapters.py      # 内置 smoke-test adapter 与 adapter 接口示例
├── loader.py        # config 与本地 factory/class 动态加载
├── pipeline.py      # 样本处理主流程
├── run.py           # CLI 入口
├── schemas.py       # TeacherPlan / EditorResult / VerifierResult
└── writer.py        # manifest 与资产落盘
```

真实本地模型通过 config 中的 `target` 接入，不在 Teletron 中硬编码模型路径或推理库。

## Adapter 接口

### VLM adapter

VLM adapter 需要实现：

```python
def plan(self, image_path, image) -> TeacherPlan:
    ...
```

返回：

```python
TeacherPlan(
    teacher_prompt="replace unsafe object with a harmless alternative ...",
    safe_flag=False,
    risk_type="weapon",
    risk_description="...",
    edit_region={"type": "bbox", "bbox": [x1, y1, x2, y2]},
    vlm_hidden=hidden_tensor,
    raw_response=raw_model_output,
)
```

安全图片应返回：

```python
TeacherPlan(
    teacher_prompt="no edit needed",
    safe_flag=True,
    risk_type="none",
    no_edit_reason="No unsafe content is visible.",
    vlm_hidden=hidden_tensor,
)
```

### Editor adapter

Editor adapter 需要实现：

```python
def edit(self, image_path, image, plan) -> EditorResult:
    ...
```

返回：

```python
EditorResult(
    teacher_condition=editor_text_condition,
    teacher_output=edited_image,
    teacher_mask=optional_mask,
)
```

`teacher_condition` 应该是编辑模型文本编码器输出的条件 embedding。第一阶段训练会直接拟合这个 tensor。

### Verifier adapter

Verifier adapter 需要实现：

```python
def verify(self, image_path, image, plan, editor_result) -> VerifierResult:
    ...
```

返回：

```python
VerifierResult(
    accepted=True,
    verifier_score=0.93,
    reject_reasons=[],
)
```

第一版至少建议做：

- 输出图安全复检。
- 安全图 no-op 复检。
- 输出图坏图/缺失检查。
- 非风险区域变化过大检查。

## 数据输出

默认输出目录：

```text
teacher_data/
├── manifest.jsonl
├── images/
├── vlm_hidden/
├── conditions/
├── outputs/
├── masks/
└── logs/
```

`manifest.jsonl` 每行一个样本：

```json
{
  "id": "sample_id",
  "image_path": "images/sample.jpg",
  "teacher_prompt": "replace unsafe object with a harmless alternative",
  "teacher_condition_path": "conditions/sample.pt",
  "teacher_output_path": "outputs/sample.jpg",
  "teacher_mask_path": "masks/sample.png",
  "vlm_hidden_path": "vlm_hidden/sample.pt",
  "safe_flag": false,
  "risk_type": "weapon",
  "risk_description": "...",
  "verifier_score": 0.93,
  "accepted": true,
  "reject_reasons": [],
  "metadata": {}
}
```

## Smoke Test

内置静态 adapter 可用于验证落盘链路：

```bash
python -m teletron.safety_edit.teacher_pipeline.run \
  --config examples/teleai/config/safety_edit_teacher_static.yaml \
  --input /path/to/images \
  --output-dir /tmp/safety_edit_teacher_static \
  --limit 4
```

这个配置使用 `StaticVLMTeacher`、`CopyEditorTeacher` 和 `PixelDiffVerifier`，不会调用真实大模型。

## 接入真实本地模型

新增一个本地 adapter 文件，例如：

```python
from teletron.safety_edit.teacher_pipeline.schemas import EditorResult, TeacherPlan

class LocalQwenVLMTeacher:
    def __init__(self, model_path, device="cuda"):
        ...

    def plan(self, image_path, image):
        ...
        return TeacherPlan(...)

class LocalWanEditorTeacher:
    def __init__(self, model_path, device="cuda"):
        ...

    def edit(self, image_path, image, plan):
        ...
        return EditorResult(...)
```

然后在 YAML 中配置：

```yaml
vlm:
  target: my_project.safety_edit_adapters:LocalQwenVLMTeacher
  params:
    model_path: /path/to/qwen
    device: cuda

editor:
  target: my_project.safety_edit_adapters:LocalWanEditorTeacher
  params:
    model_path: /path/to/editor
    device: cuda
```

## 后续训练接口

后续 `SafetyEditDataset` 应直接读取 `manifest.jsonl`，第一阶段读取：

```text
vlm_hidden_path
teacher_condition_path
safe_flag
```

第二阶段再读取：

```text
image_path
teacher_output_path
teacher_mask_path
```

