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
python3 -m teletron.safety_edit.teacher_pipeline.run \
  --config examples/teleai/config/safety_edit_teacher_static.yaml \
  --input /path/to/images \
  --output-dir /tmp/safety_edit_teacher_static \
  --limit 4
```

这个配置使用 `StaticVLMTeacher`、`CopyEditorTeacher` 和 `PixelDiffVerifier`，不会调用真实大模型。

## Qwen 本地教师配置

当前已提供 Qwen 组合的 adapter：

```text
VLM:
  teletron.safety_edit.teacher_pipeline.qwen_adapters:LocalQwen36VLMTeacher

Editor:
  teletron.safety_edit.teacher_pipeline.qwen_adapters:LocalQwenImageEditTeacher
```

示例配置：

```text
examples/teleai/config/safety_edit_teacher_qwen.yaml
```

运行示例：

```bash
python3 -m teletron.safety_edit.teacher_pipeline.run \
  --config examples/teleai/config/safety_edit_teacher_qwen.yaml \
  --input /path/to/images \
  --output-dir /path/to/teacher_data_qwen \
  --limit 8
```

### LocalQwen36VLMTeacher

该 adapter 使用本地 `transformers` 加载 `Qwen/Qwen3.6-27B`，输出：

```text
teacher_prompt
safe_flag
risk_type
risk_description
edit_region
vlm_hidden
```

默认提示词要求 VLM 只返回 JSON。解析失败时，会把原始文本作为 `teacher_prompt`，并在 `raw_response.parse_error` 中记录错误。

关键配置：

```yaml
vlm:
  target: teletron.safety_edit.teacher_pipeline.qwen_adapters:LocalQwen36VLMTeacher
  params:
    model_path: Qwen/Qwen3.6-27B
    device_map: auto
    dtype: bfloat16
    max_new_tokens: 768
    extract_hidden: true
    hidden_layer: -1
    hidden_strategy: all
```

`hidden_strategy` 可选：

```text
all   保存 [N, D] token hidden states
mean  保存 pooled hidden states
last  保存最后 token hidden states
```

第一阶段蒸馏建议先用 `all`，如果磁盘或显存压力太大，再改成 `mean` 或后续自定义图像 token 选择策略。

### LocalQwenImageEditTeacher

该 adapter 使用 Diffusers 加载 `Qwen/Qwen-Image-Edit`。它会：

```text
teacher_prompt -> pipeline.encode_prompt(...) -> teacher_condition
image + teacher_prompt -> pipeline(...) -> teacher_output
```

`teacher_condition` 是 best-effort 提取：不同 diffusers 版本的 `encode_prompt` 签名可能不同，adapter 会尝试几组常见参数，并把成功返回的 tensor detach 到 CPU 后保存。如果当前 pipeline 没有 `encode_prompt` 或签名不兼容，会保存一个包含 warning/error 的字典，方便后续排查。

关键配置：

```yaml
editor:
  target: teletron.safety_edit.teacher_pipeline.qwen_adapters:LocalQwenImageEditTeacher
  params:
    model_path: Qwen/Qwen-Image-Edit
    device: cuda
    dtype: bfloat16
    num_inference_steps: 50
    true_cfg_scale: 4.0
    negative_prompt: " "
    seed: 0
    skip_safe_editor: true
    extract_condition: true
```

`skip_safe_editor: true` 表示当 VLM 判断 `safe_flag=true` 时，不运行完整编辑采样，直接把原图作为 `teacher_output`。它仍会尝试保存 `"no edit needed"` 的 `teacher_condition`，用于训练 no-op/gate 相关能力。

## 两张或四张 H100 的建议跑法

第一版建议优先稳定，不要过早做并发。不要在同一个进程里同时常驻 Qwen3.6-27B 和 Qwen-Image-Edit；两者一起加载很容易把单张 H100 塞满。

```text
2 张 H100:
  推荐: 两阶段运行
    stage 1: 只跑 Qwen3.6 VLM，生成 teacher_prompt/safe_flag/vlm_hidden
    stage 2: 释放 VLM 后只跑 Qwen-Image-Edit，读取 stage 1 manifest 补 condition/output

4 张 H100:
  GPU 0-1: Qwen3.6-27B
  GPU 2:   Qwen-Image-Edit
  GPU 3:   verifier 或对照 editor
```

如果 Qwen3.6 和 Qwen-Image-Edit 同进程争显存，优先拆成两个阶段或两个 worker 进程，而不是在一个进程里硬塞两个大模型。

### 两阶段命令

Stage 1 只跑 VLM：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python3 -m teletron.safety_edit.teacher_pipeline.run \
  --config examples/teleai/config/safety_edit_teacher_qwen_vlm_stage.yaml \
  --input /workplace/hyx/safety_edit_sources/unsafe_bench_debug/source_manifest.jsonl \
  --output-dir /workplace/hyx/safety_edit_teacher_qwen_vlm/unsafe_bench_debug \
  --limit 5 \
  --log-level DEBUG
```

Stage 2 只跑 editor，输入 Stage 1 的 `manifest.jsonl`：

```bash
CUDA_VISIBLE_DEVICES=0 \
python3 -m teletron.safety_edit.teacher_pipeline.run \
  --config examples/teleai/config/safety_edit_teacher_qwen_editor_stage.yaml \
  --input /workplace/hyx/safety_edit_teacher_qwen_vlm/unsafe_bench_debug/manifest.jsonl \
  --output-dir /workplace/hyx/safety_edit_teacher_qwen_editor/unsafe_bench_debug \
  --limit 5 \
  --log-level DEBUG
```

如果一张 H100 放不下 Qwen-Image-Edit，把 editor 配置里的 `device_map` 改为 `balanced` 或 `auto`，并用两张卡：

```yaml
editor:
  params:
    device_map: balanced
```

## 接入其他真实本地模型

新增一个本地 adapter 文件，例如：

```python
from teletron.safety_edit.teacher_pipeline.schemas import EditorResult, TeacherPlan

class LocalQwenVLMTeacher:
    def __init__(self, model_path, device="cuda"):
        ...

    def plan(self, image_path, image):
        ...
        return TeacherPlan(...)

class LocalImageEditorTeacher:
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
  target: my_project.safety_edit_adapters:LocalImageEditorTeacher
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
