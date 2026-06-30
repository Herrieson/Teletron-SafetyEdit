# Safety Edit Data Sources

## 目标

安全编辑教师流水线需要两类输入图片：

```text
unsafe images:
  用于生成 teacher_prompt 和 teacher_output

safe images:
  用于 no-op 样本，训练模型在无风险图片上保持不变
```

第一版目标不是一次收集所有公开安全数据，而是先构造一个稳定的小闭环：

```text
UnsafeBench unsafe/safe
+ COCO safe no-op
+ T2ISafety filtered unsafe-like
```

然后再扩展到更多 benchmark 或视频数据。

## 推荐数据集

### 1. yiting/UnsafeBench

用途：

```text
主安全图片数据源。
同时提供 safe 和 unsafe 图片，适合第一版 teacher pipeline 和 no-op 训练。
```

建议：

```text
Phase 0: unsafe 50 + safe 50
Phase 1: unsafe 1k + safe 1k
Phase 2: 全量或按类别均衡采样
```

注意：

```text
通常没有精细 mask/bbox。
teacher_mask 第一版可以为空，后续由 VLM/grounding/segmentation 补。
```

### 2. OpenSafetyLab/t2i_safety_dataset

用途：

```text
补充 AI-generated image 的安全风险分布和更细风险类别。
适合扩充 unsafe-like 图片，也适合训练/评估 verifier。
```

建议：

```text
先抽 1k-5k 可局部编辑类别。
暂缓抽象 fairness/bias 类，除非能明确通过图像编辑修复。
```

### 3. lmms-lab/COCO-Caption2017

用途：

```text
主 no-op safe 数据。
```

生成 no-op 样本：

```text
safe_flag = true
teacher_prompt = "no edit needed"
teacher_output = input image
teacher_mask = null / all zeros
```

建议比例：

```text
初始 unsafe:safe = 1:1
如果模型过度编辑，调到 1:2
```

### 4. htcwang/KUN-IMAGE

用途：

```text
候选真实世界安全图片数据。
```

先用 `inspect-hf` 检查字段和可下载性，再决定是否纳入。

### 5. held-out evaluation

以下数据更适合评估，而不是第一批训练：

```text
etri-vilab/holisafe-bench
XuankunRong/SafeTag-VL-3K
oneonlee/Meme-Safety-Bench
EchoSafe-MLLM/MM-SafetyBench-plus-plus
Holly301/SaLAD
```

原因：

```text
有些风险来自图文组合、用户问题或社会语境，不一定能通过图像局部编辑解决。
```

## 统一 source manifest

所有源数据先转成统一 `source_manifest.jsonl`，再喂给 teacher pipeline。

每行格式：

```json
{
  "id": "unsafe_bench_00000001",
  "image_path": "images/unsafe_bench_00000001.jpg",
  "source_dataset": "yiting/UnsafeBench",
  "source_label": "unsafe",
  "risk_type": "weapon",
  "safe_flag": false,
  "source_metadata": {}
}
```

teacher pipeline 会读取 `image_path`，其他字段会写进输出 manifest 的 `metadata.source`，用于后续分析。

## 准备本地图片目录

```bash
python3 -m teletron.safety_edit.source_data.prepare local-dir \
  --input-dir /path/to/images \
  --output-dir /tmp/safety_edit_sources/local_safe \
  --source-dataset local_safe \
  --source-label safe \
  --risk-type none \
  --safe-flag true \
  --copy-images \
  --limit 100
```

输出：

```text
/tmp/safety_edit_sources/local_safe/source_manifest.jsonl
/tmp/safety_edit_sources/local_safe/images/
```

## 检查 Hugging Face 数据字段

先不要直接大规模下载，先检查字段：

```bash
python3 -m teletron.safety_edit.source_data.prepare inspect-hf \
  --dataset yiting/UnsafeBench \
  --split train \
  --streaming \
  --limit 3
```

实际字段已验证为：

```text
image: PIL image
safety_label: "Safe" / "Unsafe"
category: e.g. Hate
source: e.g. Laion5B
text: source caption/text
```

如果 streaming 退出时触发 Hugging Face/Xet 连接或 Python finalizing 崩溃，但已经打印出样本字段，可以继续用非 streaming 方式准备小样本。

其他候选：

```bash
python3 -m teletron.safety_edit.source_data.prepare inspect-hf \
  --dataset OpenSafetyLab/t2i_safety_dataset \
  --split train \
  --limit 3

python3 -m teletron.safety_edit.source_data.prepare inspect-hf \
  --dataset lmms-lab/COCO-Caption2017 \
  --split val \
  --streaming \
  --limit 3
```

COCO-Caption2017 当前可用 split 是 `val` 和 `test`，不是 `train`。

T2ISafety 是分卷 zip，当前不要用 `--streaming`；如果本地缓存里有坏的 zip 分片，使用 `--download-mode force_redownload` 重下。

## 准备 Hugging Face 数据

UnsafeBench：

```bash
python3 -m teletron.safety_edit.source_data.prepare hf \
  --preset unsafe_bench \
  --split train \
  --output-dir /tmp/safety_edit_sources/unsafe_bench_debug \
  --limit 100
```

T2ISafety：

```bash
python3 -m teletron.safety_edit.source_data.prepare hf \
  --preset t2i_safety \
  --split train \
  --download-mode force_redownload \
  --output-dir /tmp/safety_edit_sources/t2i_safety_debug \
  --limit 100
```

COCO safe no-op：

```bash
python3 -m teletron.safety_edit.source_data.prepare hf \
  --preset coco_caption2017 \
  --split val \
  --streaming \
  --output-dir /tmp/safety_edit_sources/coco_safe_debug \
  --limit 100
```

如果 preset 字段不匹配，先用 `inspect-hf` 看字段，再显式指定：

```bash
python3 -m teletron.safety_edit.source_data.prepare hf \
  --dataset DATASET_NAME \
  --split train \
  --streaming \
  --image-field image \
  --label-field label \
  --risk-field category \
  --output-dir /tmp/safety_edit_sources/custom \
  --limit 100
```

## 喂给 teacher pipeline

```bash
python3 -m teletron.safety_edit.teacher_pipeline.run \
  --config examples/teleai/config/safety_edit_teacher_static.yaml \
  --input /tmp/safety_edit_sources/unsafe_bench_debug/source_manifest.jsonl \
  --output-dir /tmp/safety_edit_teacher_unsafe_bench_static \
  --limit 10
```

真实 Qwen 教师：

```bash
python3 -m teletron.safety_edit.teacher_pipeline.run \
  --config examples/teleai/config/safety_edit_teacher_qwen.yaml \
  --input /tmp/safety_edit_sources/unsafe_bench_debug/source_manifest.jsonl \
  --output-dir /tmp/safety_edit_teacher_unsafe_bench_qwen \
  --limit 10 \
  --log-level DEBUG
```

两张 H100 上建议用两阶段 Qwen 教师，避免 VLM 和 editor 同进程 OOM：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python3 -m teletron.safety_edit.teacher_pipeline.run \
  --config examples/teleai/config/safety_edit_teacher_qwen_vlm_stage.yaml \
  --input /tmp/safety_edit_sources/unsafe_bench_debug/source_manifest.jsonl \
  --output-dir /tmp/safety_edit_teacher_qwen_vlm/unsafe_bench_debug \
  --limit 10 \
  --log-level DEBUG

CUDA_VISIBLE_DEVICES=0 \
python3 -m teletron.safety_edit.teacher_pipeline.run \
  --config examples/teleai/config/safety_edit_teacher_qwen_editor_stage.yaml \
  --input /tmp/safety_edit_teacher_qwen_vlm/unsafe_bench_debug/manifest.jsonl \
  --output-dir /tmp/safety_edit_teacher_qwen_editor/unsafe_bench_debug \
  --limit 10 \
  --log-level DEBUG
```

## 第一批建议

```text
Phase 0:
  UnsafeBench 100
  COCO safe 100

Phase 1:
  UnsafeBench unsafe/safe 2k
  COCO safe 2k
  T2ISafety filtered 2k

Held-out:
  holisafe-bench 200
  SafeTag-VL-3K 300
  Meme-Safety-Bench 300
```

## 从 teacher 输出构建训练数据集

真实 teacher pipeline 跑完后，使用 `teletron.safety_edit.build_dataset` 把一个或多个 teacher run 的
`manifest.jsonl` 合并成训练入口。

第一阶段 condition bridge 训练只要求：

```text
vlm_hidden_path
teacher_condition_path
safe_flag
risk_type
```

构建命令：

```bash
uv run python -m teletron.safety_edit.build_dataset build \
  --input /workplace/hyx/safety_edit_teacher_qwen \
  --output-dir /workplace/hyx/safety_edit_datasets/qwen_unsafe_bench_debug \
  --stage condition \
  --val-ratio 0.1 \
  --test-ratio 0 \
  --seed 0 \
  --inspect-tensors \
  --log-rejected
```

输出：

```text
/workplace/hyx/safety_edit_datasets/qwen_unsafe_bench_debug/
├── manifest.jsonl
├── stats.json
├── rejected.jsonl
└── splits/
    ├── train.jsonl
    ├── val.jsonl
    └── test.jsonl
```

默认行为：

```text
只保留 accepted=true
校验 required asset 是否存在
按 id 去重
不复制 teacher 资产，manifest 内保存绝对路径
```

如果要把资产复制到数据集目录，增加：

```bash
--copy-assets
```

检查已有数据统计：

```bash
uv run python -m teletron.safety_edit.build_dataset stats \
  --input /workplace/hyx/safety_edit_datasets/qwen_unsafe_bench_debug/manifest.jsonl \
  --inspect-tensors
```

当前 debug 数据集构建结果：

```text
rows: 30
train: 27
val: 3
test: 0
safe_flag=true: 21
safe_flag=false: 9
risk_type:
  none: 21
  weapon: 5
  other: 3
  self_harm: 1
```

注意：

```text
teacher_condition.prompt_embeds token 长度不固定。
vlm_hidden token 长度也不固定。
```

第一阶段训练的 collator / model 需要支持变长序列，例如 padding、attention mask，或在
`ConditionBridge` 内使用 Perceiver/CrossAttention resampler。

训练读取接口：

```python
from teletron.datasets.safety_edit_dataset import SafetyEditDataset

dataset = SafetyEditDataset(
    manifest_path="/workplace/hyx/safety_edit_datasets/qwen_unsafe_bench_debug/splits/train.jsonl",
    load_tensors=True,
    load_images=False,
)

sample = dataset[0]
vlm_hidden = sample["vlm_hidden"]
teacher_condition = sample["teacher_condition"]
```
