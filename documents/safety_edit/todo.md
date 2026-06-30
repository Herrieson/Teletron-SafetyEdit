# Safety Edit TODO

本文档记录当前 safety edit 蒸馏任务的近期 TODO。优先级按能否尽快跑通真实教师模型数据闭环排序。

## 当前目标

先完成图片版教师模型流水线，稳定生成第一批可用于蒸馏的样本：

```text
source images
  -> Qwen3.6-27B VLM planner
  -> teacher_prompt / safe_flag / risk_type / vlm_hidden
  -> Qwen-Image-Edit text encoder + editor
  -> teacher_condition / teacher_output
  -> verifier
  -> manifest.jsonl + assets
```

视频先不做完整视频模型。第一版按图片任务推进；后续如果要支持视频，先用逐帧处理或关键帧处理验证数据闭环，再考虑接视频编辑模型。

## P0: 先跑通教师模型闭环

- [ ] 在服务器确认两张 H100 空闲状态。

  ```bash
  nvidia-smi
  ```

- [ ] 确认依赖环境可用。

  ```bash
  cd /workplace/hyx/Teletron-SafetyEdit
  uv sync --index-url https://pypi.tuna.tsinghua.edu.cn/simple
  ```

- [ ] 确认模型已经下载到本地路径。

  ```text
  /workplace/hyx/models/Qwen3.6-27B
  /workplace/hyx/models/Qwen-Image-Edit
  ```

  如果没有，先下载：

  ```bash
  mkdir -p /workplace/hyx/models

  uv run huggingface-cli download Qwen/Qwen3.6-27B \
    --local-dir /workplace/hyx/models/Qwen3.6-27B \
    --local-dir-use-symlinks False

  uv run huggingface-cli download Qwen/Qwen-Image-Edit \
    --local-dir /workplace/hyx/models/Qwen-Image-Edit \
    --local-dir-use-symlinks False
  ```

- [ ] 准备 UnsafeBench debug source manifest。

  ```bash
  uv run python -m teletron.safety_edit.source_data.prepare hf \
    --preset unsafe_bench \
    --split train \
    --output-dir /workplace/hyx/safety_edit_sources/unsafe_bench_debug \
    --limit 100
  ```

- [ ] 准备 COCO safe no-op debug source manifest。

  ```bash
  uv run python -m teletron.safety_edit.source_data.prepare hf \
    --preset coco_caption2017 \
    --split val \
    --output-dir /workplace/hyx/safety_edit_sources/coco_safe_debug \
    --limit 100
  ```

- [ ] 跑一次静态 adapter smoke test，确认 manifest 和资产落盘链路正常。

  ```bash
  uv run python -m teletron.safety_edit.teacher_pipeline.run \
    --config examples/teleai/config/safety_edit_teacher_static.yaml \
    --input /workplace/hyx/safety_edit_sources/unsafe_bench_debug/source_manifest.jsonl \
    --output-dir /workplace/hyx/safety_edit_teacher_static/unsafe_bench_debug \
    --limit 10
  ```

- [ ] 跑单进程双卡 Qwen 教师流水线。

  当前配置约定：

  ```text
  cuda:0 -> Qwen3.6-27B VLM
  cuda:1 -> Qwen-Image-Edit
  ```

  命令：

  ```bash
  CUDA_VISIBLE_DEVICES=0,1 \
  uv run python -m teletron.safety_edit.teacher_pipeline.run \
    --config examples/teleai/config/safety_edit_teacher_qwen.yaml \
    --input /workplace/hyx/safety_edit_sources/unsafe_bench_debug/source_manifest.jsonl \
    --output-dir /workplace/hyx/safety_edit_teacher_qwen/unsafe_bench_debug \
    --limit 5 \
    --log-level DEBUG
  ```

- [ ] 检查教师输出 manifest。

  重点看以下字段是否正常：

  ```text
  teacher_prompt
  safe_flag
  risk_type
  risk_description
  vlm_hidden_path
  teacher_condition_path
  teacher_output_path
  accepted
  reject_reasons
  ```

  快速查看：

  ```bash
  head -n 2 /workplace/hyx/safety_edit_teacher_qwen/unsafe_bench_debug/manifest.jsonl
  ```

- [ ] 人工抽查少量输出图，确认编辑结果没有明显坏图、过度编辑或无效编辑。

## P0: OOM 和运行失败处理

- [ ] 如果 `GPU 0` OOM，说明 VLM 单卡压力过大。

  可尝试：

  ```yaml
  vlm:
    params:
      device: null
      device_map: balanced
  ```

  然后用 2 或 4 张卡重新跑。

- [ ] 如果 `GPU 1` OOM，说明 Qwen-Image-Edit 单卡压力过大。

  可尝试：

  ```yaml
  editor:
    params:
      device_map: balanced
  ```

- [ ] 如果模型能加载但生成阶段 OOM，先降低 hidden 保存压力验证通路。

  可尝试：

  ```yaml
  vlm:
    params:
      hidden_strategy: mean
  ```

  或临时关闭：

  ```yaml
  vlm:
    params:
      extract_hidden: false
  ```

- [ ] 如果单进程双卡仍不稳定，改用两阶段教师流水线。

  Stage 1:

  ```bash
  CUDA_VISIBLE_DEVICES=0,1 \
  uv run python -m teletron.safety_edit.teacher_pipeline.run \
    --config examples/teleai/config/safety_edit_teacher_qwen_vlm_stage.yaml \
    --input /workplace/hyx/safety_edit_sources/unsafe_bench_debug/source_manifest.jsonl \
    --output-dir /workplace/hyx/safety_edit_teacher_qwen_vlm/unsafe_bench_debug \
    --limit 5 \
    --log-level DEBUG
  ```

  Stage 2:

  ```bash
  CUDA_VISIBLE_DEVICES=0 \
  uv run python -m teletron.safety_edit.teacher_pipeline.run \
    --config examples/teleai/config/safety_edit_teacher_qwen_editor_stage.yaml \
    --input /workplace/hyx/safety_edit_teacher_qwen_vlm/unsafe_bench_debug/manifest.jsonl \
    --output-dir /workplace/hyx/safety_edit_teacher_qwen_editor/unsafe_bench_debug \
    --limit 5 \
    --log-level DEBUG
  ```

## P1: 数据集构建

- [ ] 把 UnsafeBench 和 COCO safe 样本扩到第一批可训练规模。

  建议第一批：

  ```text
  UnsafeBench: 1k unsafe + 1k safe
  COCO safe: 1k-2k safe no-op
  ```

- [ ] 增加按类别统计脚本，检查 `risk_type`、`safe_flag`、`source_dataset` 分布。

- [ ] 增加 source manifest 合并工具，把多个 source manifest 合成一个训练入口。

  注意保留：

  ```text
  image_path
  source_dataset
  source_label
  source_metadata
  ```

- [x] 增加 teacher manifest 构建工具，把多个真实 teacher 输出合并、校验、去重、切分为训练入口。

  当前入口：

  ```bash
  uv run python -m teletron.safety_edit.build_dataset build \
    --input /workplace/hyx/safety_edit_teacher_qwen \
    --output-dir /workplace/hyx/safety_edit_datasets/qwen_unsafe_bench_debug \
    --stage condition \
    --val-ratio 0.1 \
    --inspect-tensors \
    --log-rejected
  ```

  当前 debug 产物：

  ```text
  /workplace/hyx/safety_edit_datasets/qwen_unsafe_bench_debug
  rows: 30
  train: 27
  val: 3
  safe: 21
  unsafe: 9
  ```

- [ ] 暂缓 T2ISafety 大规模接入，先解决它的分卷 zip 下载和缓存问题。

## P1: 教师输出质量控制

- [ ] 增强 verifier。

  第一版至少补充：

  ```text
  safe no-op 图像差异阈值
  输出图是否为空/坏图
  输出尺寸是否正确
  unsafe 样本是否实际发生变化
  ```

- [ ] 增加 VLM 输出 JSON 校验。

  必须校验：

  ```text
  safe_flag 是 bool
  risk_type 在允许枚举内
  teacher_prompt 非空
  safe_flag=true 时 teacher_prompt 固定为 no edit needed
  ```

- [ ] 保存 VLM 原始 response 和解析错误统计，便于调 prompt。

- [ ] 对 `teacher_condition` 做形状统计，确认不同样本输出结构一致。

## P1: 第一阶段蒸馏训练

- [x] 定义 `SafetyEditDataset`。

  输入：

  ```text
  manifest.jsonl
  vlm_hidden_path
  teacher_condition_path
  safe_flag
  risk_type
  ```

  当前已提供轻量读取接口：

  ```text
  teletron/datasets/safety_edit_dataset.py
  ```

  可直接读取 `manifest.jsonl` / `splits/train.jsonl` 并加载 `vlm_hidden` 与 `teacher_condition`。

- [ ] 实现第一版 `ConditionBridge`。

  目标：

  ```text
  vlm_hidden -> teacher_condition
  ```

  先做简单 MLP/Transformer projector，不接编辑模型反传。

- [ ] 实现训练入口和配置。

  第一版 loss：

  ```text
  condition L2 / cosine loss
  optional no-op gate loss
  ```

- [ ] 加一个小样本 overfit 测试。

  目标：几十个样本上 loss 能明显下降，确认数据读取和 tensor 对齐没有问题。

## P2: 后续增强

- [ ] 增加局部编辑区域能力。

  可选方案：

  ```text
  VLM bbox
  grounding model bbox
  segmentation mask
  ```

- [ ] 设计图片到视频的扩展路径。

  第一阶段建议：

  ```text
  image teacher pipeline
  -> video keyframe / per-frame safety edit
  -> temporal consistency verifier
  -> video editor adapter
  ```

- [ ] 评估是否接入 Wan/StepFun/BFL 其他编辑模型作为 teacher 或对照组。

- [ ] 增加 held-out 评估集，不直接混入第一批训练。

  候选：

  ```text
  holisafe-bench
  SafeTag-VL-3K
  Meme-Safety-Bench
  MM-SafetyBench-plus-plus
  SaLAD
  ```

## 当前完成情况

- [x] 教师流水线基础代码已实现。
- [x] 静态 adapter 可用于 smoke test。
- [x] Qwen3.6-27B VLM adapter 已接入。
- [x] Qwen-Image-Edit adapter 已接入。
- [x] UnsafeBench 字段已验证。
- [x] COCO-Caption2017 可用 split 已确认是 `val` / `test`。
- [x] 单进程双卡 Qwen 配置已改为显式绑卡。
- [ ] 真实 Qwen 教师流水线还需要在服务器上重新跑通。
