from pathlib import Path

from PIL import Image

from teletron.safety_edit.teacher_pipeline.pipeline import TeacherPipeline


def test_static_teacher_pipeline_writes_manifest(tmp_path):
    image_dir = tmp_path / "input"
    image_dir.mkdir()
    image_path = image_dir / "sample.jpg"
    Image.new("RGB", (16, 12), color=(12, 34, 56)).save(image_path)

    output_dir = tmp_path / "teacher_data"
    config = {
        "metadata": {
            "teacher_vlm": "static",
            "teacher_editor": "copy",
        },
        "vlm": {
            "target": "teletron.safety_edit.teacher_pipeline.adapters:StaticVLMTeacher",
            "params": {
                "teacher_prompt": "no edit needed",
                "safe_flag": True,
                "hidden_shape": [2, 4],
            },
        },
        "editor": {
            "target": "teletron.safety_edit.teacher_pipeline.adapters:CopyEditorTeacher",
            "params": {
                "condition_shape": [2, 4],
            },
        },
        "verifier": {
            "target": "teletron.safety_edit.teacher_pipeline.adapters:PixelDiffVerifier",
        },
        "writer": {
            "output_dir": str(output_dir),
            "copy_images": True,
            "include_rejected": True,
        },
    }

    pipeline = TeacherPipeline.from_config(config)
    rows = pipeline.run([image_path])

    assert len(rows) == 1
    row = rows[0]
    assert row["safe_flag"] is True
    assert row["accepted"] is True
    assert row["teacher_prompt"] == "no edit needed"
    assert (output_dir / row["image_path"]).exists()
    assert (output_dir / row["teacher_output_path"]).exists()
    assert (output_dir / row["teacher_condition_path"]).exists()
    assert (output_dir / row["vlm_hidden_path"]).exists()
    assert (output_dir / "manifest.jsonl").exists()

