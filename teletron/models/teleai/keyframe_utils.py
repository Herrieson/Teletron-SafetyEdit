
from io import BytesIO
from PIL import Image
import sys
sys.path.insert(0, "/gemini/platform/shared/yifq1/Teletron/examples/teleai/infer")
from keyframes.keyframe_encoder import encoder_keyframe_fun, init_encoder
from keyframes.keyframe_decoder import decoder_keyframe_fun, init_decoder


def _encode_keyframe(keyframe_encoder, first_frame_bytes, last_frame_bytes, qp_i, qp_p):
    # 对关键帧进行压缩
    first_frame_bytes_io = BytesIO(first_frame_bytes)
    last_frame_bytes_io = BytesIO(last_frame_bytes)
    encoder_data = encoder_keyframe_fun(keyframe_encoder, first_frame_bytes_io, last_frame_bytes_io, qp_i=qp_i, qp_p=qp_p)
    return encoder_data


def f8tl_process_clip(first_frame, last_frame, quality=95):
    """提取首尾帧并返回 JPEG 二进制数据

    参数:
        first_frame, last_frame
        quality: JPEG 质量，默认 95

    返回:
        first_frame_bytes: 第一帧 JPEG 二进制内容
        last_frame_bytes: 最后一帧 JPEG 二进制内容
    """
    # 写入内存缓冲区而不是落盘
    first_buffer = BytesIO()
    first_frame.save(first_buffer, format='JPEG', quality=quality)
    first_frame_bytes = first_buffer.getvalue()
    last_buffer = BytesIO()
    last_frame.save(last_buffer, format='JPEG', quality=quality)
    last_frame_bytes = last_buffer.getvalue()

    return first_frame_bytes, last_frame_bytes

def frame_handle(data):
    """处理帧数据（JPEG格式）"""
    try:
        frame = Image.open(BytesIO(data))
        if frame.mode != 'RGB':
            frame = frame.convert('RGB')
        return frame
    except Exception as e:
        print("Error decoding frame data: %s", e)
        raise


def flf_process(keyframe_encoder, keyframe_decoder, first_frame, last_frame, qp_i, qp_p):
    first_frame_bytes, last_frame_bytes = f8tl_process_clip(first_frame, last_frame)
    frame_data = _encode_keyframe(keyframe_encoder, first_frame_bytes, last_frame_bytes, qp_i, qp_p)
    frame_data = BytesIO(frame_data)
    frame_bytes = len(frame_data.getvalue())
    first_frame, last_frame = decoder_keyframe_fun(keyframe_decoder, frame_data)
    last_frame = frame_handle(last_frame)
    if first_frame is not None:
        first_frame = frame_handle(first_frame)
    return first_frame, last_frame, frame_bytes

