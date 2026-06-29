# Copyright (c) Jixiang Luo. All Rights Reserved.
# Licensed under the MIT License.

import argparse
import io
import json
import os
import time

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from third_party.keyframe_src.layers.cuda_inference import replicate_pad
from third_party.keyframe_src.models.video_model import DMC
from third_party.keyframe_src.models.image_model import DMCI
from third_party.keyframe_src.utils.common import str2bool, create_folder, generate_log_json, get_state_dict, \
    dump_json, set_torch_env
from third_party.keyframe_src.utils.stream_helper import SPSHelper, NalType, write_sps, read_header, \
    read_sps_remaining, read_ip_remaining, write_ip
from third_party.keyframe_src.utils.video_reader import PNGReader, YUV420Reader
from third_party.keyframe_src.utils.video_writer import PNGWriter, YUV420Writer
from third_party.keyframe_src.utils.metrics import calc_psnr, calc_msssim, calc_msssim_rgb
from third_party.keyframe_src.utils.transforms import rgb2ycbcr, ycbcr2rgb, yuv_444_to_420, ycbcr420_to_444_np


def parse_args():
    parser = argparse.ArgumentParser(description="Example testing script")

    parser.add_argument('--force_zero_thres', type=float, default=None, required=False)
    parser.add_argument('--model_path_i', type=str)
    parser.add_argument('--model_path_p', type=str)
    parser.add_argument('--rate_num', type=int, default=4)
    parser.add_argument('--qp_i', type=int, nargs="+")
    parser.add_argument('--qp_p', type=int, nargs="+")
    parser.add_argument("--force_intra", type=str2bool, default=False)
    parser.add_argument("--force_frame_num", type=int, default=-1)
    parser.add_argument("--force_intra_period", type=int, default=-1)
    parser.add_argument('--reset_interval', type=int, default=32, required=False)
    parser.add_argument('--test_config', type=str, required=True)
    parser.add_argument('--force_root_path', type=str, default=None, required=False)
    parser.add_argument("--cuda", type=str2bool, default=True)
    parser.add_argument('--cuda_idx', type=int, nargs="+", help='GPU indexes to use')
    parser.add_argument('--calc_ssim', type=str2bool, default=True, required=False)
    parser.add_argument('--write_stream', type=str2bool, default=False)
    parser.add_argument('--check_existing', type=str2bool, default=False)
    parser.add_argument('--stream_path', type=str, default="out_bin")
    parser.add_argument('--save_decoded_frame', type=str2bool, default=False)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--verbose_json', type=str2bool, default=False)
    parser.add_argument('--verbose', type=int, default=0)

    args = parser.parse_args()
    return args


def np_image_to_tensor(img, device):
    image = torch.from_numpy(img).to(device=device).to(dtype=torch.float32) / 255.0
    image = image.unsqueeze(0)
    return image


def get_src_reader(args):
    if args['src_type'] == 'png':
        return PNGReader(args['src_path'], args['src_width'], args['src_height'])
    elif args['src_type'] == 'yuv420':
        return YUV420Reader(args['src_path'], args['src_width'], args['src_height'])
    raise ValueError(f"Unsupported source type: {args['src_type']}")


def get_src_frame(args, src_reader, device):
    if args['src_type'] == 'yuv420':
        y, uv = src_reader.read_one_frame()
        yuv = ycbcr420_to_444_np(y, uv)
        x = np_image_to_tensor(yuv, device)
        y = y[0, :, :]
        u = uv[0, :, :]
        v = uv[1, :, :]
        rgb = None
    else:
        assert args['src_type'] == 'png'
        rgb = src_reader.read_one_frame()
        x = np_image_to_tensor(rgb, device)
        x = rgb2ycbcr(x)
        y, u, v = None, None, None

    x = x.to(torch.float16)
    return x, y, u, v, rgb


def get_distortion(args, x_hat, y, u, v, rgb):
    if args['src_type'] == 'yuv420':
        y_rec, uv_rec = yuv_444_to_420(x_hat)
        y_rec = torch.clamp(y_rec * 255, 0, 255).squeeze(0).cpu().numpy()
        uv_rec = torch.clamp(uv_rec * 255, 0, 255).squeeze(0).cpu().numpy()
        y_rec = y_rec[0, :, :]
        u_rec = uv_rec[0, :, :]
        v_rec = uv_rec[1, :, :]
        psnr_y = calc_psnr(y, y_rec)
        psnr_u = calc_psnr(u, u_rec)
        psnr_v = calc_psnr(v, v_rec)
        psnr = (6 * psnr_y + psnr_u + psnr_v) / 8
        ssim = 0.
        if args['calc_ssim']:
            ssim_y = calc_msssim(y, y_rec)
            ssim_u = calc_msssim(u, u_rec)
            ssim_v = calc_msssim(v, v_rec)
            ssim = (6 * ssim_y + ssim_u + ssim_v) / 8
        return [psnr, psnr_y, psnr_u, psnr_v], [ssim, ssim_y, ssim_u, ssim_v]
    else:
        assert args['src_type'] == 'png'
        rgb_rec = ycbcr2rgb(x_hat)
        rgb_rec = torch.clamp(rgb_rec * 255, 0, 255).squeeze(0).cpu().numpy()
        psnr = calc_psnr(rgb, rgb_rec)
        msssim = calc_msssim_rgb(rgb, rgb_rec) if args['calc_ssim'] else 0.
        return [psnr], [msssim]


def encode_stream(p_frame_net, i_frame_net, args, device):
    """Encode frames and save to stream file"""
    frame_num = args['frame_num']
    verbose = args['verbose']
    reset_interval = args['reset_interval']
    intra_period = args['intra_period']

    src_reader = get_src_reader(args)
    pic_height = args['src_height']
    pic_width = args['src_width']
    padding_r, padding_b = DMCI.get_padding_size(pic_height, pic_width, 16)

    use_two_entropy_coders = pic_height * pic_width > 1280 * 720
    i_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)
    p_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)

    frame_types = []
    bits = []
    encoding_time = []
    index_map = [0, 1, 0, 2, 0, 2, 0, 2]

    output_buff = io.BytesIO()
    sps_helper = SPSHelper()

    p_frame_net.set_curr_poc(0)
    with torch.no_grad():
        last_qp = 0
        for frame_idx in range(frame_num):
            x, _, _, _, _ = get_src_frame(args, src_reader, device)

            torch.cuda.synchronize(device=device)
            frame_start_time = time.time()

            x_padded = replicate_pad(x, padding_b, padding_r)

            is_i_frame = frame_idx == 0 or (intra_period > 0 and frame_idx % intra_period == 0)
            if is_i_frame:
                curr_qp = args['qp_i']
                sps = {
                    'sps_id': -1,
                    'height': pic_height,
                    'width': pic_width,
                    'ec_part': 1 if use_two_entropy_coders else 0,
                    'use_ada_i': 0,
                }
                encoded = i_frame_net.compress(x_padded, args['qp_i'])
                p_frame_net.clear_dpb()
                p_frame_net.add_ref_frame(None, encoded['x_hat'])
                frame_types.append(0)
            else:
                fa_idx = index_map[frame_idx % 8]
                use_ada_i = 1 if (reset_interval > 0 and frame_idx % reset_interval == 1) else 0
                if use_ada_i:
                    p_frame_net.prepare_feature_adaptor_i(last_qp)
                curr_qp = p_frame_net.shift_qp(args['qp_p'], fa_idx)
                sps = {
                    'sps_id': -1,
                    'height': pic_height,
                    'width': pic_width,
                    'ec_part': 1 if use_two_entropy_coders else 0,
                    'use_ada_i': use_ada_i,
                }
                encoded = p_frame_net.compress(x_padded, curr_qp)
                last_qp = curr_qp
                frame_types.append(1)

            sps_id, sps_new = sps_helper.get_sps_id(sps)
            sps['sps_id'] = sps_id
            sps_bytes = write_sps(output_buff, sps) if sps_new else 0
            if verbose >= 2 and sps_new:
                print("new sps", sps)
            stream_bytes = write_ip(output_buff, is_i_frame, sps_id, curr_qp, encoded['bit_stream'])
            bits.append(stream_bytes * 8 + sps_bytes * 8)

            torch.cuda.synchronize(device=device)
            frame_time = time.time() - frame_start_time
            encoding_time.append(frame_time)

            if verbose >= 2:
                print(f"frame {frame_idx} encoded, {frame_time * 1000:.3f} ms, bits: {bits[-1]}")

    src_reader.close()
    with open(args['curr_bin_path'], "wb") as output_file:
        bytes_buffer = output_buff.getbuffer()
        output_file.write(bytes_buffer)
        total_bytes = bytes_buffer.nbytes
        bytes_buffer.release()
    output_buff.close()

    return frame_types, bits, encoding_time, total_bytes

def run_one_point_with_stream(p_frame_net, i_frame_net, args, device):
    if args['check_existing'] and os.path.exists(args['curr_json_path']) and os.path.exists(args['curr_bin_path']):
        with open(args['curr_json_path']) as f:
            log_result = json.load(f)
            if log_result['i_frame_num'] + log_result['p_frame_num'] == args['frame_num']:
                return log_result
        print(f"incorrect log for {args['curr_json_path']}, try to rerun.")

    start_time = time.time()

    frame_types, bits, encoding_time, _ = encode_stream(p_frame_net, i_frame_net, args, device)

    return 


def init(args):
    """Initialize all necessary components and generate tasks"""
    if args.force_zero_thres is not None and args.force_zero_thres < 0:
        args.force_zero_thres = None

    # Initialize device
    device = "cpu"
    if args.cuda:
        if args.cuda_idx is not None and len(args.cuda_idx) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_idx[0])
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Initialize models
    set_torch_env()
    i_frame_net = DMCI()
    i_state_dict = get_state_dict(args.model_path_i)
    i_frame_net.load_state_dict(i_state_dict)
    i_frame_net = i_frame_net.to(device).eval()
    i_frame_net.update(args.force_zero_thres)
    i_frame_net.half()

    p_frame_net = DMC()
    if not args.force_intra:
        p_state_dict = get_state_dict(args.model_path_p)
        p_frame_net.load_state_dict(p_state_dict)
        p_frame_net = p_frame_net.to(device).eval()
        p_frame_net.update(args.force_zero_thres)
        p_frame_net.half()

    # Load config
    with open(args.test_config) as f:
        config = json.load(f)
    root_path = args.force_root_path if args.force_root_path is not None else config['root_path']
    config = config['test_classes']

    # Process QP
    rate_num = args.rate_num
    qp_i = args.qp_i if args.qp_i is not None else [int(i + 0.5) for i in np.linspace(0, DMC.get_qp_num() - 1, num=rate_num)]
    assert len(qp_i) == rate_num
    qp_p = args.qp_p if (not args.force_intra and args.qp_p is not None) else qp_i
    if not args.force_intra:
        assert len(qp_p) == rate_num

    print(f"testing {rate_num} rates, using qp: {', '.join(map(str, qp_i))}")

    # Generate tasks
    tasks = []
    count_frames = 0
    count_sequences = 0
    for ds_name in config:
        if config[ds_name]['test'] == 0:
            continue
        for seq in config[ds_name]['sequences']:
            count_sequences += 1
            for rate_idx in range(rate_num):
                cur_args = {
                    'rate_idx': rate_idx,
                    'qp_i': qp_i[rate_idx],
                    'force_intra': args.force_intra,
                    'reset_interval': args.reset_interval,
                    'seq': seq,
                    'src_type': config[ds_name]['src_type'],
                    'src_height': config[ds_name]['sequences'][seq]['height'],
                    'src_width': config[ds_name]['sequences'][seq]['width'],
                    'intra_period': config[ds_name]['sequences'][seq]['intra_period'],
                    'frame_num': config[ds_name]['sequences'][seq]['frames'],
                    'calc_ssim': args.calc_ssim,
                    'dataset_path': os.path.join(root_path, config[ds_name]['base_path']),
                    'write_stream': args.write_stream,
                    'check_existing': args.check_existing,
                    'stream_path': args.stream_path,
                    'save_decoded_frame': args.save_decoded_frame,
                    'ds_name': ds_name,
                    'verbose': args.verbose,
                    'verbose_json': args.verbose_json
                }
                if not args.force_intra:
                    cur_args['qp_p'] = qp_p[rate_idx]
                if args.force_intra:
                    cur_args['intra_period'] = 1
                if args.force_intra_period > 0:
                    cur_args['intra_period'] = args.force_intra_period
                if args.force_frame_num > 0:
                    cur_args['frame_num'] = args.force_frame_num

                # Create output directories
                bin_folder = os.path.join(args.stream_path, ds_name)
                create_folder(bin_folder, True)
                cur_args['src_path'] = os.path.join(cur_args['dataset_path'], seq)
                cur_args['bin_folder'] = bin_folder
                cur_args['curr_bin_path'] = os.path.join(bin_folder, f"{seq}_q{cur_args['qp_i']}.bin")
                cur_args['curr_rec_path'] = cur_args['curr_bin_path'].replace('.bin', '.yuv')
                cur_args['curr_json_path'] = cur_args['curr_bin_path'].replace('.bin', '.json')

                count_frames += cur_args['frame_num']
                tasks.append(cur_args)

    return device, i_frame_net, p_frame_net, tasks, count_frames, count_sequences


def main():
    begin_time = time.time()
    args = parse_args()

    # Initialize all components and get tasks
    device, i_frame_net, p_frame_net, tasks, count_frames, count_sequences = init(args)

    # Process all tasks
    for task in tqdm(tasks, desc="Processing tasks"):
        result = run_one_point_with_stream(p_frame_net, i_frame_net, task, device)

    # Save final results


if __name__ == "__main__":
    main()

from PIL import Image

class KeyframeEncoder:
    def __init__(self, model_path_i, model_path_p, force_zero_thres, device):
        self.device = device
        set_torch_env()
        
        self.i_frame_net = DMCI()
        self.i_frame_net.load_state_dict(get_state_dict(model_path_i))
        self.i_frame_net = self.i_frame_net.to(self.device).eval()
        self.i_frame_net.update(force_zero_thres)
        self.i_frame_net.half()
        # self.i_frame_net.double()

        self.p_frame_net = DMC()
        self.p_frame_net.load_state_dict(get_state_dict(model_path_p))
        self.p_frame_net = self.p_frame_net.to(self.device).eval()
        self.p_frame_net.update(force_zero_thres)
        self.p_frame_net.half()
        # self.p_frame_net.double()
        
        self.sps_helper = SPSHelper()
        self.frame_idx = 0
        self.last_qp = 0
        
        # Runtime params
        self.pic_height = 0
        self.pic_width = 0
        self.padding_r = 0
        self.padding_b = 0
        self.use_two_entropy_coders = False

def init_encoder(
    model_path_i="/mnt/nvme0/yfq/linlx/keyframe_i.pth.tar",
    model_path_p="/mnt/nvme0/yfq/linlx/keyframe_p.pth.tar",
    force_zero_thres=0.12,
    device="cpu"
):
    return KeyframeEncoder(model_path_i, model_path_p, force_zero_thres, device)

def _process_frame_internal(encoder, img_bytes_io, is_i_frame, qp_i, qp_p, reset_interval):
    # Read image
    img = Image.open(img_bytes_io).convert('RGB')
    width, height = img.size
    
    # Init dimensions if first frame
    if encoder.frame_idx == 0:
        encoder.pic_width = width
        encoder.pic_height = height
        encoder.padding_r, encoder.padding_b = DMCI.get_padding_size(height, width, 16)
        encoder.use_two_entropy_coders = height * width > 1280 * 720
        encoder.i_frame_net.set_use_two_entropy_coders(encoder.use_two_entropy_coders)
        encoder.p_frame_net.set_use_two_entropy_coders(encoder.use_two_entropy_coders)
        encoder.p_frame_net.set_curr_poc(0)

    # Convert to tensor
    img_np = np.array(img).transpose(2, 0, 1)
    x = np_image_to_tensor(img_np, encoder.device)
    x = rgb2ycbcr(x)
    x = x.to(torch.float16)
    
    x_padded = replicate_pad(x, encoder.padding_b, encoder.padding_r)
    
    output_buff = io.BytesIO()
    
    if is_i_frame:
        curr_qp = qp_i
        sps = {
            'sps_id': -1,
            'height': encoder.pic_height,
            'width': encoder.pic_width,
            'ec_part': 1 if encoder.use_two_entropy_coders else 0,
            'use_ada_i': 0,
        }
        encoded = encoder.i_frame_net.compress(x_padded, curr_qp)
        encoder.p_frame_net.clear_dpb()
        encoder.p_frame_net.add_ref_frame(None, encoded['x_hat'].detach())
    else:
        # P-frame logic
        index_map = [0, 1, 0, 2, 0, 2, 0, 2]
        fa_idx = index_map[encoder.frame_idx % 8]
        
        use_ada_i = 1 if (reset_interval > 0 and encoder.frame_idx % reset_interval == 1) else 0
        if use_ada_i:
            encoder.p_frame_net.prepare_feature_adaptor_i(encoder.last_qp)
            
        curr_qp = encoder.p_frame_net.shift_qp(qp_p, fa_idx)
        sps = {
            'sps_id': -1,
            'height': encoder.pic_height,
            'width': encoder.pic_width,
            'ec_part': 1 if encoder.use_two_entropy_coders else 0,
            'use_ada_i': use_ada_i,
        }
        encoded = encoder.p_frame_net.compress(x_padded, curr_qp)
        encoder.last_qp = curr_qp

    sps_id, sps_new = encoder.sps_helper.get_sps_id(sps)
    sps['sps_id'] = sps_id
    sps_bytes = write_sps(output_buff, sps) if sps_new else 0
    
    write_ip(output_buff, is_i_frame, sps_id, curr_qp, encoded['bit_stream'])
    
    encoder.frame_idx += 1
    
    return output_buff.getvalue()

def encoder_keyframe_fun(
    encoder,
    first_frame_bytes_io, 
    last_frame_bytes_io,
    qp_i=42,
    qp_p=42,
    reset_interval=64
):
    result_stream = io.BytesIO()
    if encoder.frame_idx == 0:
        if last_frame_bytes_io == None:
            stream1 = _process_frame_internal(encoder, first_frame_bytes_io, True, qp_i, qp_p, reset_interval)
            result_stream.write(stream1)
            return result_stream.getvalue()

        # First time: Encode both
        stream1 = _process_frame_internal(encoder, first_frame_bytes_io, True, qp_i, qp_p, reset_interval)
        result_stream.write(stream1)
        
        stream2 = _process_frame_internal(encoder, last_frame_bytes_io, False, qp_i, qp_p, reset_interval)
        result_stream.write(stream2)
    else:
        # Subsequent times: Encode only last_frame (as P-frame)
        stream = _process_frame_internal(encoder, last_frame_bytes_io, False, qp_i, qp_p, reset_interval)
        result_stream.write(stream)
        
    return result_stream.getvalue()

