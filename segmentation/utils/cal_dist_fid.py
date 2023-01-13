import argparse

import h5py

import os
import PIL.Image
from shutil import rmtree
import numpy as np
import sys
from pytorch_fid import fid_score


def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data


def calculate_dist_fid_from_h5(real_data_stats_list, fake_h5_list, lv1_name="train", tmp_folder="./_calculate_fid_dir", oldformat=False, fake_h5_ch=-1, isrgb=False):
    real_data_stats = []
    print("load real data statistics ...")
    total_sample_num = 0
    for data_stat_file in real_data_stats_list:
        stats = np.load(data_stat_file)
        print("load ", data_stat_file)
        real_data_stats.append(stats)
        total_sample_num += stats['count']

    # base_name = os.path.basename(real_h5).replace(".h5","")
    # tmp_folder = tmp_folder.replace("#real_id#", base_name)
    print(f"tmp folder:{tmp_folder}")
    # real_path = os.path.join(tmp_folder,"real")
    fake_path = os.path.join(tmp_folder,"fake")

    if not oldformat:
        level_append = "/data"
    else:
        level_append = ""

    for fake_h5 in fake_h5_list:
        if os.path.isdir(fake_path):
            rmtree(fake_path, ignore_errors=True)
        os.makedirs(fake_path, exist_ok=True)

        fakeh5 = h5py.File(fake_h5, 'r')
        fake_keys = list(fakeh5[lv1_name].keys())
        print(f"start save fake images from {fake_h5}({len(fake_keys)})...")
        for idx, fake_key in enumerate(fake_keys):
            data = fakeh5[f'{lv1_name}/{fake_key}{level_append}'][()].squeeze()

            if isrgb:
                if data.shape[0] == 3:
                    data = np.moveaxis(data, 0, -1)
                im = PIL.Image.fromarray(data).convert("RGB").resize((256, 256))
                im.save(os.path.join(fake_path, str(idx) + ".png"))
            else:

                if data.shape[0] == 3 or len(data.shape) == 2:
                    data = adjust_dynamic_range(data, [data.min(),data.max()],[0,255]).astype("uint8")
                    if data.shape[0]==3:
                        data = np.moveaxis(data,0,-1)
                    im = PIL.Image.fromarray(data).convert("RGB").resize((256,256))
                    im.save(os.path.join(fake_path, str(idx)+".png"))
                else:
                    if idx == 0 and fake_h5_ch == -1:
                        print(f"fake img shape is:{data.shape}, save multiple channels")
                    elif idx == 0:
                        print(f"fake img shape is:{data.shape}, save channel: {fake_h5_ch}")
                    if fake_h5_ch >= 0:
                        data[fake_h5_ch] = adjust_dynamic_range(data[fake_h5_ch], [data[fake_h5_ch].min(),data[fake_h5_ch].max()],[0,255]).astype("uint8")
                        im_ch = PIL.Image.fromarray(data[fake_h5_ch]).convert("RGB").resize((256,256))
                        im_ch.save(os.path.join(fake_path, str(idx)+".png"))
                    else:
                        for ch in range(data.shape[0]):
                            data[ch] = adjust_dynamic_range(data[ch], [data[ch].min(),data[ch].max()],[0,255]).astype("uint8")
                            im_ch = PIL.Image.fromarray(data[ch]).convert("RGB").resize((256,256))
                            im_ch.save(os.path.join(fake_path, str(idx)+"_"+str(ch)+".png"))

        print(f"start calculate FID -- file:{fake_h5}")
        fake_stat_file = os.path.basename(fake_h5).replace(".h5", ".npz")
        mu2, sigma2 = fid_score.save_fid_stats([str(fake_path), fake_stat_file], 256, 0, 2048)

        dist_fid = 0
        for client_idx in range(len(real_data_stats)):
            m1, s1 = real_data_stats[client_idx]['mu'], real_data_stats[client_idx]['sigma']
            weight = real_data_stats[client_idx]['count'] / total_sample_num
            fid_client = weight * fid_score.calculate_frechet_distance(m1, s1, mu2, sigma2)
            dist_fid += fid_client

        print(f"***Dist FID score: {dist_fid}***")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate FID"
    )
    parser.add_argument(
        "--real_stats", nargs="+", required=True, help="paths to the real data stat files"
    )
    parser.add_argument(
        "--fake_h5", nargs="+", required=True, help="paths to the synthetic h5 data"
    )

    parser.add_argument(
        "--lv1_name", type=str, default="train", help="path to the fake h5 data"
    )

    parser.add_argument(
        "--oldformat", action='store_true', help="path to the fake h5 data"
    )

    parser.add_argument(
        "--isrgb", action='store_true', help="rgb data (histopathology  image)"
    )

    parser.add_argument(
        "--fake_h5_ch", type=int, default=-1, help="select channel if only choose one channel"
    )

    args = parser.parse_args()

    calculate_dist_fid_from_h5(args.real_stats, args.fake_h5, lv1_name=args.lv1_name, oldformat=args.oldformat, fake_h5_ch=args.fake_h5_ch, isrgb=args.isrgb)

