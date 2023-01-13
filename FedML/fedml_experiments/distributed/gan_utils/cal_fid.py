import argparse

import h5py

import os
import PIL.Image
from shutil import rmtree
import numpy as np
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from fedml_api.Dist_FID import fid_score


def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data


def calculate_fid_from_h5(real_h5,fake_h5_list, lv1_name="train", tmp_folder="./_calculate_fid_dir", oldformat=False, fake_h5_ch=-1, isrgb=False):
    # base_name = os.path.basename(real_h5).replace(".h5","")
    # tmp_folder = tmp_folder.replace("#real_id#", base_name)
    print(f"tmp folder:{tmp_folder}")
    real_path = os.path.join(tmp_folder,"real")
    fake_path = os.path.join(tmp_folder,"fake")

    if os.path.isdir(real_path):
        rmtree(real_path, ignore_errors=True)
    os.makedirs(real_path, exist_ok=True)

    if not oldformat:
        level_append = "/data"
    else:
        level_append = ""

    realh5 = h5py.File(real_h5,'r')
    real_keys = list(realh5[lv1_name].keys())
    print(f"start save real images({len(real_keys)})...")
    for idx, real_key in enumerate(real_keys):
        # print(f"({lv1_name},{real_key})")
        data = realh5[f'{lv1_name}/{real_key}{level_append}'][()].squeeze()

        if isrgb:
            if data.shape[0] == 3:
                data = np.moveaxis(data, 0, -1)
            im = PIL.Image.fromarray(data).convert("RGB").resize((256, 256))
            im.save(os.path.join(real_path, str(idx) + ".png"))
        else:

            if data.shape[0] == 3 or len(data.shape) == 2 :
                data = adjust_dynamic_range(data, [data.min(),data.max()],[0,255]).astype("uint8")
                if data.shape[0]==3:
                    data = np.moveaxis(data,0,-1)
                    # print(f"move axis as:{data.shape}")
                im = PIL.Image.fromarray(data).convert("RGB").resize((256,256))
                im.save(os.path.join(real_path, str(idx)+".png"))
            else:
                if idx == 0:
                    print(f"real img shape is:{data.shape}, save multiple channels")
                for ch in range(data.shape[0]):
                    data[ch] = adjust_dynamic_range(data[ch], [data[ch].min(),data[ch].max()],[0,255]).astype("uint8")
                    im_ch = PIL.Image.fromarray(data[ch]).convert("RGB").resize((256,256))
                    im_ch.save(os.path.join(real_path, str(idx)+"_"+str(ch)+".png"))

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

        result = fid_score.calculate_fid_given_paths([str(real_path), str(fake_path)], 256, 0, 2048)
        print(f"***FID score: {result}***")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate FID"
    )
    parser.add_argument(
        "--real_h5", type=str, required=True, help="path to the real h5 data"
    )
    # parser.add_argument(
    #     "--fake_h5", type=str, required=True, help="path to the fake h5 data"
    # )
    parser.add_argument(
        "--fake_h5", nargs="+", help="paths to the fake h5 data"
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

    calculate_fid_from_h5(args.real_h5, args.fake_h5, lv1_name=args.lv1_name, oldformat=args.oldformat, fake_h5_ch=args.fake_h5_ch, isrgb=args.isrgb)

