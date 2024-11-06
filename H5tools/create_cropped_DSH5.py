"""
This script creates the HDF5 files from the zip files downloaded from UK Biobank.
This is a generic version, was used for the creation of the HDF5 files for the short and long axis heart images.
"""

import numpy as np
import h5py
import argparse
import os

def pre_pad(vol, newshape):
    pad = np.array(newshape) - np.array(vol.shape[-2:])
    pad[pad<0] = 0
    pad = pad//2
    pad = np.array([[0,0]]*(len(vol.shape)-2) +  [[pad[0], pad[0]], [pad[1], pad[1]]])
    return np.pad(vol, pad, mode='constant')

def contour_crop(vol, contour, exact=True, newshape=None):
    idx = np.nonzero(contour)

    if exact:
        return vol[..., idx[-2].min():idx[-2].max(), idx[-1].min():idx[-1].max()]
    
    if (np.array(newshape) - np.array(vol.shape[-2:]) > 0).any(): #if any one of them has a smaller dimension, then we would pad it to make it 128x128
        vol = pre_pad(vol, newshape)
        idx = np.nonzero(contour)
    
    mid_contour = (idx[-2].min()+idx[-2].max())//2, (idx[-1].min()+idx[-1].max())//2
    mid_newshape = newshape[0]//2, newshape[1]//2

    if mid_contour[0]-mid_newshape[0] < 0:
        mid_contour = (mid_newshape[0], mid_contour[1])

    if mid_contour[1]-mid_newshape[1] < 0:
        mid_contour = (mid_contour[0], mid_newshape[1])

    if mid_contour[0]-mid_newshape[0]+newshape[0] > vol.shape[-2]:
        diff = (mid_contour[0]-mid_newshape[0]+newshape[0]) - vol.shape[-2]
        mid_contour = (mid_contour[0]-diff, mid_contour[1])

    if mid_contour[1]-mid_newshape[1]+newshape[1] > vol.shape[-1]:
        diff = (mid_contour[1]-mid_newshape[1]+newshape[1]) - vol.shape[-1]
        mid_contour = (mid_contour[0], mid_contour[1]-diff)

    return vol[..., mid_contour[0]-mid_newshape[0]:mid_contour[0]-mid_newshape[0]+newshape[0], mid_contour[1]-mid_newshape[1]:mid_contour[1]-mid_newshape[1]+newshape[1]]


parser = argparse.ArgumentParser()
parser.add_argument("--in_path", help="path to store the HDF5 file", default=r"../ukbbH5s/_tmp_newV3/F20208_Long_axis_heart_images_DICOM_H5")
parser.add_argument("--acq_filter", help="any particular acquisition to consider", default="LAX_4Ch_transverse")
parser.add_argument("--mode", type=int, help="0: fetch all the subject IDs, 1: create subset of the dataset and return the subject IDs based on the number of cardiac cycles (for heart 208 and 209)", default=0)
parser.add_argument("--exact_mask_crop", action=argparse.BooleanOptionalAction, help="If True, they will be cropped with exact shape of the mask (ignoring shapeX and shapeY)", default=False)
parser.add_argument("--shapeX", type=int, help="Desired shape X", default=128)
parser.add_argument("--shapeY", type=int, help="Desired shape Y", default=128)
args = parser.parse_args()

out_path = args.in_path

if bool(args.acq_filter):
    out_path += f"_{args.acq_filter}"

if args.exact_mask_crop:
    out_path += "_fitcropped"
else:
    out_path += f"_cropped_{args.shapeX}_{args.shapeY}"

os.makedirs(out_path, exist_ok=True)
cropped_file = h5py.File(f"{out_path}/data.h5", 'w')

mask_file = h5py.File(f"{args.in_path}/meta_mask.h5", 'r')

def crop_with_heartmask(name, obj):
    if bool(args.acq_filter) and args.acq_filter not in name:
        return

    if isinstance(obj, h5py.Dataset):
        try:
            _crop_with_heartmask_single(name, obj)
        except Exception as e:
            print(f"Error for {name}: {e}")

def _crop_with_heartmask_single(name, obj):
    contour = mask_file[name][:]

    cropped = contour_crop(obj, contour, exact=args.exact_mask_crop, newshape=(args.shapeX, args.shapeY))

    if not args.exact_mask_crop:
        assert cropped.shape[-2] == args.shapeX and cropped.shape[-1] == args.shapeY, f"Shape mismatch: {cropped.shape} vs {args.shapeX}x{args.shapeY}, for {name} with original shape {obj.shape}"

    path_parts = name.split('/')
    current_mskgroup = cropped_file
    for part in path_parts[:-1]:
        if part not in current_mskgroup:
            current_mskgroup.create_group(part)
        current_mskgroup = current_mskgroup[part]

    dset = current_mskgroup.create_dataset(path_parts[-1], data=cropped)
    dset.attrs.update(obj.attrs)
    dset.attrs["min_val"] = cropped.min()
    dset.attrs["max_val"] = cropped.max()

with h5py.File(f"{args.in_path}/data.h5", 'r') as f:    
    if args.mode == 0: 
        f.visititems(crop_with_heartmask)


print("Done!")