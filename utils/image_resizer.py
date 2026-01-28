import os
import numpy as np
from kitti_loader import find_image_files

from PIL import Image, ImageOps

def image_resize(img, target_size=(1200, 375)):
    """
    Crop the image to match the target aspect ratio, then resize.
    Returns uint8 RGB output.
    """

    # Convert torch → numpy
    try:
        import torch
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
    except:
        pass

    arr = np.array(img)

    # (C,H,W) → (H,W,C)
    if arr.ndim == 3 and arr.shape[0] in (1,3,4) and arr.shape[0] != arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))

    # grayscale → 3-channel
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)

    # float → uint8
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0,255)
        else:
            arr = arr.clip(0,255)
        arr = arr.astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)

    arr = np.ascontiguousarray(arr)

    # PIL object
    pil_img = Image.fromarray(arr)

    # CROP + RESIZE
    out = ImageOps.fit(
        pil_img,
        target_size,          # (width, height)
        method=Image.LANCZOS, # best quality
        centering=(0.5, 0.5)  # center crop
    )

    return np.array(out)


def resize_and_save_split(root_dir, split='train', target_size=(1200, 375), out_subfolder='image_2_resized', verbose=True):
    """
    Resize/crop images for one split and save them preserving filenames and formats.

    - root_dir: path to dataset folder (same as ROOTDIR used by find_image_files)
    - split: 'train' or 'test'  (find_image_files treats 'train' as training; anything else -> testing)
    - target_size: (width, height) passed to image_resize
    - out_subfolder: name of the folder to create next to image_2 (inside training/testing)
    """
    # determine training/testing directory names consistent with your find_image_files
    split_dirname = 'training' if split == 'train' else 'testing'

    # source image_2 folder (where original images live)
    src_base = os.path.join(root_dir, 'data_object_image_2', split_dirname, 'image_2')
    if not os.path.exists(src_base):
        raise FileNotFoundError(f"Source folder not found: {src_base}")

    # destination folder (image_2_resized inside same training/testing folder)
    dest_base = os.path.join(root_dir, 'image_processed', split_dirname, out_subfolder)
    os.makedirs(dest_base, exist_ok=True)

    # get list of files using the provided function (keeps original ordering)
    image_files = find_image_files(root_dir, split=split if split == 'train' else 'test')

    saved = 0
    for src_path in image_files:
        # keep the same filename
        fname = os.path.basename(src_path)
        dest_path = os.path.join(dest_base, fname)

        try:
            # Load with PIL to preserve file format, exif etc.
            with Image.open(src_path) as im:
                # force RGB so we have consistent 3 channels (image_resize expects arrays)
                im_rgb = im.convert('RGB')
                arr = np.array(im_rgb)

            # call your image_resize (crop + resize, returns uint8 HxWxC)
            resized_arr = image_resize(arr, target_size=target_size)

            # convert to PIL and save with same extension/format
            out_im = Image.fromarray(resized_arr)
            # infer format from file extension
            ext = os.path.splitext(fname)[1].lower()
            if ext in ('.jpg', '.jpeg'):
                out_im.save(dest_path, format='JPEG', quality=95)
            elif ext == '.png':
                out_im.save(dest_path, format='PNG')
            else:
                # fallback: use PNG for unknown extensions but keep filename
                out_im.save(dest_path, format='PNG')

            saved += 1
            if verbose and saved % 100 == 0:
                print(f"Saved {saved} images so far...")

        except Exception as e:
            # don't crash the whole run if one image fails
            print(f"Failed to process {src_path}: {e}")

    if verbose:
        print(f"Done. Saved {saved} images to: {dest_base}")
    return saved

ROOTDIR = r"F:\Work\DeepLearning\Research\dataset"   # set to your dataset root
# resize training images
resize_and_save_split(ROOTDIR, split='train', target_size=(1200, 375))
# resize testing images
resize_and_save_split(ROOTDIR, split='test', target_size=(1200, 375))
