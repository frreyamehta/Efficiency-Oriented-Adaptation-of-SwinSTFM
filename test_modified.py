import os
import random
import argparse
import rasterio
import numpy as np
from tqdm import tqdm
from sewar import rmse, ssim, sam

import torch

from datasets.data_1 import load_image_pair, transform_image
from models.swin_nano import SwinNanoSTFM   # <- your SwinNano class


def uiqi(im1, im2, block_size=64, return_map=False):
    if len(im1.shape) == 3:
        return np.array([uiqi(im1[:, :, i], im2[:, :, i], block_size, return_map=return_map)
                         for i in range(im1.shape[2])])
    delta_x = np.std(im1, ddof=1)
    delta_y = np.std(im2, ddof=1)
    # unbiased covariance estimator
    delta_xy = np.sum((im1 - np.mean(im1)) * (im2 - np.mean(im2))) / (im1.shape[0] * im1.shape[1] - 1)
    mu_x = np.mean(im1)
    mu_y = np.mean(im2)
    q1 = delta_xy / (delta_x * delta_y + 1e-100)
    q2 = 2 * mu_x * mu_y / (mu_x ** 2 + mu_y ** 2 + 1e-100)
    q3 = 2 * delta_x * delta_y / (delta_x ** 2 + delta_y ** 2 + 1e-100)
    q = q1 * q2 * q3
    return q


def compute_psnr(img_true, img_pred, data_range=1.0):
    # img_true, img_pred: numpy arrays (H,W) or (H,W,C) with values in [0, data_range]
    mse = np.mean((img_true - img_pred) ** 2)
    if mse <= 0:
        return float('inf')
    return 10.0 * np.log10((data_range ** 2) / (mse + 1e-12))


def test(opt, model, test_dates, IMAGE_SIZE, PATCH_SIZE, device):
    model.eval()

    PATCH_STRIDE = PATCH_SIZE // 2
    end_h = (IMAGE_SIZE[0] - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
    end_w = (IMAGE_SIZE[1] - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
    h_index_list = [i for i in range(0, end_h, PATCH_STRIDE)]
    w_index_list = [i for i in range(0, end_w, PATCH_STRIDE)]
    if (IMAGE_SIZE[0] - PATCH_STRIDE) % PATCH_STRIDE != 0:
        h_index_list.append(IMAGE_SIZE[0] - PATCH_SIZE)
    if (IMAGE_SIZE[1] - PATCH_STRIDE) % PATCH_STRIDE != 0:
        w_index_list.append(IMAGE_SIZE[1] - PATCH_SIZE)

    total_image = 0
    for cur_date in test_dates:
        cur_day = int(cur_date.split('_')[1])
        if cur_day == 347:
            for ref_date in test_dates:
                if ref_date == cur_date:
                    continue

                images = load_image_pair(opt.root_dir, cur_date, ref_date)
                # images expected shape: (C, H, W) per entry (authors' format)
                output_image = np.zeros(images[1].shape, dtype=np.float32)
                image_mask = np.ones(images[1].shape, dtype=np.float32)

                for i in range(4):
                    negtive_mask = np.where(images[i] < 0)
                    inf_mask = np.where(images[i] > 10000.)
                    image_mask[negtive_mask] = 0
                    image_mask[inf_mask] = 0

                # sliding-window inference, identical to original logic
                for i in range(len(h_index_list)):
                    for j in range(len(w_index_list)):
                        h_start = h_index_list[i]
                        w_start = w_index_list[j]

                        input_lr = images[0][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]
                        target_hr = images[1][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]
                        ref_lr = images[2][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]
                        ref_hr = images[3][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]

                        # no augmentation (same as authors)
                        flip_num = 0
                        rotate_num0 = 0
                        rotate_num = 0
                        input_lr, _ = transform_image(input_lr, flip_num, rotate_num0, rotate_num)
                        target_hr, _ = transform_image(target_hr, flip_num, rotate_num0, rotate_num)
                        ref_lr, _ = transform_image(ref_lr, flip_num, rotate_num0, rotate_num)
                        ref_hr, _ = transform_image(ref_hr, flip_num, rotate_num0, rotate_num)

                        # send to device
                        input_lr = input_lr.unsqueeze(0).to(device)
                        ref_lr = ref_lr.unsqueeze(0).to(device)
                        ref_hr = ref_hr.unsqueeze(0).to(device)

                        # Forward signature in your SwinNanoSTFM is (c0, f0, c1).
                        # The original authors called model(ref_lr, ref_hr, input_lr).
                        # That order already maps to (c0=ref_lr, f0=ref_hr, c1=input_lr)
                        with torch.no_grad():
                            output = model(ref_lr, ref_hr, input_lr)

                        # squeeze and move to cpu numpy
                        output = output.squeeze().cpu().detach().numpy()

                        h_end = h_start + PATCH_SIZE
                        w_end = w_start + PATCH_SIZE
                        cur_h_start = 0
                        cur_h_end = PATCH_SIZE
                        cur_w_start = 0
                        cur_w_end = PATCH_SIZE

                        if i != 0:
                            h_start = h_start + PATCH_SIZE // 4
                            cur_h_start = PATCH_SIZE // 4

                        if i != len(h_index_list) - 1:
                            h_end = h_end - PATCH_SIZE // 4
                            cur_h_end = cur_h_end - PATCH_SIZE // 4

                        if j != 0:
                            w_start = w_start + PATCH_SIZE // 4
                            cur_w_start = PATCH_SIZE // 4

                        if j != len(w_index_list) - 1:
                            w_end = w_end - PATCH_SIZE // 4
                            cur_w_end = cur_w_end - PATCH_SIZE // 4

                        output_image[:, h_start: h_end, w_start: w_end] = \
                            output[:, cur_h_start: cur_h_end, cur_w_start: cur_w_end]

                # scale and mask (same as authors)
                real_im = images[1] * 0.0001 * image_mask           # ground truth in [0,1] range
                real_output = (output_image + 1) * 0.5 * image_mask  # model outputs mapped to [0,1]

                # compute metrics per band
                rmse_list = []
                r2_list = []
                ssim_list = []
                cc_list = []
                uiqi_list = []
                ergas_val = 0.0
                psnr_list = []

                for b in range(6):
                    band_true = real_im[b]
                    band_pred = real_output[b]

                    rm = rmse(band_true, band_pred)
                    ss, _ = ssim(band_true, band_pred, MAX=1.0)
                    uq = uiqi(band_true, band_pred)
                    
                    #r^2
                    ss_res = np.sum((band_true - band_pred) ** 2)
                    ss_tot = np.sum((band_true - np.mean(band_true)) ** 2) + 1e-100
                    r2_val = 1 - (ss_res / ss_tot)
                    r2_list.append(r2_val)

                    # correlation coefficient (Pearson)
                    num = np.sum((band_true - np.mean(band_true)) * (band_pred - np.mean(band_pred)))
                    den = np.sqrt(np.sum((band_true - np.mean(band_true)) ** 2) * np.sum((band_pred - np.mean(band_pred)) ** 2) + 1e-100)
                    cc_val = num / (den + 1e-100)

                    rmse_list.append(rm)
                    ssim_list.append(ss)
                    uiqi_list.append(uq)
                    cc_list.append(cc_val)

                    ergas_val += (rm ** 2) / (np.mean(band_true) ** 2 + 1e-100)

                    # PSNR on [0,1] scaled images
                    psnr_list.append(compute_psnr(band_true, band_pred, data_range=1.0))

                ergas_val = np.sqrt(ergas_val / 6.0) * 6.0

                # SAM expects scaled intensities (authors multiply by 10000)
                cur_im = real_im * 10000.0
                cur_predict = real_output * 10000.0
                sam_val = sam(cur_im.transpose(1, 2, 0), cur_predict.transpose(1, 2, 0)) * 180.0 / np.pi

                print('[%s/%s] RMSE: %.4f R2: %.4f SSIM: %.4f SAM: %.4f PSNR: %.4f ERGAS %.4f UIQI: %.4f CC: %.4f' % (
                    cur_date, ref_date,
                    np.mean(np.array(rmse_list)),
                    np.mean(np.array(r2_list)),
                    np.mean(np.array(ssim_list)),
                    sam_val,
                    np.mean(np.array(psnr_list)),
                    ergas_val,
                    np.mean(np.array(uiqi_list)),
                    np.mean(np.array(cc_list))
                ))

                # original saving logic (unchanged)
                ref_day = int(ref_date.split('_')[1])
                total_image += cur_predict
                if ref_day != 363:
                    final_im = cur_predict.astype(np.int16)
                    metadata = {
                        'driver': 'GTiff',
                        'width': final_im.shape[2],
                        'height': final_im.shape[1],
                        'count': final_im.shape[0],
                        'dtype': np.int16
                    }
                    save_dir = '/Users/frreyamehta/Documents/CV_Lab/mini_project/swinstfm/paper_images'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    im_name = os.path.join(save_dir, 'swin_LGC_1.tif')
                    assert final_im.ndim == 2 or final_im.ndim == 3
                    with rasterio.open(im_name, mode='w', **metadata) as dst:
                        if final_im.ndim == 3:
                            for k in range(final_im.shape[0]):
                                dst.write(final_im[k], k + 1)
                        else:
                            dst.write(final_im, 1)

    # restore total_image behaviour from original script
    total_image = np.array(total_image)
    if total_image.ndim == 0:
        total_image = total_image.reshape(1, 1, 1)
    elif total_image.ndim == 2:
        total_image = total_image[:, :, np.newaxis]

    final_im = (total_image / 2.0).astype(np.int16)
    metadata = {
        'driver': 'GTiff',
        'width': final_im.shape[2],
        'height': final_im.shape[1],
        'count': final_im.shape[0],
        'dtype': np.int16
    }
    save_dir = os.path.join(os.getcwd(), 'paper_images')
    os.makedirs(save_dir, exist_ok=True)
    im_name = os.path.join(save_dir, 'swinfuse_LGC_1.tif')
    assert final_im.ndim == 2 or final_im.ndim == 3
    with rasterio.open(im_name, mode='w', **metadata) as dst:
        if final_im.ndim == 3:
            for k in range(final_im.shape[0]):
                dst.write(final_im[k], k + 1)
        else:
            dst.write(final_im, 1)


def train(opt, train_dates, test_dates, IMAGE_SIZE, PATCH_SIZE, device):
    # This function here only bootstraps model & loads checkpoint like the original script did.
    model_G = SwinNanoSTFM()
    G_dict = model_G.state_dict()

    ckpt_path = opt.checkpoint
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    model_CKPT = torch.load(ckpt_path, map_location='cpu')

    # ckpt may be a state_dict or may contain keys; handle both
    if isinstance(model_CKPT, dict) and ('state_dict' in model_CKPT):
        loaded_state = model_CKPT['state_dict']
    else:
        loaded_state = model_CKPT

    pretained_dict = {k: v for k, v in loaded_state.items() if k in G_dict}
    G_dict.update(pretained_dict)
    model_G.load_state_dict(G_dict)
    model_G.to(device)

    test(opt, model_G, test_dates, IMAGE_SIZE, PATCH_SIZE, device)


def main():
    # set seeds
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description='Test SwinNano on LGC')
    parser.add_argument('--image_size', default=[2720, 3200], type=int, help='the image size (height, width)')
    parser.add_argument('--patch_size', default=256, type=int, help='patch size')
    parser.add_argument('--root_dir', default='/Users/frreyamehta/Documents/CV_Lab/mini_project/swinstfm/LGC', help='Datasets root directory')
    parser.add_argument('--checkpoint', default='/Users/frreyamehta/Documents/CV_Lab/mini_project/swinstfm/models/experiment_best_1/epoch_best.pth', help='checkpoint path (state_dict)')
    opt = parser.parse_args()

    IMAGE_SIZE = opt.image_size
    PATCH_SIZE = opt.patch_size

    # device: MPS if available, else CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # collect test dates (same selection rules as original)
    train_dates = []
    test_dates = []
    for dir_name in os.listdir(opt.root_dir):
        if dir_name.startswith('.'):
            continue
        parts = dir_name.split('_')
        if len(parts) < 2:
            continue
        if not parts[1].isdigit():
            continue
        cur_day = int(parts[1])
        if cur_day in [331, 347, 363]:
            test_dates.append(dir_name)

    train(opt, train_dates, test_dates, IMAGE_SIZE, PATCH_SIZE, device)


if __name__ == '__main__':
    main()

