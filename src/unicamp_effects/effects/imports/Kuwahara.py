
import numpy as np
from numba import njit, prange

from . import colorspaces

np.set_printoptions(precision=3, suppress=True)
#Parece inviável fazer tantos loops em python

def kuwahara_square_filter(img_raw, kernel_size):
    print("Square!")
    assert kernel_size % 2 == 1, "Kernel size must be odd"
    assert kernel_size > 0, "Kernel size must be positive"

    img = colorspaces.to_float(img_raw)
    img_color = colorspaces.rgb_to_oklab(img)
    height, width = img_color.shape[0], img_color.shape[1]
    half = kernel_size // 2

    new_image = np.zeros_like(img_color)

    for r in range(height):
        for c in range(width):
            sectors = [
                img_color[max(0, r - half):min(height, r + 1), max(0, c - half):min(width, c + 1)],
                img_color[max(0, r - half):min(height, r + 1), max(0, c):min(width, c + half + 1)],
                img_color[max(0, r):min(height, r + half + 1), max(0, c - half):min(width, c + 1)],
                img_color[max(0, r):min(height, r + half + 1), max(0, c):min(width, c + half + 1)],
            ]

            best_mean = sectors[0].reshape(-1, 3).mean(axis=0)
            best_var = sectors[0].reshape(-1, 3)[:, 0].var()

            for sector in sectors[1:]:
                flat = sector.reshape(-1, 3)
                var_l = flat[:, 0].var()
                print(var_l)
                if var_l < best_var:
                    best_var = var_l
                    best_mean = flat.mean(axis=0)

            new_image[r, c] = best_mean

    new_image = colorspaces.oklab_to_rgb(new_image)
    return colorspaces.to_uint8(np.clip(new_image, 0.0, 1.0))


def kuwahara_filter(img_raw, kernel_size):
    assert kernel_size % 2 == 1, "Kernel size must be odd"
    img = colorspaces.to_float(img_raw)
    img_color = colorspaces.rgb_to_oklab(img)
    del img

    new_image = _ku(img_color, kernel_size)
    new_image = colorspaces.oklab_to_rgb(new_image)
    return colorspaces.to_uint8(new_image)

@njit
def _ku(img_color, kernel_size):

    new_image = np.zeros_like(img_color)
    for r in range(img_color.shape[0]):
        #print(f"Processing row {r+1}/{img_color.shape[0]}", end="\r")
        for c in range(img_color.shape[1]):
            l = np.zeros((8, kernel_size * kernel_size, 3), dtype=np.float64)
            # print(f"l shape: {l.shape}")
            ind = np.zeros((8), dtype=np.int64)
            debug1 = 0
            debug2 = 0
            half = kernel_size // 2
            for j in range(-half, half +1):
                for k in range(-half, half+1):
                    if np.sqrt(np.square(j) + np.square(k)) > kernel_size//2:
                        continue
                    #pixel = img_gray[r, c]
                    pixel_color = img_color[r, c]
                    debug1 +=1
                    if not (r+j < 0 or r+j >= img_color.shape[0] or c+k < 0 or c+k >= img_color.shape[1]):
                        #pixel = img_gray[r + j, c + k]
                        pixel_color = img_color[r + j, c + k]
                        debug2 += 1
                    angle = np.arctan2(j, k)


                    if angle < 0:
                        angle = angle + 2*np.pi

                    for h in range(8):
                        if (j) == 0 and k == 0:
                            l[h, ind[h]] = pixel_color
                            ind[h] += 1
                            continue
                        if (h * np.pi/4.0 - 1e-7 <= angle <= (h + 1) * np.pi/4.0 + 1e-7) or (h * np.pi/4.0 - 1e-7 <= angle + 2*np.pi <= (h + 1) * np.pi/4.0 + 1e-7):
                            l[h, ind[h]] = pixel_color
                            ind[h] += 1
                            # if( h == 6):
                            #     print(f"{h}: ({k}, {j})")
            #print(ind)



            #std = np.empty(8)
            avg = np.zeros((8, 3))
            new_pixel = np.zeros(3, dtype = np.float64)
            min_std = 1e9
            for i in range(8):
                mean0 = 0.0

                for j in range(ind[0]):
                    v0 = l[i, j, 0]
                    mean0 += v0
                    avg[i] += l[i, j]
                mean0 /= ind[0]
                var = 0.0
                for j in range(ind[0]):
                    diff = l[i, j, 0] - mean0
                    var += diff * diff
                std_i = np.sqrt(var / ind[0])
                avg[i] /= ind[0]
                if (std_i < min_std):
                    min_std = std_i
                    new_pixel = avg[i]
                # std[i] = np.sqrt(var / ind[0])
            #print("avg:", avg)


            # print("std:", std, flush=True)
            # if debug1 == debug2:
            #     print("Debug!")
            #     raise Exception

            new_image[r, c] = new_pixel

    return new_image
