from scipy.ndimage import zoom
import numpy as np


NBITS = [4, 16, 32, 64, 64, 64]


def regrain(img_arr_in, img_arr_col, smoothness: float = 1):
    """keep gradient of img_arr_in and color of img_arr_col."""

    img_arr_in = img_arr_in / 255.0
    img_arr_col = img_arr_col / 255.0
    img_arr_out = np.array(img_arr_in)
    img_arr_out = _regrain_rec(
        img_arr_out, img_arr_in, img_arr_col, NBITS, 0, smoothness
    )
    img_arr_out[img_arr_out < 0] = 0
    img_arr_out[img_arr_out > 1] = 1
    img_arr_out = (255.0 * img_arr_out).astype("uint8")
    return img_arr_out


def _regrain_rec(img_arr_out, img_arr_in, img_arr_col, nbits, level, smoothness):
    [h, w, _] = img_arr_in.shape
    h2 = (h + 1) // 2
    w2 = (w + 1) // 2
    if len(nbits) > 1 and h2 > 20 and w2 > 20:
        resize_arr_in = _resize_image(img_arr_in, w2, h2)
        resize_arr_col = _resize_image(img_arr_col, w2, h2)
        resize_arr_out = _resize_image(img_arr_out, w2, h2)
        resize_arr_out = _regrain_rec(
            resize_arr_out,
            resize_arr_in,
            resize_arr_col,
            nbits[1:],
            level + 1,
            smoothness,
        )
        img_arr_out = _resize_image(resize_arr_out, w, h)
    img_arr_out = _solve(
        img_arr_out, img_arr_in, img_arr_col, nbits[0], level, smoothness
    )
    return img_arr_out


def _solve(img_arr_out, img_arr_in, img_arr_col, nbit, level, smoothness):
    [width, height, c] = img_arr_in.shape
    first_pad_0 = lambda arr: np.concatenate((arr[:1, :], arr[:-1, :]), axis=0)
    first_pad_1 = lambda arr: np.concatenate((arr[:, :1], arr[:, :-1]), axis=1)
    last_pad_0 = lambda arr: np.concatenate((arr[1:, :], arr[-1:, :]), axis=0)
    last_pad_1 = lambda arr: np.concatenate((arr[:, 1:], arr[:, -1:]), axis=1)

    delta_x = last_pad_1(img_arr_in) - first_pad_1(img_arr_in)
    delta_y = last_pad_0(img_arr_in) - first_pad_0(img_arr_in)
    delta = np.sqrt((delta_x**2 + delta_y**2).sum(axis=2, keepdims=True))

    psi = 256 * delta / 5
    psi[psi > 1] = 1
    phi = 30 * 2 ** (-level) / (1 + 10 * delta / smoothness)

    phi1 = (last_pad_1(phi) + phi) / 2
    phi2 = (last_pad_0(phi) + phi) / 2
    phi3 = (first_pad_1(phi) + phi) / 2
    phi4 = (first_pad_0(phi) + phi) / 2

    rho = 1 / 5.0
    for i in range(nbit):
        den = psi + phi1 + phi2 + phi3 + phi4
        num = (
            np.tile(psi, [1, 1, c]) * img_arr_col
            + np.tile(phi1, [1, 1, c])
            * (last_pad_1(img_arr_out) - last_pad_1(img_arr_in) + img_arr_in)
            + np.tile(phi2, [1, 1, c])
            * (last_pad_0(img_arr_out) - last_pad_0(img_arr_in) + img_arr_in)
            + np.tile(phi3, [1, 1, c])
            * (first_pad_1(img_arr_out) - first_pad_1(img_arr_in) + img_arr_in)
            + np.tile(phi4, [1, 1, c])
            * (first_pad_0(img_arr_out) - first_pad_0(img_arr_in) + img_arr_in)
        )
        img_arr_out = (
            num / np.tile(den + 1e-6, [1, 1, c]) * (1 - rho) + rho * img_arr_out
        )
    return img_arr_out


def _resize_image(data, target_width, target_height):
    return zoom(data, (target_height / data.shape[0], target_width / data.shape[1], 1))
