"""
Authors: Eugene Belilovsky, Edouard Oyallon and Sergey Zagoruyko
All rights reserved, 2017.
"""

__all__ = ['filters_bank']

import torch
import numpy as np
import scipy.fftpack as fft

import operator
import functools


def band_filter(fourier_signal, res):
    cropped_signal = crop_freq(fourier_signal, res)
    return torch.FloatTensor(
        np.stack(
            (
                np.real(cropped_signal),
                np.imag(cropped_signal)
            ),
            axis=2
        )
    )


def filters_bank(M, N, J, L=8, offset=0):
    return {
        'psi': [
            dict(
                [
                    ('j', j),
                    ('theta', theta)
                ] + [
                    (
                        res,
                        band_filter(
                            fft.fft2(
                                morlet_2d(
                                    M,
                                    N,
                                    4 * pow(2, j) / 5,
                                    (int(L-L/2-1)-theta) * np.pi / L,
                                    (3 * np.pi) / (4 * pow(2, j)),
                                    offset=offset
                                )
                            ),
                            res
                        )
                        /
                        (M * N // pow(2, 2*j))
                    )
                    for res in range(1 + j)
                ]
            )
            for j in range(J) for theta in range(L)
        ],
        'phi': dict(
            [('j', J)] + [
                (
                    res,
                    band_filter(
                        fft.fft2(
                            gabor_2d(
                                M,
                                N,
                                4 * pow(2, J - 1) / 5,
                                0,
                                0,
                                offset=offset
                            )
                        ),
                        res
                    ) / (M * N // pow(2, 2 * J))
                )
                for res in range(J)
            ]
        )
    }


def crop_freq(x, res):
    M, N = x.shape[:2]

    mask = np.ones(x.shape, np.float32)
    len_x = int(M * (1 - pow(2, -res)))
    start_x = int(M * pow(2, -res - 1))
    len_y = int(N * (1 - pow(2, -res)))
    start_y = int(N * pow(2, -res - 1))
    mask[
        start_x: start_x + len_x,
        :
    ] = 0
    mask[
        :,
        start_y: start_y + len_y
    ] = 0
    x = np.multiply(x, mask)

    crop = np.zeros(
        (M // pow(2, res), N // pow(2, res)),
        np.complex64
    )
    for k in range(int(M / pow(2, res))):
        for l in range(int(N / pow(2, res))):
            crop[k, l] = functools.reduce(
                operator.add,
                [
                    x[
                        k + i * int(M / pow(2, res)),
                        l + j * int(N / pow(2, res))
                    ]
                    for i in range(pow(2, res))
                    for j in range(pow(2, res))
                ],
                crop[k, l]
            )

    return crop


def morlet_2d(M, N, sigma, theta, xi, slant=0.5, offset=0, fft_shift=None):
    """ This function generated a morlet"""
    wv = gabor_2d(M, N, sigma, theta, xi, slant, offset, fft_shift)
    wv_modulus = gabor_2d(M, N, sigma, theta, 0, slant, offset, fft_shift)
    K = np.sum(wv) / np.sum(wv_modulus)

    mor = wv - K * wv_modulus
    return mor


def gabor_2d(M, N, sigma, theta, xi, slant=1.0, offset=0, fft_shift=None):
    gab = np.zeros((M, N), np.complex64)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float32)
    R_inv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], np.float32)
    D = np.array([[1, 0], [0, slant * slant]])
    curv = np.dot(R, np.dot(D, R_inv)) / ( 2 * sigma * sigma)

    for ex in [-2, -1, 0, 1, 2]:
        for ey in [-2, -1, 0, 1, 2]:
            [xx, yy] = np.mgrid[offset + ex * M:offset + M + ex * M, offset + ey * N:offset + N + ey * N]
            arg = -(curv[0, 0] * np.multiply(xx, xx) + (curv[0, 1] + curv[1, 0]) * np.multiply(xx, yy) + curv[
                1, 1] * np.multiply(yy, yy)) + 1.j * (xx * xi * np.cos(theta) + yy * xi * np.sin(theta))
            gab = gab + np.exp(arg)

    norm_factor = (2 * 3.1415 * sigma * sigma / slant)
    gab = gab / norm_factor

    if (fft_shift):
        gab = np.fft.fftshift(gab, axes=(0, 1))
    return gab
