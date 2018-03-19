"""
Authors: Eugene Belilovsky, Edouard Oyallon and Sergey Zagoruyko
All rights reserved, 2017.
"""

__all__ = ['Scattering']

import warnings

import torch

from .utils import cdgmm, Modulus, Periodize, Fft
from .filters_bank import filters_bank
from torch.legacy.nn import SpatialReflectionPadding


class Scattering(object):
    """
    Scattering.

    Runs scattering on an input image in NCHW format

    Input args:
        M, N: input image size
        J: number of layers
        pre_pad: if set to True, module expect pre-padded images
        jit: compile kernels on the fly for speed
    """
    def __init__(self, M, N, J, pre_pad=False, jit=True):
        super(Scattering, self).__init__()
        self.M, self.N, self.J = M, N, J
        self.pre_pad = pre_pad
        self.jit = jit
        self.fft = Fft()
        self.modulus = Modulus(jit=jit)
        self.periodize = Periodize(jit=jit)

        self._prepare_padding_size([1, 1, M, N])

        self.padding_module = SpatialReflectionPadding(pow(2, J))

        # Create the filters
        filters = filters_bank(self.M_padded, self.N_padded, J)

        self.Psi = filters['psi']
        self.Phi = [filters['phi'][j] for j in range(J)]

    def _type(self, _type):
        for key, item in enumerate(self.Psi):
            for key2, item2 in self.Psi[key].items():
                if torch.is_tensor(item2):
                    self.Psi[key][key2] = item2.type(_type)
        self.Phi = [v.type(_type) for v in self.Phi]
        self.padding_module.type(str(_type).split('\'')[1])
        return self

    def cuda(self):
        return self._type(torch.cuda.FloatTensor)

    def cpu(self):
        return self._type(torch.FloatTensor)

    def _prepare_padding_size(self, s):
        M, N = s[-2:]

        self.M_padded = ((M + 2 ** (self.J))//pow(2, self.J)+1)*pow(2, self.J)
        self.N_padded = ((N + 2 ** (self.J))//pow(2, self.J)+1)*pow(2, self.J)

        if self.pre_pad:
            warnings.warn(
                'Make sure you padded the input before hand!',
                RuntimeWarning,
                stacklevel=2
            )

        s[-2] = self.M_padded
        s[-1] = self.N_padded
        self.padded_size_batch = torch.Size([a for a in s])

    def pad(self, src):
        """
            This function copies and views the real to complex.
        """
        if(self.pre_pad):
            dst = src.new(
                src.size(0),
                src.size(1),
                src.size(2),
                src.size(3),
                2
            ).zero_()
            dst.narrow(dst.ndimension()-1, 0, 1).copy_(
                torch.unsqueeze(src, 4)
            )
        else:
            padded = self.padding_module.updateOutput(src)
            dst = src.new(
                padded.size(0),
                padded.size(1),
                padded.size(2),
                padded.size(3),
                2
            ).zero_()
            dst.narrow(4, 0, 1).copy_(
                torch.unsqueeze(padded, 4)
            )
        return dst

    def unpad(self, src):
        return src[..., 1:-1, 1:-1]

    def pre_checks(self, src):
        if src.dim() != 4:
            raise RuntimeError('Input tensor must be 4D')
        if not torch.is_tensor(src):
            raise TypeError(
                'The src should be a torch.cuda.FloatTensor, a '
                + 'torch.FloatTensor or a torch.DoubleTensor'
            )
        if not src.is_contiguous():
            raise RuntimeError('Tensor must be contiguous!')
        if(
            (src.size(-1) != self.N or src.size(-2) != self.M)
            and not self.pre_pad
        ):
            raise RuntimeError(
                'Tensor must be of spatial size {}!'.format((self.M, self.N))
            )
        if(
            (
                src.size(-1) != self.N_padded
                or src.size(-2) != self.M_padded
            )
            and self.pre_pad
        ):
            raise RuntimeError(
                'Padded tensor must be of spatial size {}!'.format(
                    (self.M_padded, self.N_padded)
                )
            )

    def second_level_scatter(self, psi, U):
        print(len(self.Psi))
        U_1_c = self.fft(
            self.modulus(
                self.fft(
                    self.periodize(
                        cdgmm(U, psi[0], jit=self.jit),
                        k=pow(2, psi['j'])
                    ) if psi['j'] > 0 else cdgmm(U, psi[0], jit=self.jit),
                    'C2C',
                    inverse=True
                )
            ),
            'C2C'
        )
        return [
            self.unpad(
                self.fft(
                    self.periodize(
                        cdgmm(U_1_c, self.Phi[j1], jit=self.jit),
                        k=pow(2, self.J - j1)
                    ),
                    'C2R'
                )
            ).unsqueeze(2)
        ] + [
                self.unpad(
                    self.fft(
                        self.periodize(
                            cdgmm(
                                self.fft(
                                    self.modulus(
                                        self.fft(
                                            self.periodize(
                                                cdgmm(
                                                    U_1_c,
                                                    _psi[psi['j']],
                                                    jit=self.jit
                                                ),
                                                k=pow(2, _psi['j'] - psi['j'])
                                            ),
                                            'C2C',
                                            inverse=True
                                        )
                                    ),
                                    'C2C'
                                ),
                                self.Phi[_psi['j']],
                                jit=self.jit
                            ),
                            k=pow(2, self.J - _psi['j'])
                        ),
                        'C2R'
                    )
                ).unsqueeze(2)
                for _psi in self.Psi
                if psi['j'] < _psi['j']
        ]

    def forward(self, input_tensor):
        self.pre_checks(input_tensor)
        U_0 = self.fft(
            self.pad(input_tensor),
            'C2C'
        )

        return torch.cat(
            [
                self.unpad(
                    self.fft(
                        self.periodize(
                            cdgmm(U_0, self.Phi[0], jit=self.jit),
                            k=pow(2, self.J)
                        ),
                        'C2R'
                    )
                ).unsqueeze(2)
            ] + [
                self.second_level_scatter(U_0, psi)
                for psi in self.Psi
            ],
            dim=2
        )

    def __call__(self, input_tensor):
        return self.forward(input_tensor)
