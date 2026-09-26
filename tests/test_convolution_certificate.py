import numpy as np
from qrwatermark.proposed.convolution import gaussian_kernel, circular_convolve, spectral_distance_from_identity
from qrwatermark.proposed.certificate import theoretical_group_mse


def test_circular_convolution_constant_fixed_point():
    x=np.full((32,32),17.0)
    k=gaussian_kernel(3,0.75)
    y=circular_convolve(x,k)
    assert np.max(np.abs(y-x)) < 1e-10


def test_spectral_identity_distance_positive_for_blur():
    k=gaussian_kernel(3,0.75)
    eta=spectral_distance_from_identity(k,(32,32))
    assert 0.0 < eta < 2.0


def test_exact_r12_distortion_model_nonnegative():
    values=np.asarray([10.0,20.0,30.0])
    d,ok=theoretical_group_mse(values,48.0)
    assert d >= 0.0 and ok
