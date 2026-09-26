import numpy as np
from qrwatermark.attacks import apply_attack


def test_standard_attacks_preserve_shape():
    img=np.full((64,64,3),127,dtype=np.uint8)
    cases=[('gaussian_blur',{'sigma':1}),('gaussian_noise',{'variance':0.003,'seed':0}),('salt_pepper',{'density':0.1,'seed':0}),('jpeg',{'quality':50}),('scale_resample',{'scale':0.5}),('rotation_resample',{'angle':10}),('gamma',{'gamma':1.5}),('occlusion',{'fraction':0.2,'seed':0})]
    for name,params in cases:
        out=apply_attack(name,img,**params)
        assert out.shape==img.shape and out.dtype==np.uint8
