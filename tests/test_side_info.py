import numpy as np
import pytest
from qrwatermark.proposed.side_info import build_side_info,unpack_modes


def test_side_info_pack_and_authenticate():
    modes=np.array([0,1,1,0,1,0,0,1,1],dtype=np.uint8)
    side=build_side_info(modes,{"watermark_shape":[3,3]},b'key')
    decoded,_=unpack_modes(side,b'key')
    assert np.array_equal(decoded,modes)
    with pytest.raises(ValueError):
        unpack_modes(side,b'wrong')
