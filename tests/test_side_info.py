import numpy as np
from qrwatermark.proposed.side_info import build_side_info, unpack_period_indices


def test_period_side_info_pack_and_authenticate():
    codes=np.asarray([0,1,2,1,0,2,2,1,0],dtype=np.uint8)
    side=build_side_info(codes,{"watermark_shape":[3,3],"period_count":3},b'key')
    recovered,meta=unpack_period_indices(side,b'key')
    assert np.array_equal(codes,recovered)
    assert meta['period_code_bits']==2
    assert len(side['packed_period_codes']) == 3  # 9*2=18 bits -> 3 bytes
