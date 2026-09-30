import numpy as np
from qrwatermark.proposed.side_info import build_side_info, unpack_period_indices


def test_period_side_info_pack_and_authenticate():
    codes=np.asarray([0,1,2,1,0,2,2,1,0],dtype=np.uint8)
    side=build_side_info(codes,{"watermark_shape":[3,3],"period_count":3},b'key')
    recovered,meta=unpack_period_indices(side,b'key')
    assert np.array_equal(codes,recovered)
    assert meta['period_code_bits']==2
    assert len(side['packed_period_codes']) == 3  # 9*2=18 bits -> 3 bytes


def test_certified_mask_is_authenticated():
    from qrwatermark.proposed.side_info import unpack_certified_mask
    codes=np.asarray([0,1,2,3,1,0],dtype=np.uint8)
    mask=np.asarray([1,0,1,1,0,0],dtype=np.uint8)
    side=build_side_info(codes,{"watermark_shape":[2,3],"period_count":4},b'key',certified_mask=mask)
    assert np.array_equal(unpack_certified_mask(side,b'key'),mask.astype(bool))
    tampered=dict(side); tampered["packed_certified_mask"]=side["packed_certified_mask"].copy(); tampered["packed_certified_mask"][0]^=1
    import pytest
    with pytest.raises(ValueError,match="authentication"):
        unpack_certified_mask(tampered,b'key')
