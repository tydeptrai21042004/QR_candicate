import numpy as np
from qrwatermark.evaluation.metrics import ber,nc


def test_metrics_do_not_auto_invert_extracted_watermark():
    a=np.array([[0,255],[255,0]],dtype=np.uint8)
    inv=255-a
    assert ber(a,inv)==1.0
    assert nc(a,inv)==0.0
