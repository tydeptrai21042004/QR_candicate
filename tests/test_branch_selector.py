import inspect
import numpy as np
from qrwatermark.core.config import ProposedConfig,NLMConfig
from qrwatermark.proposed.certificate import choose_period_for_group


def test_period_selector_has_no_payload_bit_argument():
    params=inspect.signature(choose_period_for_group).parameters
    assert 'bit' not in params and 'watermark_bit' not in params


def test_period_selector_deterministic_and_payload_independent():
    cfg=ProposedConfig(
        watermark_size=8,repetition=3,period_candidates=(32.0,40.0,48.0),
        convolution_gaussian_sigmas=(0.5,),nlm=NLMConfig(enabled=False)
    )
    base=np.asarray([50.0,72.0,91.0])
    conv=np.asarray([[52.0,70.0,95.0]])
    a=choose_period_for_group(base,conv,cfg)
    b=choose_period_for_group(base.copy(),conv.copy(),cfg)
    assert a==b
