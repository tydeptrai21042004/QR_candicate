import inspect
import numpy as np
from qrwatermark.core.config import ProposedConfig,NLMConfig
from qrwatermark.proposed.branch_selector import choose_mode_for_group


def test_selector_has_no_payload_bit_argument():
    params=inspect.signature(choose_mode_for_group).parameters
    assert 'bit' not in params and 'watermark_bit' not in params


def test_selector_deterministic_for_same_host_group():
    rng=np.random.default_rng(4)
    blocks=[rng.uniform(10,245,size=(2,2)) for _ in range(3)]
    cfg=ProposedConfig(watermark_size=8,repetition=3,nlm=NLMConfig(enabled=False))
    a,_=choose_mode_for_group(blocks,cfg); b,_=choose_mode_for_group(blocks,cfg)
    assert a==b
