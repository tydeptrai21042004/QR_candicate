from __future__ import annotations

from ...core.config import BaselineConfig
from ..su2016_hessenberg.method import Su2016Hessenberg


class Su2017Hessenberg(Su2016Hessenberg):
    """Deprecated compatibility alias.

    The old repository incorrectly attached a 2017 label to the Hessenberg
    baseline and embedded into H.  The audited paper baseline is Su (2016),
    which embeds in q22/q32 of the orthogonal Q.  Keeping this class prevents
    old scripts from crashing while routing them to the corrected equations.
    New experiments should use ``su2016_hessenberg`` explicitly.
    """

    name = "su2017_hessenberg"

    def __init__(self, config: BaselineConfig | None = None):
        super().__init__(config or BaselineConfig(block_size=4, threshold=0.042))
