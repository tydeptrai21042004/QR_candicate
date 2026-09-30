from __future__ import annotations
import pandas as pd

def aggregate_records(df:pd.DataFrame)->pd.DataFrame:
    numeric=["embedding_psnr","embedding_ssim","attacked_psnr","attacked_ssim","nc","ber","embed_seconds","extract_seconds","confidence","side_information_bits","side_information_serialized_bits","certified_fraction","certified_ber","uncertified_ber"]
    cols=[c for c in numeric if c in df.columns]; group=["method","watermark","attack","attack_params"]
    agg=df.groupby(group,dropna=False)[cols].agg(["mean","std","count"]).reset_index()
    agg.columns=["_".join([str(x) for x in c if x!=""]).rstrip("_") if isinstance(c,tuple) else c for c in agg.columns]
    return agg
