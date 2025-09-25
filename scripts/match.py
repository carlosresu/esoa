#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, pandas as pd
from .match_features import build_features
from .match_scoring import score_and_classify
from .match_outputs import write_outputs

def match(pnf_prepared_csv: str, esoa_prepared_csv: str, out_csv: str = "esoa_matched.csv") -> str:
    pnf_df = pd.read_csv(pnf_prepared_csv)
    esoa_df = pd.read_csv(esoa_prepared_csv)
    features_df, _pnf_name_set, _who_name_set, _fda_gen_set = build_features(pnf_df, esoa_df)
    out_df = score_and_classify(features_df, pnf_df)
    out_path = os.path.abspath(out_csv)
    write_outputs(out_df, out_path)
    return out_path
