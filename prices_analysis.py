# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

display(train.sample(7))
display(test.sample(7))

train_num_values = train.select_dtypes(["int64","float64"])
train_disc_values = train.select_dtypes(["object"])
train_disc_values.sample(5)


