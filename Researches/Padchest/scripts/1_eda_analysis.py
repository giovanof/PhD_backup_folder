"""
Filename: 1_eda_analysis.py
Description:
    PadChest Dataset Exploratory Data Analysis (EDA)
Author: Giovanni Officioso
"""
# ==============================================================================
# 1. EDA ANALYSIS
# ==============================================================================

from pathlib import Path

import pandas as pd

from src.padchest_class import PadChestEDA

eda = PadChestEDA(
    data_path=Path(
        "data/csv/chest_x_ray_images_labels_sample.csv", output_path="eda_results"
    ),
    output_path=Path("eda_results"),
)

dataset = pd.read_csv("data/csv/chest_x_ray_images_labels_sample.csv")
print(dataset.head())
