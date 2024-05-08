import os
import sys

import pandas as pd
from scipy import stats


def ks_test():
    base_dir = os.getenv('GITHUB_WORKSPACE', '../../')
    current_data = pd.read_csv(f'{base_dir}/data/processed/current_data.csv')
    reference_data = pd.read_csv(f'{base_dir}/data/processed/reference_data.csv')

    all_features_valid = True

    for column in current_data.columns:
        D, p_value = stats.ks_2samp(current_data[column], reference_data[column])

        if p_value < 0.05:
            all_features_valid = False

    return all_features_valid


if __name__ == '__main__':
    if not ks_test():
        sys.exit(1)