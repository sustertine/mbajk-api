import os

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from evidently.test_suite import TestSuite

from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.tests import TestNumberOfColumnsWithMissingValues, TestNumberOfRowsWithMissingValues, \
    TestNumberOfConstantColumns, TestNumberOfDuplicatedRows, TestNumberOfDuplicatedColumns, TestColumnsType, \
    TestNumberOfDriftedColumns


def data_drift():
    base_dir = os.getenv('GITHUB_WORKSPACE', '../../')
    report = Report(metrics=[DataDriftPreset()])

    current = pd.read_csv(f'{base_dir}/data/processed/current_data.csv')
    reference = pd.read_csv(f'{base_dir}/data/processed/reference_data.csv')

    report.run(reference_data=reference, current_data=current)
    report.save_html(f'{base_dir}/reports/sites/data_drift.html')


def stability_test():
    base_dir = os.getenv('GITHUB_WORKSPACE', '../../')
    suite = TestSuite(tests=[
        TestNumberOfColumnsWithMissingValues(),
        TestNumberOfRowsWithMissingValues(),
        TestNumberOfConstantColumns(),
        TestNumberOfDuplicatedRows(),
        TestNumberOfDuplicatedColumns(),
        TestColumnsType(),
        TestNumberOfDriftedColumns(),
        NoTargetPerformanceTestPreset(),
        DataStabilityTestPreset()
    ])

    current = pd.read_csv(f'{base_dir}/data/processed/current_data.csv')
    reference = pd.read_csv(f'{base_dir}/data/processed/reference_data.csv')

    suite.run(reference_data=reference, current_data=current)
    suite.save_html(f'{base_dir}/reports/sites/data_stability.html')


if __name__ == '__main__':
    data_drift()
    stability_test()
