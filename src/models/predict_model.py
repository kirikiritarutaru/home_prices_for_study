import sys
from pathlib import Path
from pprint import pprint

import pandas as pd

from train_model import train_model_pipe

parent_dir = str(Path(__file__).parent.parent.resolve())
sys.path.append(parent_dir)

if __name__ == '__main__':
    from features.build_features import clean_for_reg
    raw_d = Path('../../data/raw')
    repo_d = Path('../../reports')
    repo_fig_d = Path('../../reports/figures')

    df_train = pd.read_csv(raw_d / 'train.csv')
    df_test = pd.read_csv(raw_d / 'test.csv')
    test_ID = df_test['Id']

    df_train, y_train, df_test = clean_for_reg(df_train, df_test)
    ensemble = train_model_pipe(
        df_train,
        y_train,
        df_test
    )

    pprint(ensemble)

    sub = pd.DataFrame()
    sub['Id'] = test_ID
    sub['SalePrice'] = ensemble
    # sub.to_csv(repo_d/'submission.csv', index=False)
