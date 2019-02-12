from fire import Fire

import pandas as pd

def main(results_fp):
    df_results = pd.read_csv(results_fp, names=['classifier', 'f1_score'])
    df_means = df_results.groupby(['classifier']).mean()
    df_means['f1_sd'] = df_results.groupby(['classifier']).std()
    df_means.to_csv(results_fp)

if __name__ == '__main__':
    Fire(main)