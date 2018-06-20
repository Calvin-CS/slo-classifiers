import pandas as pd
import fire


def main(input_filepath, output_filepath, label):
    df_results = pd.read_csv(input_filepath, encoding="utf-8-sig")

    df_results_cp = df_results.copy().append([pd.Series(df_results.mean())], ignore_index=True)
    df_results_cp = df_results_cp.append([pd.Series(df_results.median())], ignore_index=True)
    df_results_cp = df_results_cp.append([pd.Series(df_results.std())], ignore_index=True)

    trial_series = pd.Series(df_results.index.values)
    trial_series = trial_series.copy().append(pd.Series(['Mean', 'Median', 'Std. Dev.']), ignore_index=True)

    df_results_cp.insert(0, label, trial_series)

    df_results_cp.to_csv(output_filepath, index=False)

    print(df_results_cp)


if __name__ == '__main__':
    fire.Fire(main)