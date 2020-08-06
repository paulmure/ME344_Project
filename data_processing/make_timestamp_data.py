import pandas as pd
from multiprocessing import Pool
import os

WINDOW_SIZE = 4
PRED_DIST = 7


def get_data():
    filename = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data', 'processed', 'cases_with_population_and_hospital.csv'))
    df = pd.read_csv(filename)

    fips = df['fips'].unique()
    df_by_fips = []

    for fip in fips:
        df_by_fips.append(df[df['fips'] == fip].copy())

    return df_by_fips

def parse_by_fips(df):
    if len(df) - WINDOW_SIZE - PRED_DIST >= 0:
        result = []
        for i in range(len(df) - WINDOW_SIZE - PRED_DIST):
            tmp = df.iloc[i:i + WINDOW_SIZE, 11].tolist()
            tmp += df.iloc[0, 2:11].tolist()
            tmp.append(df.iloc[WINDOW_SIZE + i + PRED_DIST - 1].loc['deaths'])
            result.append(tmp)
        return result


def main():
    df_by_fips = get_data()
    with Pool() as pool:
        data = pool.map(parse_by_fips, df_by_fips)
    
    filtered = [sublist for sublist in data if sublist]
    flat = [item for sublist in filtered for item in sublist]

    df = pd.DataFrame(flat)

    filename = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data', 'processed', 'timestamped_cases_with_population_and_hospital.csv'))
    df.to_csv(filename, index=False, header=False)


if __name__ == '__main__':
    main()
