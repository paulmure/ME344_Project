import pandas as pd
import numpy as np
import os


def get_filepath(filename, dir):
    return os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data', dir, filename))


# This section deals with the cases data

def make_case_df_from_csv(filename):
    df = pd.read_csv(filename)
    df = df[pd.notna(df['fips'])]

    dtypes = {'date': str, 'county': str, 'state': str, 'fips': np.int32, 'cases': np.int32, 'deaths': np.int32}
    df = df.astype(dtypes)

    return df


def make_date_indeces(dates):
    seen = set()
    seen_add = seen.add
    unique_dates = [x for x in dates if not(x in seen or seen_add(x))]
    return {date: idx for idx, date in enumerate(unique_dates)}


def convert_date_to_index(df):
    dates = df['date'].tolist()
    date_to_index = make_date_indeces(dates)
    df['date_index'] = df.apply(lambda row: date_to_index[row['date']], axis=1)


def change_cols_for_case_df(df):
    cols = ['date_index', 'fips', 'cases', 'deaths']
    return df[cols]


def parse_case_data():
    raw_case_data = make_case_df_from_csv(get_filepath('us-counties.csv', 'raw'))
    convert_date_to_index(raw_case_data)
    return change_cols_for_case_df(raw_case_data)


# This section processes other data like population and hospital capacity

def process_population_data():
    raw_data = pd.read_csv(get_filepath('Average_Household_Size_and_Population_Density_-_County.csv', 'raw'))

    result = {}

    result['house_size'] = make_dict(raw_data, 'GEOID', 'B25010_001E')
    result['pop_dens'] = make_dict(raw_data, 'GEOID', 'B01001_calc_PopDensity')
    result['total_pop'] = make_dict(raw_data, 'GEOID', 'B01001_001E')

    result['unique_fips'] = list(result['house_size'].keys())

    return result


def make_dict(df, key_name, value_name):
    df_for_dict = df[[key_name, value_name]].copy()
    return df_for_dict.set_index(key_name).to_dict()[value_name]


def make_dict_with_index(df, value_name):
    df_for_dict = df[[value_name]].copy()
    return df_for_dict.to_dict()[value_name]


def process_hospital_resouces():
    hospital_resource = pd.read_csv(get_filepath('Definitive_Healthcare%3A_USA_Hospital_Beds.csv', 'raw'))

    hospital_resource = hospital_resource[pd.notna(hospital_resource['BED_UTILIZATION'])]
    hospital_resource = hospital_resource[pd.notna(hospital_resource['FIPS'])]

    hospital_resource = hospital_resource.astype({'FIPS': np.int32})
    hospital_resource = hospital_resource[['FIPS', 'NUM_LICENSED_BEDS',	'NUM_STAFFED_BEDS',	'NUM_ICU_BEDS',	'ADULT_ICU_BEDS', 'Potential_Increase_In_Bed_Capac', 'AVG_VENTILATOR_USAGE']]

    return hospital_resource.groupby('FIPS').sum()


def make_hospital_resouce_dicts(df):
    result = {}

    for col in df.columns:
        result[col] = make_dict_with_index(df, col)

    return result


# This section combines everything

def add_population_data(case_data):
    population_data = process_population_data()
    case_data = case_data[case_data['fips'].isin(population_data['unique_fips'])]

    case_data['house_size'] = case_data.apply(lambda row: population_data['house_size'][row['fips']], axis=1)
    case_data['pop_dens'] = case_data.apply(lambda row: population_data['pop_dens'][row['fips']], axis=1)
    case_data['total_pop'] = case_data.apply(lambda row: population_data['total_pop'][row['fips']], axis=1)

    case_data = case_data[['fips', 'date_index', 'house_size', 'pop_dens', 'total_pop', 'cases', 'deaths']]

    return case_data


def add_hospital_data(case_data):
    hospital_resource = process_hospital_resouces()
    case_data = case_data[case_data['fips'].isin(hospital_resource.index)]

    hospital_dicts = make_hospital_resouce_dicts(hospital_resource)

    for key in hospital_dicts.keys():
        case_data[key] = case_data.apply(lambda row: hospital_dicts[key][row['fips']], axis=1)

    case_data = case_data[['fips', 'date_index', 'house_size', 'pop_dens', 'total_pop', 
                           'NUM_LICENSED_BEDS',	'NUM_STAFFED_BEDS',	'NUM_ICU_BEDS',	
                           'ADULT_ICU_BEDS', 'Potential_Increase_In_Bed_Capac', 'AVG_VENTILATOR_USAGE', 'cases', 'deaths']]

    return case_data


def main():
    cases_with_population = add_population_data(parse_case_data())
    cases_with_population.to_csv(get_filepath('cases_with_population.csv', 'processed'), index=False)

    cases_with_population_and_hospital = add_hospital_data(cases_with_population)
    cases_with_population_and_hospital.to_csv(get_filepath('cases_with_population_and_hospital.csv', 'processed'), index=False)


if __name__ == '__main__':
    main()
