import pandas as pd


def get_country_code_dic():

    file_path = "/Users/irischeng/Documents/ISO-3166-Countries-with-Regional-Codes/all/all.csv"

    country_df = pd.read_csv(file_path)

    country_df = country_df[['name', 'alpha_2']]

    #now convert it into a dictionary.....

    country_dic = dict([(abbrev, name) for abbrev, name in zip(country_df.alpha_2, country_df.name)])
    country_dic['XK'] = 'Kosovo'    
    return  country_dic

def get_list_of_countries():
    df = pd.read_csv("/Users/irischeng/Documents/GeoTrainr/country_data.csv")
    eu_countries= (pd.unique(df['2']))
    country_dic = get_country_code_dic()
    return [country_dic[country] for country in eu_countries]

def get_eu_dic():
    df = pd.read_csv("/Users/irischeng/Documents/GeoTrainr/country_data.csv")
    eu_countries= (pd.unique(df['2']))
    country_dic = get_country_code_dic()
    return {eu_country:country_dic[eu_country] for eu_country in eu_countries}


# print(get_eu_dic())

