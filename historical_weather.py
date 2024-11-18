import click
import json
import pandas as pd

CSV_FILE = "data/noaa_historical_weather_10yr.csv"


## Extract
def extract_data(csv_path: str) -> pd.DataFrame:
    """
    Read in the csv data, parse dates, and set index to DATE column. This function assumes
    that the input has a column called 'DATE' than can be parsed into datetime objects.
    
    Args:
      csv_path (str): The path to the provided csv data.
    
    Returns:
      df (pd.DataFrame): Pandas dataframe representation of the csv file with datetime index.
    """
    df = pd.read_csv(csv_path, parse_dates=['DATE'])
    df = df.set_index('DATE').sort_index()
    return df


## Transform
def add_measurable_precip(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add column with a boolean indicator of whether or not there was precipiation or snow on a 
    given day. To give correct results, this should only be run on subsets of data that 
    contain only a single location. In other words, this should be run from within the 
    `augment_data` function rather than directly on the full dataset.
    
    Args:
      df (pd.DataFrame): A single city subset of the pandas dataframe representation of the 
          csv file with datetime index.
    
    Returns:
      df_out (pd.DataFrame): Same dataframe with additional column called `had_precipitation`.
    """
    df_out = df.copy()
    df_out['had_precipitation'] = (df_out.PRCP > 0) | (df_out.SNOW > 0)
    return df_out


def augment_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a boolean column called `had_precipitation` that indicates if snow or rain was 
    greater than zero on a given day (for a particular city). Also add a `city` column that has a 
    3 letter abbreviation derived from `NAME`.
    
    Args:
      df (pd.DataFrame): DataFrame of historical precipitation data as output by the `extract_data` 
          function.
    
    Returns:
      df_out (pd.DataFrame): The input dataset with added `city` and `had_precipitation` columns.
    """
    df_out = df.copy()
    df_out['city'] = df_out.NAME.str[:3].str.upper().replace('JUN', 'JNU')
    grp = df_out.groupby('city')
    df_out = grp.apply(add_measurable_precip,
                       include_groups=False).reset_index(level='city')
    return df_out


def day_of_year_probability(df: pd.DataFrame,
                            roll_period='30D') -> pd.DataFrame:
    """
    Transform the input dataframe into a dataframe of probability of precipitation for each location.
    
    Args:
      df (pd.DataFrame): DataFrame of historical precipitation data as output by the `extract_data` 
          function and augmented by the `augment_data` function.
      roll_period (str): Size of the moving window. If an integer, the fixed number of observations 
          used for each window. If a timedelta, str, or offset, the time period of each window. Each 
          window will be a variable sized based on the observations included in the time-period. This is 
          only valid for datetimelike indexes. To learn more about the offsets & frequency strings, please 
          see this [link](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).
    
    Returns:
      df_out (pd.DataFrame): A pandas dataframe with days of the year as index (1 - 366) and cities
          as columns. The values represent the estimated probability of precipitation for a given
          city on a given day of the year.
    """
    df_out = df.copy()
    # Group by city then calculate the rolling mean of the boolean `had_precipitation` flag.
    # This will smooth the short term fluctuations to make seasonal trends more apparent.
    df_out = df_out.groupby('city').had_precipitation.rolling(
        roll_period, center=True).mean()
    # Unstack the resulting multi index so that each city is in a column
    df_out = df_out.unstack('city')
    # Get the mean (across all years) for each day of the year.
    df_out = df_out.groupby(df_out.index.day_of_year).mean()
    return df_out


def days_of_precip(
    df: pd.DataFrame,
    city: str,
) -> str:
    """
    Calculate the average (i.e., mean) number of days per year the given city had non-zero 
    precipitation (either rain or snow) based on the entire 10 year period.
    
    Args:
      df (pd.DataFrame): DataFrame of historical precipitation data as output by the `extract_data` 
          function and augmented by the `augment_data` function.
      city (str): The name of the city to calculate values for. Expected values are bos, jun, or mia.
    
    Returns:
      str: JSON representation of the name of the `city` (str) and mean `days_of_precip` (float) 
          for that city.
    """
    # Limit our data frame to the city of interest
    city_df = df.query("city == @city")
    # Group precip boolean by year, sum to find total days for each year, then get the mean across years
    mean_days = city_df['had_precipitation'].groupby(
        city_df.index.year).sum().mean()
    # Format the output as json
    json_out = json.dumps(dict(city=city, days_of_precip=mean_days))

    return json_out


def chance_of_precip(
    df: pd.DataFrame,
    city: str,
    month: int,
    day: int = 0,
) -> str:
    """
    Given the provided dataset, predict how likely it is a given city will experience rain or snow
    for a given month or day. Just please don't ask for February 29th.
    
    Args:
      df (pd.DataFrame): DataFrame of historical precipitation data as output by the `extract_data` 
          function and augmented by the `augment_data` function.
      city (str): The name of the city to calculate values for. Expected values are bos, jun, or mia.
      month (int): Month to use for the prediction. Expected values 1 - 12.
      day (int, optional): Day to use for prediction. Expected values 1 - 31. If left out (or 0), 
          then the prediction will be for the whole month.
    
    Returns:
      str: JSON representation of the `city` (str) and `precipitation_likelihood` (float) for that city.
    
    """
    # Convert to day of year probability
    doy_prob = day_of_year_probability(df)

    if day:
        # Just need to look up value for day and city.
        doy = pd.to_datetime(f'{month}-{day}', format='%m-%d').day_of_year
        prob = doy_prob.loc[doy, city]
    else:
        # get start and end day of year from month
        p = pd.Period(pd.to_datetime(month, format='%m'), 'M')
        start = p.to_timestamp(how='start').day_of_year
        end = p.to_timestamp(how='end').day_of_year
        # get series of daily PoP (probability of precipitation)
        daily_pop = doy_prob.loc[slice(start, end), city]
        # To figure out how likely it is to rain at least once during the whole month
        # we will first figure out how likely it is that it won't rain at all.
        prob_no_rain = (1 - daily_pop).prod()
        # Convert to probability that it will rain at least once
        prob = 1 - prob_no_rain
    # Format the output as json
    json_out = json.dumps(dict(city=city, precipitation_likelihood=prob))

    return json_out


## CLI
@click.group()
def cli():
    pass


@click.command('days-of-precip')
@click.argument('city', type=str)
def days(city):
    """
    Mean number of days per year the given city had non-zero precipitation (either
    snow or rain) based on the entire 10 year period.
    """
    df = extract_data(CSV_FILE).pipe(augment_data)
    json_out = days_of_precip(df, city)
    click.echo(json_out)


@click.command('chance-of-precip')
@click.argument('city', type=str)
@click.argument('month', type=int)
@click.argument('day', default=0)
def chance(city, month, day):
    df = extract_data(CSV_FILE).pipe(augment_data)
    json_out = chance_of_precip(df, city, month, day)
    click.echo(json_out)


cli.add_command(days)
cli.add_command(chance)

if __name__ == '__main__':
    cli()
