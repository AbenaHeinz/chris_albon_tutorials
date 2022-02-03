import pandas as pd
from pytz import all_timezones

# show 10 timezones  
all_timezones[0:10]

# create 10 dates 
dates = pd.Series(pd.date_range("2/2/2002", periods=10, freq="M"))

# set_time_zone
dates_with_abidjan_time_zone = dates.dt.tz_localize("Africa/Abidjan")

#veiw pandas series
dates_with_abidjan_time_zone