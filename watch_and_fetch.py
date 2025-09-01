import datetime
import configparser
import xarray as xr
from pywaf import UmmhverfistofnumApi, CopernicusAPI, Observations

# initialise the API object ...
api = UmmhverfistofnumApi(
    # ... which will store fetched data in the local_storage directory ...
    local_storage="local_storage/"
)

# ...the API object will search for data from stations that ...
result = api.get_data(

    # ... lie within this bounding box ...
    minlat = 63.7,
    maxlat = 64.3,
    minlon = -23.0,
    maxlon = -21.0,

    # ... and were operational at some point between these two dates ...
    start = datetime.datetime(2021,4,7),
    end = datetime.datetime(2021,4,9),

    # ... for this species:
    species = 'SO2'

    # If the data is present in local_storage, that data will be returned,
    # otherwise, it will fetch the data using the Umhverfisstofnum API
)


# save the dataset as an example of DT5402
result.to_netcdf("dt5402.nc")




