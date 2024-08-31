import os
import json
import urllib
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

class RawApi:
    """Python implementation of Umhverfisstonfnum's web API, see 
        https://api.ust.is/aq for details
    """

    def __init__(self):
        pass
        

    def get(self, url):
        
        with urllib.request.urlopen(url) as u:
            
            data = json.load(u)

        return(data)

    
    def getLatest(self):

        return self.get("https://api.ust.is/aq/a/getLatest")
        

    def getCurent(self, local_id='STA-IS0005A'):

        base_url = "https://api.ust.is/aq.php/a/getCurrent/id/"

        return self.get(base_url + local_id)
        
       
    def getDate(self, date="2018-01-01"):
        print("fetching date", date, "from Ust api")
        
        base_url = "https://api.ust.is/aq/a/getDate/date/"

        return self.get(base_url + date)

            
    def getStations(self):

        return self.get("https://api.ust.is/aq/a/getStations")
        

    def getStation(self, local_id='STA-IS0005A'):

        base_url = "https://api.ust.is/aq.php/a/getStation/id/"

        return self.get(base_url + local_id)


class Api:
    """Fetches data from Umhverfisstofnum's web API, facilitating search
    by date, species and latlon bounding box, and returns the data combined 
    with metadata
    """

    def __init__(self, local_storage=""):

        self.local_storage = local_storage
        
        self.rawapi = RawApi()
        
        self.get_stations()

   

    def get_stations(self):
        
        data = self.rawapi.getStations()
        
        #  ... the data is a list of dicts, so easy to put into a dataframe ...
        df  = pd.DataFrame(data)
        
        df['latitude'] = df['latitude'].astype(float)

        df['longitude'] = df['longitude'].astype(float)

        df['activity_begin'] = df['activity_begin'].apply(lambda r: pd.to_datetime(r))

        df['activity_end'] = df['activity_end'].apply(lambda r: pd.to_datetime(r))
        
        df.loc[
                np.isnat(df['activity_end']),
                'activity_end'
                ] = datetime.datetime.now()
    

        
        # ... except the species are a list in a single column - we want this
        # in a format easier to search (one column for each species with 
        # values of True or False for each station. To do this we need
        # a unique list of all the available species
        all_params = []
        
        for params in df['parameters'].values:
            
            params = params[1:-1].split(',')
            
            all_params.extend(params)
        
        # once we have a list of every occurrence of a species
        # we turn it into a set to get a list of unique elements
        all_params = set(all_params)

        self.stations = df
        
        self.parameters = all_params
        

    def json_to_xarray(self, data):
        
        dss = []

        attrs_for_names = {}
        # json data comes by day, indexed by station

        # first we iterate over the stations ...
        for station, station_data in data.items():

            # .. we pop the full name and station id ...
            
            name = station_data.pop('name')
            
            local_id = station_data.pop('local_id')

            # .. saving them for storage in dataray attributes later ..

            attrs_for_names[local_id] = name

            # ... which leaves the data, indexed by parameter
            # (SO2, PM2.5, etc...)
        
            parameters_data = station_data['parameters']

            # we iterate over the different parameters ...
            for param, param_data in parameters_data.items():

                # ... and for each we get the metadata ...
                unit = param_data.pop('unit')
                resolution = param_data.pop('resolution')

                # ... and get the other metadata we need ...
                lat = self.stations[ self.stations['local_id']==local_id]['latitude'].item()
                lon = self.stations[ self.stations['local_id']==local_id]['longitude'].item()

                # ... what remains is the data, which we put in a dataframe ...
                df = pd.DataFrame(param_data).T

                # ... convert data from strings to the correct datatype ...
                df['value'] = df['value'].astype(float)
                df['verification'] =df['verification'].astype(int)
                df['endtime'] = df['endtime'].apply(lambda r: pd.to_datetime(r))

                # ... and fix the column names so they are unique and can be merged with 
                # data from other days ...
                full_name_for_value = "#".join([local_id, param, 'value'])
                full_name_for_verification = "#".join([local_id, param, "verification"])
                
                df[full_name_for_value] = df['value']
                df[full_name_for_verification] = df['verification']

                del(df['value'])
                del(df['verification'])

                # .. we then set the index, so the data will merge correctly ...
                df = df.set_index("endtime")

                # ... convert to an xarray dataset ...
                ds = df.to_xarray()

                # ... add in the attributes ..
                attrs= {
                    'unit': unit, 
                    'resolution':resolution, 
                    'lat':lat, 
                    'lon':lon
                    }
                
                ds[full_name_for_value].attrs = attrs
                ds[full_name_for_verification].attrs = attrs

                
                dss.append(ds)

            ds_all = xr.merge(dss)

            ds_all.attrs = attrs_for_names

            ds_all.attrs['creation date'] = str(datetime.datetime.now())
        
        
        return ds_all

    def get_date(self, date):

        date_as_string = datetime.datetime.strftime(date,"%Y-%m-%d")

        output_filename = date_as_string + ".nc"
        
        output_path = os.path.join(self.local_storage, output_filename)

        if os.path.isfile(output_path):
            print(output_filename,"exists, loading from local storage.")

            data_as_xarray = xr.open_dataset(output_path)

        else:
            print(output_filename,"not found, fetching data using API")

            data_as_json = self.rawapi.getDate(date_as_string)
    
            data_as_xarray = self.json_to_xarray(data_as_json)
    
            data_as_xarray.to_netcdf(output_path)

        return(data_as_xarray)
        
        

    def search_stations(self, minlat, maxlat, minlon, maxlon, start:datetime, end:datetime, species:str):

        # raise an exception if the species isn't recognised ansd inform
        # the user of valid optionas
        if species not in self.parameters:
            
            raise Exception("species must be one of", self.parameters)

        # get a list of selected satations
        selected_stations = self.stations[
        
            (self.stations['latitude']>minlat)&
        
            (self.stations['latitude']<maxlat)&
        
            (self.stations['longitude']>minlon)&
            
            (self.stations['longitude']<maxlon)&
            
            (self.stations['activity_begin']<end)&
                
            (self.stations['activity_end']>start)&
            
            (self.stations['parameters'].apply(lambda r: species in r))            
        ]
        return selected_stations


    def get_data(self,start:datetime, end:datetime,  minlat=None, maxlat=None, minlon=None, maxlon=None, species=None):

        # raise an exception if the species isn't recognised ansd inform
        # the user of valid optionas
        if species not in self.parameters:
            
            raise Exception("species must be one of", self.parameters)

        
        # get the data covering the date interval
        duration = (end-start).days

        all_data = []

        all_attrs = {}
        
        for day in range(duration):
            
            date = start + datetime.timedelta(days=day)
            
            data_as_xarray = self.get_date(date)

            all_data.append(data_as_xarray)

            all_attrs = {**all_attrs, **data_as_xarray.attrs}

        all_data = xr.merge(all_data)

        all_data.attrs = all_attrs

        # now we select only the relevant stations

        # get a list of all the local_ids in our dataset ....
        names = list(all_data)
        num = len(set([name.split('#')[0] for name in names]))
        print(num, "stations between",start, "and", end)

        # filter for species
        if species is not None:
            names  = [ name for name in  names if  species in name.split('#')]
            num = len(set([name.split('#')[0] for name in names]))
            print(num, "stations measuring",species)


        # filter for extent
        if all([x is not None for x in [minlat, maxlat, minlon, maxlon]]):

            
            selected_stations = self.stations[
            
                (self.stations['latitude']>minlat)&
            
                (self.stations['latitude']<maxlat)&
            
                (self.stations['longitude']>minlon)&
                
                (self.stations['longitude']<maxlon)
            ]

            names = [name for name in names if name.split('#')[0] in selected_stations['local_id'].values]
            num = len(set([name.split('#')[0] for name in names]))


            print(num,"within bounding box",minlat, maxlat, minlon,maxlon)

    
        return all_data[names]




    def plot_data(self,data):
                
        plt.figure("Test Map",figsize=(10,10))
        crs = ccrs.PlateCarree()
        
        ax0 = plt.subplot(211, projection=crs)
        #ax0.set_extent(extent, crs=crs)
        ax0.coastlines(resolution='10m',color='blue')
        ax0.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
        
        ax1 = plt.subplot(212)
        
        axs = [ax1, ax0]
        
        for name in data:
            
            local_id, species, type  = name.split('#')
            
            full_name  = data.attrs[local_id]
            
            label = " ".join([local_id, full_name])
            
            if type=='value':
                
                data[name].plot(ax=axs[0],label=label)
                
                lat = data[name].attrs['lat']
        
                lon = data[name].attrs['lon']
        
                axs[1].scatter([lon],[lat],label=label)
        
        axs[0].legend(bbox_to_anchor=(1.1, 1.05))
        axs[1].legend(bbox_to_anchor=(1.1, 1.05))



