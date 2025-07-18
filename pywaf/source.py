import os
import glob
import json
import time
import urllib
import requests
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


class UmmhverfistofnumApi:
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





class CopernicusAPI:

    def __init__(self, 
                 client_id, client_secret,
                 base_url="https://catalogue.dataspace.copernicus.eu/odata/v1/Products", 
                 access_token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
                 download_dir= "local_storage"
                ):

        self.base_url = base_url
        
        self.download_dir = download_dir

        self.access_token_url = access_token_url
        
        self.access_token =  self.get_access_token( access_token_url, client_id, client_secret)

        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        #print("Access token:",self.access_token)

    
    def get_access_token(self, url, client_id, client_secret):
        response = requests.post(
            url,
            data={"grant_type": "client_credentials"},
            auth=(client_id, client_secret),
        )
        return response.json()["access_token"]
    
    def search(self, min_lat, max_lat, min_lon, max_lon, start_datetime, end_datetime):

        start_date = datetime.datetime.strftime(start_datetime, "%Y-%m-%d")

        end_date = datetime.datetime.strftime(end_datetime, "%Y-%m-%d")

        # Current (counter-clockwise):
        bbox_query_clockwise = (f"OData.CSC.Intersects(area=geography'SRID=4326;"
                      f"POLYGON(({min_lon} {min_lat},"
                      f"{max_lon} {min_lat},"
                      f"{max_lon} {max_lat},"
                      f"{min_lon} {max_lat},"
                      f"{min_lon} {min_lat}))')")
        
        # Try clockwise instead:
        bbox_query_anti = (f"OData.CSC.Intersects(area=geography'SRID=4326;"
                      f"POLYGON(({min_lon} {min_lat},"
                      f"{min_lon} {max_lat},"
                      f"{max_lon} {max_lat},"
                      f"{max_lon} {min_lat},"
                      f"{min_lon} {min_lat}))')")

        # Create the filter string
        filter_params = [
            bbox_query_clockwise,
            "Collection/Name eq 'SENTINEL-5P'",
            f"ContentDate/Start gt {start_date}T00:00:00.000Z",
            f"ContentDate/Start lt {end_date}T23:59:59.999Z",
            "contains(Name,'_L2__SO2__')" , # This filters for SO2 products
            #"contains(Name,'_03_')",         # Filter for collection 03
           # "contains(Name,'_OFFL_')"         # Filter for collection 03


        ]
        
        # Combine all filters with 'and'
        filter_string = " and ".join(filter_params)
        
        # Construct the full URL
        url = f"{self.base_url}?$filter={filter_string}&$top=100" #
        
        # Make the request
        response = requests.get(url)
        json_data = response.json()

        data = json_data.get('value', [])

        files = [d['Name'] for d in data]
        
        # Print the URL for debugging
        print(f"Query URL: {url}")
        print(f"Number of products found: {len(files)}")

        return json_data


    def search_point(self, lat, lon,tart_datetime, end_datetime):

        start_date = datetime.datetime.strftime(start_datetime, "%Y-%m-%d")

        end_date = datetime.datetime.strftime(end_datetime, "%Y-%m-%d")
        
        # Create the filter string
        filter_params = [
            f"OData.CSC.Intersects(area=geography'SRID=4326;POINT({lon} {lat})')",
            "Collection/Name eq 'SENTINEL-5P'",
            f"ContentDate/Start gt {start_date}T00:00:00.000Z",
            f"ContentDate/Start lt {end_date}T23:59:59.999Z",
            "contains(Name,'L2__SO2')",  # This filters for SO2 products
            "contains(Name,'_03_')",         # Filter for collection 03
            "contains(Name,'_OFFL_')"         # Filter for collection 03

        ]
        
        # Combine all filters with 'and'
        filter_string = " and ".join(filter_params)
        
        # Construct the full URL
        url = f"{self.base_url}?$filter={filter_string}&$top=20"
        
        # Make the request
        response = requests.get(url)
        json_data = response.json()

        data = json_data.get('value', [])

        files = [d['Name'] for d in data]
        
        # Print the URL for debugging
        print(f"Query URL: {url}")
        print(f"Number of products found: {len(files)}")

        return json_data


    
    # Function to download a product
    def download_product(self, product_id, product_name, access_token):

        download_url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"

        
        
        headers = {
            "Authorization": f"Bearer {access_token}"
        }
        
        filepath = os.path.join(self.download_dir, f"{product_name}")
        
        # Skip if already downloaded
        if os.path.exists(filepath):
            print(f"Already downloaded: {product_name}")
            return
        
        print(f"Downloading: {product_name}")
        
        # Download with streaming to handle large files
        with requests.get(download_url, headers=headers, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"Progress: {progress:.1f}%", end='\r')
        
        print(f"\nCompleted: {product_name}")

    def search_and_download(self, min_lat, max_lat, min_lon, max_lon, start_date, end_date):

        json_data = self.search(min_lat, max_lat, min_lon, max_lon, start_date, end_date)

        # Main download process
        if json_data.get('value'):
            print("\nStarting downloads...")
            
            # Download each product
            for product in json_data.get('value', []):
                product_id = product['Id']
                product_name = product['Name']
                
                
                print(f"\nProduct: {product_name}")
                print(f"Date: {product['ContentDate']['Start']}")
                print(f"Size: {product.get('ContentLength', 0) / (1024**2):.1f} MB")
                
                try:
                    self.download_product( product_id, product_name, self.access_token)
                    time.sleep(1)  # Be polite to the server
                except Exception as e:
                    print(f"Error downloading {product_name}: {e}")
                    continue
            
            print(f"\nDownloads complete! Files saved to: {self.download_dir}")
        else:
            print("No products found for the specified criteria")
        
        # Optional: Print summary of downloaded files
        files = []
        print("\nDownloaded files:")
        search_string = os.path.join(self.download_dir,"*L2__SO2*.nc")
        
        for file in glob.glob(search_string): #os.listdir(self.download_dir):
            if file.endswith('.nc'):
                #file_path = os.path.join(self.download_dir, file)
                files.append(file)
                file_size = os.path.getsize(file) / (1024**2)
                print(f"  {file} ({file_size:.1f} MB)")

        return files
    
    def search_and_download_latest(self, min_lat, max_lat, min_lon, max_lon, start_date, end_date):
        
        json_data = self.search(min_lat, max_lat, min_lon, max_lon, start_date, end_date)
        
            
        ids = [p['Id']for p in json_data.get('value', [])]
        
        names = [p['Name'] for p in json_data.get('value', [])]
        
        info = np.array([n.replace("___","_").replace("__","_").replace(".","_").split("_") for n in names])
        
        df_all = pd.DataFrame({
            "satellite":info[:,0],
            "type":info[:,1],
            "level":info[:,2],
            "species":info[:,3],
            "start":info[:,4],
            "end":info[:,5],
            "orbit":info[:,6],
            "collection":info[:,7],
            "processor":info[:,8],
            "creation":info[:,9],
            "file type":info[:,10],
            "file name":names,
            "id":ids
        })
        
        df_all['start'] = df_all['start'].apply(lambda r: datetime.datetime.strptime(r, "%Y%m%dT%H%M%S"))
        df_all['end'] = df_all['end'].apply(lambda r: datetime.datetime.strptime(r, "%Y%m%dT%H%M%S"))
        df_all['creation'] = df_all['creation'].apply(lambda r: datetime.datetime.strptime(r, "%Y%m%dT%H%M%S"))
        
        df_latest = pd.concat([
            gp[gp['creation']==gp['creation'].max()] for name, gp in df_all.groupby("start")
        ])
    
        for i, r in df_latest.iterrows():
            self.download_product(r['id'],r['file name'], self.access_token)

        #files = df_latest['file name'].values
    
        #return files
        return df_latest

    def extract_data(self, file, min_lat, max_lat, min_lon, max_lon):

        print(file)

        ds = xr.open_dataset(file, group='PRODUCT')

        ds = ds.drop(['layer','corner'])

        ds_sub=(
                (ds['latitude']>min_lat)&
                (ds['latitude']<max_lat)&
                (ds['longitude']>min_lon)&
                (ds['longitude']<max_lon)
            )

        df=(
                ds
                .where(ds_sub)
                .dropna(dim='scanline', how='all')
                .dropna(dim='ground_pixel', how='all')
                .to_dataframe()
                .reset_index()
            )
            
            
        df = df[df['latitude'].notna()]

        df['file'] = file
            
        return(df)
        
        
        

    def fetch_data(self, min_lat, max_lat, min_lon, max_lon, start_date, end_date):

        #lon  = (min_lon + max_lon)/2

        #lat = (min_lat + max_lat)/2

        files = self.search_and_download(min_lat, max_lat, min_lon, max_lon, start_date, end_date)

        dfs = [self.extract_data(file, min_lat, max_lat, min_lon, max_lon) for file in files]

        df = pd.concat(dfs, axis=0)

        df = df[df['sulfurdioxide_total_vertical_column'].notna()].reset_index()

        return df

    def fetch_latest_data(self, min_lat, max_lat, min_lon, max_lon, start_date, end_date):

        #lon  = (min_lon + max_lon)/2

        #lat = (min_lat + max_lat)/2

        df_latest = self.search_and_download_latest(min_lat, max_lat, min_lon, max_lon, start_date, end_date)

        #files  = [os.path.join(self.download_dir, f) for f in files]
        df_latest['file name'] = df_latest['file name'].apply(lambda file: os.path.join(self.download_dir, file))
        
        #dfs = [self.extract_data(file, min_lat, max_lat, min_lon, max_lon) for file in files]
        dfs = []

        for i, r in df_latest.iterrows():

            df = self.extract_data(r['file name'], min_lat, max_lat, min_lon, max_lon)

            for name, val in r.items():
                df[name] = val

            dfs.append(df)
            

        df = pd.concat(dfs, axis=0)

        df = df[df['sulfurdioxide_total_vertical_column'].notna()].reset_index(drop=True)

        return df

    



class Observations:

    def __init__(self, copernicus_client_id, copernicus_client_secret, local_storage='local_storage'):

        self.copernicus_client_id = copernicus_client_id

        self.copernicus_client_secret = copernicus_client_secret

        self.local_storage = local_storage

        self.uapi = UmmhverfistofnumApi(local_storage=local_storage)

        self.capi = CopernicusAPI(copernicus_client_id, copernicus_client_secret, download_dir=local_storage)

        


    def fetch(self, minlat, maxlat, minlon, maxlon, start, end):

        result = self.uapi.get_data(start=start, end=end, minlat=minlat, maxlat=maxlat, minlon=minlon, maxlon=maxlon,  species = 'SO2')

        # NOTE! Copernicus API automatically adds 23:59:59 to the end date
        end  = end - datetime.timedelta(days=1)
    
        df = self.capi.fetch_latest_data(minlat, maxlat, minlon, maxlon, start, end)
    
        return xr.merge([result, df.to_xarray()])
        
    
    
    def plot_data(self, data, qa_threshold=0.5):
                
        plt.figure("Test Map",figsize=(20,20))
        crs = ccrs.PlateCarree()
        
        ax0 = plt.subplot(311, projection=crs)

        ax0.coastlines(resolution='10m',color='blue')
        
        ax0.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
        
        ax1 = plt.subplot(312)
    
        
        
        axs = [ax1, ax0]
        
        for name in data:
            
            if "#" in name:
                
                local_id, species, type  = name.split('#')
                
                full_name  = data.attrs[local_id]
                
                label = " ".join([local_id, full_name])
                
                if type=='value':
                    
                    data[name].plot(ax=axs[0],label=label)
                    
                    lat = data[name].attrs['lat']
            
                    lon = data[name].attrs['lon']
            
                    axs[1].scatter([lon],[lat],label=label)
    
        df = data[[
            'latitude',
            'longitude',
            'sulfurdioxide_total_vertical_column', 
            'sulfurdioxide_total_vertical_column_precision',
            'delta_time',
            'time',
            'orbit',
            'qa_value'
        ]].to_dataframe()
        
        groupby = df.groupby('orbit')
    
        num = len(groupby)
    
        ylim = [df['latitude'].min(),df['latitude'].max()]
    
        xlim = [df['longitude'].min(),df['longitude'].max()]
    
        
        for i, (date, gp) in enumerate(groupby):
    
            ijk = int("3" + str(num) + str(i+1+6))
        
            ax2 = plt.subplot(ijk, projection=crs)
    
            ax2.coastlines(resolution='10m',color='blue')
        
            ax2.gridlines(draw_labels=False, dms=False, x_inline=False, y_inline=False)
    
        
            dt = gp['delta_time'].mean()
    
            gp = gp[gp['qa_value']>qa_threshold]
    
            scatter = ax2.scatter(
                gp['longitude'],
                gp['latitude'],
                c=gp['sulfurdioxide_total_vertical_column'],
                transform=crs,  # Important for cartopy!
                s=20  # Adjust marker size as needed
            )
            
            # Add title
            ax2.set_title(dt)
            
            # Add horizontal colorbar
            cbar = plt.colorbar(scatter, ax=ax2, orientation='horizontal', 
                                pad=0.1, shrink=0.8)
    
            ax2.set_xlim(xlim)
            ax2.set_ylim(ylim)
    
            ax2.set_xticks([])
            ax2.set_yticks([])
    
        
        
        axs[0].legend(bbox_to_anchor=(1.1, 1.05))
        
        axs[1].legend(bbox_to_anchor=(1.1, 1.05))
    
        # Adjust the colorbar position if needed
        #plt.tight_layout()
        #plt.show()



