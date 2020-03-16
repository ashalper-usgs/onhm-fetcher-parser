import geopandas as gpd
import logging
import netCDF4
import numpy as np
import pandas as pd
import requests
import sys
import xarray as xr
from datetime import datetime
from helper import np_get_wval, get_gm_url
from pathlib import Path
from requests.exceptions import HTTPError

class FpoNHM:
    """ Class for fetching climate data and parsing into NetCDF
        input files for use with the USGS operational National Hydrologic
        Model (oNHM). Workflow:
            1) Initialize(): fetch climate data
            2) Run(): map/interpolate onto HRU
            3) Finalize(): write NetCDF input files
        Mapping options:
            1) weighted average based on intersection area of HRU
                with NetCDF file cells.
            2) rasterstats - zonal averaging

    """
    def __init__(self, climsource='GridMetSS'):
        """
        Initialize class

        :param  numdays: number of days past to retrieve
        :param climsource: Constant for now but may have multiple
            choice for data sources in the future.  Currently default is
            gridMET: http://www.climatologylab.org/gridmet.html
        """
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.climsource = climsource
        if climsource == 'GridMetSS':
            self.gmss_vars = {
                'tmax': 'daily_maximum_temperature',
                'tmin': 'daily_minimum_temperature',
                'ppt': 'precipitation_amount',
                'rhmax': 'daily_maximum_relative_humidity',
                'rhmin': 'daily_minimum_relative_humidity',
                'ws': 'daily_mean_wind_speed'}
            
        # type of retrieval (days) retrieve by previous number of days
        # - used in operational mode or (date) used to retrieve
        # specific period of time
        self.type = None

        self.numdays = None

        # prefix for file names - default is ''.
        self.fileprefix = None

        # xarray containers for temperature max., temperature min. and
        # precipitation
        self.dstmax = None
        self.dstmin = None
        self.dsppt = None
        self.dsrhmax = None
        self.dsrhmin = None
        self.dsws = None

        # Geopandas dataframe that will hold HRU ID and geometry
        self.gdf = None

        # input and output path directories
        self.iptpath = None
        self.optpath = None

        # weights file
        self.wghts_file = None

        # start and end dates of using type == date
        self.start_date = None
        self.end_date = None

        # handles to NetCDF climate data
        
        # Coordinates
        self.lat_h = None
        self.lon_h = None
        self.time_h = None
        
        # Geotransform
        self.crs_h = None
        
        # Climate data
        self.tmax_h = None
        self.tmin_h = None
        self.tppt_h = None
        self.rhmax_h = None
        self.rhmin_h = None
        self.ws_h = None
        
        # Dimensions
        self.dayshape = None
        self.lonshape = None
        self.latshape = None

        # num HRUs
        self.num_hru = None

        # group by hru_id_nat on weights file
        self.unique_hru_ids = None

        # NumPY arrays to store mapped climate data
        self.np_tmax = None
        self.np_tmin = None
        self.np_ppt = None
        self.np_rhmax = None
        self.np_rhmin = None
        self.np_ws = None

        # logical use_date
        self.use_date = False

        # Starting date based on numdays
        self.str_start = None

    def initialize(self, iptpath, optpath, weights_file, type=None, days=None,
                   start_date=None, end_date=None, fileprefix=''):
        """
        Initialize the fp_ohm class:
            1) initialize geopandas dataframe of concatenated hru_shapefiles
            2) initialize climate data using xarray

        :param iptpath: directory containing HRU shapefiles and weight file,
                        geotiffs if using rasterstats
        :param optpath: directory to save NetCDF input files
        :return: success or failure
        """
        
        self.iptpath = Path(iptpath)
        if not self.iptpath.exists():
            sys.exit(f'input directory "{iptpath}" does not exist')

        self.optpath = Path(optpath)
        if not self.optpath.exists():
            sys.exit(f'output directory "{optpath}" does not exist')

        self.wghts_file = Path(weights_file)
        if not self.wghts_file.exists():
            sys.exit(f'weights file "{weights_file}" not exist') 
            
        self.wghts_id = None
        self.type = type
        self.numdays = days
        self.start_date = start_date
        self.end_date = end_date
        self.fileprefix = fileprefix

        self.logger.info(Path.cwd())
        
        if self.type == 'date':
            self.logger.info(
                f'start_date: {self.start_date} and end_date: {self.end_date}'
            )
        else:
            self.logger.info(f'number of days: {self.numdays}')
            
        # glob.glob produces different results on Windows and
        # Linux. Adding sorted makes result consistent.
        # filenames = sorted(glob.glob('*.shp'))
        
        # use pathlib glob
        filenames = sorted(self.iptpath.glob('*.shp'))
        self.gdf = pd.concat([gpd.read_file(f) for f in filenames],
                             sort=True).pipe(gpd.GeoDataFrame)
        self.gdf.reset_index(drop=True, inplace=True)

        msg = 'in directory ' + str(filenames[0].parent) + ':'
        for f in filenames:
            msg = msg + ' ' + f.name + ','
        self.logger.info(msg[:-1])
            
        self.logger.debug(self.gdf.head())

        self.num_hru = len(self.gdf.index)
        tmaxfile = None
        tminfile = None
        pptfile = None
        rhminfile = None
        rhmaxfile = None
        wsfile = None

        if self.type == 'date':
            self.numdays = ((self.end_date - self.start_date).days + 1)
            
        # NetCDF subsetted data
        try:
            # maximum temperature
            self.str_start, tmxurl, tmxparams = get_gm_url(
                self.type, 'tmax', self.numdays,
                self.start_date, self.end_date
            )
            self.logger.debug(f'GET {tmxurl}')
            tmaxfile = requests.get(tmxurl, params=tmxparams)
            tmaxfile.raise_for_status()
            
            # minimum temperature
            self.str_start, tmnurl, tmnparams = get_gm_url(
                self.type, 'tmin', self.numdays,
                self.start_date, self.end_date
            )
            self.logger.debug(f'GET {tmnurl}')
            tminfile = requests.get(tmnurl, params=tmnparams)
            tminfile.raise_for_status()
            
            # precipitation
            self.str_start, ppturl, pptparams = get_gm_url(
                self.type, 'ppt', self.numdays,
                self.start_date, self.end_date
            )
            self.logger.debug(f'GET {ppturl}')
            pptfile = requests.get(ppturl, params=pptparams)
            pptfile.raise_for_status()

            # maximum relative humidity
            self.str_start, rhmaxurl, rhmaxparams = get_gm_url(
                self.type, 'rhmax', self.numdays,
                self.start_date, self.end_date
            )
            self.logger.debug(f'GET {rhmaxurl}')
            rhmaxfile = requests.get(rhmaxurl, params=rhmaxparams)
            rhmaxfile.raise_for_status()
            
            # minimum relative humidity
            self.str_start, rhminurl, rhminparams = get_gm_url(
                self.type, 'rhmin', self.numdays,
                self.start_date, self.end_date
            )
            self.logger.debug(f'GET {rhminurl}')
            rhminfile = requests.get(rhminurl, params=rhminparams)
            rhminfile.raise_for_status()

            # mean daily wind speed
            self.str_start, wsurl, wsparams = get_gm_url(
                self.type, 'ws', self.numdays,
                self.start_date, self.end_date
            )
            self.logger.debug(f'GET {wsurl}')
            wsfile = requests.get(wsurl, params=wsparams)
            wsfile.raise_for_status()

        except HTTPError as http_err:
            self.logger.error(f'HTTP error occured: {http_err}')
            if self.numdays == 1:
                sys.exit("numdays == 1: gridMET not updated")
            else:
                sys.exit("gridMET not available or a bad request")
        except Exception as err:
            self.logger.error(f'Other error occured: {err}')
        else:
            self.logger.info('gridMET data retrieved')

        # write downloaded data to local NetCDF files and open as xarray
        ncfile = (self.iptpath /
                  (self.fileprefix + 'tmax_' +
                   (datetime.now().strftime('%Y_%m_%d')) + '.nc'),
                  self.iptpath /
                  (self.fileprefix + 'tmin_' +
                   str(datetime.now().strftime('%Y_%m_%d')) + '.nc'),
                  self.iptpath /
                  (self.fileprefix + 'ppt_' +
                   str(datetime.now().strftime('%Y_%m_%d')) + '.nc'),
                  self.iptpath /
                  (self.fileprefix + 'rhmax_' +
                   str(datetime.now().strftime('%Y_%m_%d')) + '.nc'),
                  self.iptpath /
                  (self.fileprefix + 'rhmin_' +
                   str(datetime.now().strftime('%Y_%m_%d')) + '.nc'),
                  self.iptpath /
                  (self.fileprefix + 'ws_' +
                   str(datetime.now().strftime('%Y_%m_%d')) + '.nc'))

        for index, tfile in enumerate(ncfile):
            with open(tfile, 'wb') as fh:
                if index == 0:
                    fh.write(tmaxfile.content)
                elif index == 1:
                    fh.write(tminfile.content)
                elif index == 2:
                    fh.write(pptfile.content)
                elif index == 3:
                    fh.write(rhmaxfile.content)
                elif index == 4:
                    fh.write(rhminfile.content)
                elif index == 5:
                    fh.write(wsfile.content)

            fh.close()
            if index == 0:
                self.dstmax = xr.open_dataset(tfile)
            elif index == 1:
                self.dstmin = xr.open_dataset(tfile)
            elif index == 2:
                self.dsppt = xr.open_dataset(tfile)
            elif index == 3:
                self.dsrhmax = xr.open_dataset(tfile)
            elif index == 4:
                self.dsrhmin = xr.open_dataset(tfile)
            elif index == 5:
                self.dsws = xr.open_dataset(tfile)

        # =========================================================
        # Get handles to shape/Lat/Lon/DataArray
        #
        # All the datahandles including shape, lat, lon should be the
        # same for each NetCDF file. In the future this may change but
        # for now will open and assume these data handles are the same
        # for each of the climate NetCDF files, so grab them from
        # dstmax.
        # =========================================================

        self.lat_h = self.dstmax['lat']
        self.lon_h = self.dstmax['lon']
        self.time_h = self.dstmax['day']

        if self.climsource == 'GridMetSS':
            self.tmax_h = self.dstmax[self.gmss_vars['tmax']]
            self.tmin_h = self.dstmin[self.gmss_vars['tmin']]
            self.tppt_h = self.dsppt[self.gmss_vars['ppt']]
            self.rhmax_h = self.dsrhmax[self.gmss_vars['rhmax']]
            self.rhmin_h = self.dsrhmin[self.gmss_vars['rhmin']]
            self.ws_h = self.dsws[self.gmss_vars['ws']]
        else:
            self.logger.error('climate source data not specified')

        ts = self.tmax_h.sizes
        self.dayshape = ts['day']
        self.lonshape = ts['lon']
        self.latshape = ts['lat']

        self.logger.info(
            f'gridMET returned days = {self.dayshape} and expected number of days {self.numdays}'
        )
        if self.dayshape == self.numdays:
            return True
        else:
            self.logger.error('returned and expected days not equal')
            return False

    def run_weights(self):

        # =========================================================
        #       Read HRU weights
        # =========================================================
        
        # read the weights file
        wght_uofi = pd.read_csv(self.wghts_file)
        
        # get hru_id from the weights file and use as identifier below
        self.wghts_id = wght_uofi.columns[1]

        # group by the weights_id for processing
        self.unique_hru_ids = wght_uofi.groupby(self.wghts_id)

        self.logger.info('finished reading weight file')

        # intialize NumPY arrays to store climate vars
        self.np_tmax = np.zeros((self.numdays, self.num_hru))
        self.np_tmin = np.zeros((self.numdays, self.num_hru))
        self.np_ppt = np.zeros((self.numdays, self.num_hru))
        self.np_rhmax = np.zeros((self.numdays, self.num_hru))
        self.np_rhmin = np.zeros((self.numdays, self.num_hru))
        self.np_ws = np.zeros((self.numdays, self.num_hru))

        for day in np.arange(self.numdays):
            self.logger.info(f'day: {day}')
            tmax = np.zeros(self.num_hru)
            tmin = np.zeros(self.num_hru)
            ppt = np.zeros(self.num_hru)
            rhmax = np.zeros(self.num_hru)
            rhmin = np.zeros(self.num_hru)
            ws = np.zeros(self.num_hru)

            tmax_h_flt = self.tmax_h.values[day, :, :].flatten(order='K')
            tmin_h_flt = self.tmin_h.values[day, :, :].flatten(order='K')
            tppt_h_flt = self.tppt_h.values[day, :, :].flatten(order='K')
            trhmax_h_flt = self.rhmax_h.values[day, :, :].flatten(order='K')
            trhmin_h_flt = self.rhmin_h.values[day, :, :].flatten(order='K')
            tws_h_flt = self.ws_h.values[day, :, :].flatten(order='K')

            for index, row in self.gdf.iterrows():
                weight_id_rows = self.unique_hru_ids.get_group(
                    row[self.wghts_id]
                )
                tmax[index] = np.nan_to_num(
                    np_get_wval(tmax_h_flt, weight_id_rows, index + 1) - 273.5
                )
                tmin[index] = np.nan_to_num(
                    np_get_wval(tmin_h_flt, weight_id_rows, index + 1) - 273.5
                    )
                ppt[index] = np.nan_to_num(
                    np_get_wval(tppt_h_flt, weight_id_rows, index + 1)
                )
                rhmax[index] = np.nan_to_num(
                    np_get_wval(trhmax_h_flt, weight_id_rows, index + 1)
                )
                rhmin[index] = np.nan_to_num(
                    np_get_wval(trhmin_h_flt, weight_id_rows, index + 1)
                )
                ws[index] = np.nan_to_num(
                    np_get_wval(tws_h_flt, weight_id_rows, index + 1)
                )

                if index % 10000 == 0:
                    self.logger.info(
                        f'index: {index}, row: {row[self.wghts_id]}'
                    )

            self.np_tmax[day, :] = tmax
            self.np_tmin[day, :] = tmin
            self.np_ppt[day, :] = ppt
            self.np_rhmax[day, :] = rhmax
            self.np_rhmin[day, :] = rhmin
            self.np_ws[day, :] = ws

        # close xarray datasets
        self.dstmax.close()
        self.dstmin.close()
        self.dsppt.close()
        self.dsrhmax.close()
        self.dsrhmin.close()
        self.dsws.close()

    def run_rasterstat(self):
        tmp = 0

    def finalize(self):
        self.logger.info(Path.cwd())
        ncfile = netCDF4.Dataset(
            self.optpath /
            (self.fileprefix + 'climate_' +
             str(datetime.now().strftime('%Y_%m_%d')) + '.nc'),
            mode='w', format='NETCDF4_CLASSIC'
        )

        # global attributes
        ncfile.Conventions = 'CF-1.8'
        ncfile.featureType = 'timeSeries'
        ncfile.history = ''

        sp_dim = len(self.gdf.index)

        hruid_dim = ncfile.createDimension('hruid', sp_dim) # hru_id

        # unlimited axis (can be appended to)
        time_dim = ncfile.createDimension('time', self.numdays)
        
        for dim in ncfile.dimensions.items():
            self.logger.info(f'dim: {dim}')

        # create variables

        time = ncfile.createVariable('time', 'i', ('time',))
        time.long_name = 'time'
        time.standard_name = 'time'
        time.units = 'days since ' + self.str_start

        hru = ncfile.createVariable('hruid', 'i', ('hruid',))
        hru.cf_role = 'timeseries_id'
        hru.long_name = 'local model hru id'

        lat = ncfile.createVariable(
            'hru_lat', np.dtype(np.float32).char, ('hruid',)
        )
        lat.long_name = 'Latitude of HRU centroid'
        lat.units = 'degrees_north'
        lat.standard_name = 'hru_latitude'

        lon = ncfile.createVariable(
            'hru_lon', np.dtype(np.float32).char, ('hruid',)
        )
        lon.long_name = 'Longitude of HRU centroid'
        lon.units = 'degrees_east'
        lon.standard_name = 'hru_longitude'

        prcp = ncfile.createVariable(
            'prcp', np.dtype(np.float32).char, ('time', 'hruid')
        )
        prcp.long_name = 'Daily precipitation rate'
        prcp.units = 'mm/day'
        prcp.standard_name = 'lwe_precipitation_rate'

        tmax = ncfile.createVariable(
            'tmax', np.dtype(np.float32).char, ('time', 'hruid')
        )
        tmax.long_name = 'Maximum daily air temperature'
        tmax.units = 'degree_Celsius'
        tmax.standard_name = 'maximum_daily_air_temperature'

        tmin = ncfile.createVariable(
            'tmin', np.dtype(np.float32).char, ('time', 'hruid')
        )
        tmin.long_name = 'Minimum daily air temperature'
        tmin.units = 'degree_Celsius'
        tmin.standard_name = 'minimum_daily_air_temperature'

        rhmax = ncfile.createVariable(
            'rhmax', np.dtype(np.float32).char, ('time', 'hruid')
        )
        rhmax.long_name = 'Maximum daily relative humidity'
        rhmax.units = 'percent'
        rhmax.standard_name = 'daily_maximum_relative_humidity'

        rhmin = ncfile.createVariable(
            'rhmin', np.dtype(np.float32).char, ('time', 'hruid')
        )
        rhmin.long_name = 'Minimum daily relative humidity'
        rhmin.units = 'percent'
        rhmin.standard_name = 'daily_minimum_relative_humidity'

        ws = ncfile.createVariable(
            'ws', np.dtype(np.float32).char, ('time', 'hruid')
        )
        ws.long_name = 'Mean daily wind speed'
        ws.units = 'm/s'
        ws.standard_name = 'daily_mean_wind_speed'

        # fill variables with available data
        def getxy(pt):
            return pt.x, pt.y

        centroidseries = self.gdf.geometry.centroid
        tlon, tlat = [list(t) for t in zip(*map(getxy, centroidseries))]

        time[:] = np.arange(0, self.numdays)
        lon[:] = tlon
        lat[:] = tlat
        hru[:] = self.gdf[self.wghts_id].values

        tmax[:, :] = self.np_tmax[:, :]
        tmin[:, :] = self.np_tmin[:, :]
        prcp[:, :] = self.np_ppt[:, :]
        rhmax[:, :] = self.np_rhmax[:, :]
        rhmin[:, :] = self.np_rhmin[:, :]
        ws[:, :] = self.np_ws[:, :]

        ncfile.close()
        self.logger.info("data set is closed")

    def setNumdays(self, num_d):
        self.numdays = num_d
