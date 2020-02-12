# onhm-fetcher-parser

Fetch and parse climate data for oNHM

Set up environment using Conda

	* conda env create -f pgk/environment.yml
	* conda activate ofp_env

* Add `Data` directory to `root` directory and get data from:
* `ftp://ftpext.usgs.gov/pub/cr/co/denver/BRR-CR/pub/rmcd/Data_hru_shp_v2.tar.gz`
* unzip `Data_hru_shp_v2.tar.gz` into `Data` folder: contains shapefiles of HRU by region

To run onhm-fetcher-parser from a Conda command prompt using the `ofp_env` environment:

	* python pkg/climate_etl.py -t date -p 2015-01-01 2015-12-31 -f 2015_ -i Data -o Output -w Data/weights.csv
