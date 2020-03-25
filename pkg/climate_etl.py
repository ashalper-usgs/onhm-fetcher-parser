import argparse
import datetime
import logging
import sys
import time
from fponhm import FpoNHM
from nhm_logging import NHMLogger

logger = NHMLogger.manager.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
ch.setFormatter(formatter)
logger.addHandler(ch)
    
def valid_date(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)

# convenience function to log time spent in a function
def log_time_elapsed(function_name, time_in):
    logger.debug(
        '{:s} ran in {:.{prec}f} seconds'.format(function_name,
                                                 time.time() - time_in, prec=0)
    )
        
def main():
    numdays = None
    startdate = None
    enddate = None
    idir = None
    odir = None
    wght_file = None
    extract_type = None
    file_prefix = None

    my_parser = argparse.ArgumentParser(
        prog='climate_etl',
        description='map gridded climate data to polygon using zonal area weighted mean'
    )

    my_parser.add_argument('-t', '--extract_type', type=str,
                           help='extract method: (days) or (date)',
                           metavar='input_path', default=None,
                           required=True, choices=['days', 'date'])

    my_parser.add_argument('-p', '--period', type=valid_date,
        help='option: start date and end date of retrieval (YYYY-MM-DD)',
                           metavar='date', default=None, nargs=2)

    my_parser.add_argument('-d', '--days', type=int,
                           help='option: number of days to retrieve; ' +
                           'if specified take precedence over -s & -e option',
                           metavar='numdays', default=None)

    my_parser.add_argument('-f', '--file_prefix', type=str,
                           help='option: prefix for output files',
                           metavar='file_prefix', default='')

    my_parser.add_argument('-i', '--inpath',type=str,
                           help='input path (location of shapefiles)',
                           metavar='input_path', default=None,
                           required=True)

    my_parser.add_argument('-o', '--outpath', type=str,
        help='Output path (location of NetCDF by shapefile output)',
                           metavar='output_path',
                           default=None, required=True)

    my_parser.add_argument('-w', '--weightsfile', type=str,
                           help='path/weight.csv - path/name of weight file',
                           metavar='weight_file', default=None,
                           required=True)

    args = my_parser.parse_args()

    if all(i is not None for i in [args.period, args.days]):
        my_parser.error(
            'Either the --days or --period option must be specified not both'
        )

    if all(i is None for i in [args.period, args.days]):
        my_parser.error(
            'Either the --days or --period option must be specified'
        )

    if args.extract_type is not None:
        extract_type = args.extract_type
        if args.extract_type == 'days':
            if args.days is not None:
                numdays = args.days
            else:
                my_parser.error(
                    'if -t --extract_type == days then -d must be specified'
                )
        elif args.extract_type == 'date':
            if args.period is not None:
                startdate = args.period[0]
                enddate = args.period[1]
                if startdate >= enddate:
                    my_parser.error(
                    'when using -p the first date must occur before the second'
                    )
            else:
                my_parser.error(
                    'if -t --extract_type == dates then -p must be specified'
                )

    if args.outpath is not None:
        odir = args.outpath
    if args.inpath is not None:
        idir = args.inpath
    if args.weightsfile is not None:
        wght_file = args.weightsfile
    if args.file_prefix is not None:
        file_prefix = args.file_prefix

    logger.info('starting script...')
    fp = FpoNHM()
    logger.info('...instantiated')

    time_in = time.time()
    ready = fp.initialize(idir, odir, wght_file, type=extract_type,
                          days=numdays, start_date=startdate,
                          end_date=enddate, fileprefix=file_prefix)
    log_time_elapsed('FpoNHM.initialize()', time_in)
    
    if ready:
        logger.info('initialized')
        logger.info('running')
        time_in = time.time()
        fp.run_weights()
        log_time_elapsed('FpoNHM.run_weights()', time_in)
        logger.info('finished running')
        time_in = time.time()
        fp.finalize()
        log_time_elapsed('FpoNHM.finalize()', time_in)
        logger.info('finalized')
    else:
        if extract_type == 'days':
            logger.info('gridMET not updated continue with numdays - 1')
            fp.setNumdays(numdays-1)
            logger.info('initalized')
            logger.info('running...')
            fp.run_weights()
            logger.info('...finished running...')
            fp.finalize()
            logger.info('...finalized')
        else:
            logger.error(
                'extract did not return period specified; gridMET not updated'
            )
            sys.exit(1)

if __name__ == "__main__":
    time_in = time.time()
    main()
    log_time_elapsed('main()', time_in)
