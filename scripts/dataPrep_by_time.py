import argparse
import multiprocessing as mp
import os
import re

import h5py
import netCDF4
import numpy as np
from functools import partial
import pickle
#from tqdm import tqdm

# logging.basicConfig(
#     filename="dataProcessing.log", encoding="utf-8", level=logging.DEBUG
# )


# NAME: get_variables
# PURPOSE: to get the variables that are spatially and temporally varying
# We want only those that are spatially and temporally varying because we
# want the variables that change throughout the whole grid and also
# change over the course of a year.
# PARAMETERS: data which is the netCDF4 file
# RETURNS: the list of variables of size (num_days_in_month, 60, 100, 100)
def get_variables(data, var_list=['timeDaily_avg_layerThickness',
 'timeDaily_avg_velocityZonal',
 'timeDaily_avg_velocityMeridional',
 'timeDaily_avg_activeTracers_temperature',
 'timeDaily_avg_activeTracers_salinity']):
    list_of_variables = []
    for v in var_list: # only the last 5 variables are the ones we care about
        var_shape = data.variables[v][:].shape
        #print(var_shape)
        list_of_variables.append(data.variables[v][:])
    return np.stack(list_of_variables, axis=-1)


# NAME: get_params
# PURPOSE: get the physical param values for a particular forward because we need to
# include it in our dataset as it is the input for training.
# PARAMETERS: forward which is the number of the forward we care about
# RETURNS: the gm value for that particular forward
# In latin hyper cude we sample for 4 parameters: GM, Redi, cvmix, implicit_bottom_drag
def extract_num(line):
    pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    v = re.findall(pattern, line)
    return v[0]


def get_params(forward, path):
    os.chdir(f"{path}")
    namelist = open("namelist.ocean")
    namelist_lines = namelist.readlines()
    gm = float(extract_num(namelist_lines[81]))
    redi = float(extract_num(namelist_lines[57]))
    cvmix_b_diff = float(extract_num(namelist_lines[109]))
    imp_bot_drag = float(extract_num(namelist_lines[275]))
    namelist.close()
    # os.chdir("..")
    return [gm, redi, cvmix_b_diff, imp_bot_drag]


# NAME: populate_empty_array
# PURPOSE: to populate the array with the values for all of the different
# variables that we care about, i.e. the ones that we found from
# calling get_variables
# PARAMETERS: list_of_variables which is the result of calling
# get_variables, data which is the netCDF4 file for a forward, and gm
# which is the result of calling get_GM
# RETURNS: a numpy array that is of size (30, 60, 100, 100, 17) that
# contains the data for each of the different variables that we care about
def populate_empty_array(list_of_variables, data, params, time_res=1):
    all_the_data = []
    FULL_LEN = 30
    TIME_STEPS = FULL_LEN // time_res
    time_idx = np.arange(0, FULL_LEN, time_res)
    for v in list_of_variables:
        init = np.array(data.variables[v][time_idx])
        reshaped_init = np.reshape(init, (TIME_STEPS, 60, 100, 100, 1))
        all_the_data.append(reshaped_init)
    for p in params:
        all_the_data.append(np.full((TIME_STEPS, 60, 100, 100, 1), p))
    return np.concatenate(all_the_data, axis=4)


# NAME: find_max_depth_index
# PURPOSE: to figure out the index where the maximum depth of a particular
# spatial point is exceeded. This is necessary because past a certain
# depth, some cells return NaNs.
# PARAMETERS: x and y which is the location of the point in question, data
# which is the netCDF4 data for a forward
# RETURNS: the index of the last vertical layer where data should exist
# based on the layer thicknesses and maximum depths
def find_max_depth_index(x, y, data):
    thickness = list(data.variables["refBottomDepth"][:])
    bottom = list(data.variables["bottomDepth"][:])[x][y]
    for t in range(len(thickness)):
        if bottom < thickness[t]:
            return t - 1
    return len(thickness) - 1


# go through all the simulations
def process_single_year(forward, args):
    results_year = []
    for year in range(7, 23):
        dir_path = os.path.join(
            args.SAVE_PATH, f"forwards/forward_{forward}/"
        )
        # dir_path = os.path.join(args.path, f"output_{forward}")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        os.chdir(dir_path)
        scrip_src = f"{args.PROJECT_DIR}/gm/output_0/scrip.nc"
        grd_src = f"{args.PROJECT_DIR}/gm/output_0/grd.nc"
        if os.path.isfile(scrip_src) and os.path.getsize(scrip_src) > 0:
            os.system(f"cp {scrip_src} scrip.nc")
        else:
            print(f"Warning: {scrip_src} does not exist or is empty.")
        if os.path.isfile(grd_src) and os.path.getsize(grd_src) > 0:
            os.system(f"cp {grd_src} grd.nc")
        else:
            print(f"Warning: {grd_src} does not exist or is empty.")
        os.system(
            f"cp {args.path}/forward_{forward}/analysis_members/mpaso.hist.am.timeSeriesStatsDaily.00{year:02}-01-01.nc mpaso.hist.am.timeSeriesStatsDaily.00{year:02}-01-01.nc"
        )

        # need to regrid each of the .nc files for a forward based on the
        # scrip.nc and grd.nc files
        # try:
        #print(f" ==> running ncremap for forward {forward} year {year}")
        output_fname = f"output-00{year:02}-01-01-rgr.nc"
        if not os.path.exists(output_fname):
            os.system(
                f"ncremap -P mpas -s scrip.nc -g grd.nc mpaso.hist.am.timeSeriesStatsDaily.00{year:02}-01-01.nc {output_fname} > /dev/null 2>&1"
            )
        else:
            #print(f"Output file {output_fname} already exists, skipping regrid.")
            pass
        # os.system(
        #     f"ncremap -P mpas -s scrip.nc -g grd.nc output.0003-{month:02}-01_00.00.00.nc output-0003-{month:02}-01-rgr.nc"
        # )

        data = netCDF4.Dataset(f"{dir_path}/{output_fname}")
        stacked_variables = get_variables(data)

        params = get_params(forward, f"{args.path}/forward_{forward}")[
            args.var_id
        ]  # specifiy which parameter to include in the input
        if not isinstance(params, list):
            params = [params]
        params = np.ones_like(stacked_variables[..., 0:len(params)]) * params
        results = np.concatenate([stacked_variables, params], axis=-1)
        results_year.append(results)
        #os.system(f"rm {dir_path}/{output_fname}")
    return np.concatenate(results_year, axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type=str) # /global/cfs/projectdirs/m4259/snarayan/soma_gm_long/ocean/soma/32km/ensemble_long
    parser.add_argument("-var_id", type=int, default=0)
    parser.add_argument("-time_res", type=int, default=15)  # time resolution by days
    parser.add_argument("-PAR_NAME", type=str, default="gm-year7-22-biweekly")
    parser.add_argument(
        "-append", 
        action="store_true", 
        help="Continue processing if some temporary files failed to process. This will add missing temp files to the dataset specified by -SAVE_PATH.")
    parser.add_argument(
        "-PROJECT_DIR",
        type=str,
        default="/global/cfs/projectdirs/m4259/ecucuzzella/soma_ppe_data/",
    )
    parser.add_argument(
        "-SAVE_PATH",
        type=str,
        default="/global/cfs/projectdirs/m4259/abgulhan/ml_converted",
    )
    parser.add_argument("--no_parallel", action="store_true", help="Disable parallel processing for debugging")
    args = parser.parse_args()

    nprocs = mp.cpu_count()
    print(f"Number of processors: {nprocs}")

    # open a hdf5 file to write the results to
    existing_forwards = []
    data_file = f"{args.SAVE_PATH}/data-{args.PAR_NAME}.hdf5"
    if args.append and os.path.exists(data_file):
        print("Continuing processing with existing dataset.")
        f = h5py.File(data_file, "a")
        print("Datasets in the HDF5 file:")
        for name in f.keys():
            print(name)
            existing_forwards.append(int(name.split("_")[-1]))
    else:
        f = h5py.File(data_file, "w")
    DIR = args.path
    print(f"Data from {args.path} are being processed!!!!")
    PROJECT_DIR = "/global/cfs/projectdirs/m4259/ecucuzzella/soma_ppe_data/"


    start_idx, end_idx = 1, 100  #101 forward 100 does not contain proper files
    def process_and_save(forward, args):
        tmp_save = f"{args.SAVE_PATH}/tmp/forward_{forward}.npy"
        if os.path.exists(tmp_save) and os.path.getsize(tmp_save) > 0:
            print(f"Temporary file {tmp_save} already exists. Skipping processing for forward {forward}.")
            return (forward, tmp_save)
        print(f"Processing forward {forward} out of 100")
        result = process_single_year(forward, args)
        # assert result is a numpy array
        assert isinstance(result, np.ndarray), f"Result should be a numpy array. Got: {type(result)}, forward: {forward}, result: {result}"
        #np.save(tmp_save, result)
        with open(tmp_save, "wb") as f_out:
            pickle.dump(result, f_out)
        print(f"==> Forward {forward} saved to file {tmp_save}.")
        return (forward, tmp_save)

    results = []
    if args.no_parallel:
        print("Running in sequential (no parallel) mode for debugging.")
        for forward in range(start_idx, end_idx):
            results.append(process_and_save(forward, args))
    else:
        with mp.Pool(processes=nprocs) as pool:
            results = pool.map(partial(process_and_save, args=args), range(start_idx, end_idx))
    
    for (forward, tmp_save) in results:
        # Load the result from the temporary file and save it to the HDF5 dataset
        #result = np.load(tmp_save)
        if int(forward) in existing_forwards:
            print(f"Forward {forward} already exists in the dataset. Skipping.")
            continue
        with open(tmp_save, "rb") as f_in:
            print(f"Loading forward {forward} from temporary file {tmp_save}.")
            try:
                result = pickle.load(f_in)
            except Exception as e:
                print(f"Error loading forward {forward} from {tmp_save}: {e}")
                os.remove(tmp_save)
                continue
        #os.remove(tmp_save)  # Remove the temporary file after loading
        print(f"Saving forward {forward} to HDF5 file.")
        dset = f.create_dataset(
            f"forward_{forward}", result.shape, dtype="f", data=result
        )
        


