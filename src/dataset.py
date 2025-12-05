
import functools

import h5py

import numpy as np

import random

import torch
from torch.utils.data import Dataset

import numpy as np
import os, json
import logging
from typing import List,Tuple, Optional

from src.data_transformations import Transforms


class KolmogorovDataset_Rozet(Dataset):
    def __init__(self, name: str, folderPath: str, mode: str, resolution: str, sequenceLength:List[Tuple[int, int]]=[],
                 framesPerTimeStep: int = 1, limit_trajectories: Optional[int] = None, usegrid: bool = False, conditioned: bool = False) -> None:
        super().__init__()
        self.name = name
        self.folderPath = folderPath
        self.mode = mode
        self.resolution = resolution

        if self.resolution < 64:
            self.resolution_downfactor = 64//self.resolution
        else:
            self.resolution_downfactor = None

        self.sequenceLength = sequenceLength
        self.limit_trajectories = limit_trajectories
        self.usegrid = usegrid
        self.conditioned = conditioned
        
        self.seqLength = sequenceLength[0]
        self.time_step = sequenceLength[1]

        self.time_gaps = np.linspace(0, self.time_step, framesPerTimeStep, dtype = int, endpoint=False)

        file_path = os.path.join(folderPath, self.mode + ".h5")
        self.data = torch.Tensor(np.array(h5py.File(file_path, mode='r')["x"]))

        self.n_trajectories = self.data.shape[0]
        print("aaaa", self.n_trajectories)
        if self.limit_trajectories is not None:
            self.n_trajectories = min(self.n_trajectories, self.limit_trajectories)
        self.n_frames = self.data.shape[1] - self.seqLength + 1  # Ignore timestep for now
          

    def __len__(self) -> int:
        return self.n_trajectories * self.n_frames

    def __getitem__(self, idx:int) -> dict:
        idx_sim = idx // self.n_frames
        idx_frame = idx % self.n_frames
        data_idx = self.data[idx_sim][idx_frame:idx_frame+self.seqLength]
        #data_idx_lf, data_idx_hf = self.separateFrequencies(data_idx, cutoff_frequency=8)
        if self.resolution_downfactor is not None:
            data_idx = data_idx[..., ::self.resolution_downfactor, ::self.resolution_downfactor]
        return {"data" : data_idx, "simParameters": {}} # data_idx_lf + data_idx_hf/4 # data_idx_lf*4 + data_idx_hf



# OBTAINED FROM 
#
# @inproceedings{shehata2025improved,
#   title={Improved Sampling Of Diffusion Models In Fluid Dynamics With Tweedie's Formula},
#   author={Shehata, Youssef and Holzschuh, Benjamin and Thuerey, Nils},
#   booktitle={The Thirteenth International Conference on Learning Representations},
#   year={2025}
# }


class TurbulenceDataset(Dataset):
    """Data set for turbulence and wavelet noise data

    Args:
        name: name of the dataset
        dataDirs: list of paths to data directories
        filterTop: filter for top level folder names (e.g. different types of data)
        excludeFilterTop: mode for filterTop (exclude or include)
        filterSim: filter simulations by min and max (min inclusive, max exclusive)
        excludefilterSim: mode for filterSim (exclude or include)
        filterFrame: mandatory filter for simulation frames by min and max (min inclusive, max exclusive)
        sequenceLength: number of frames to group into a sequence and number of frames to omit in between
        randSeqOffset: randomizes the starting frame of each sequence
        simFields: list of simulation fields to include (vel is always included) ["dens", "pres"]
        simParams: list of simulation parameters to include ["rey", "mach"]
        printLevel: print mode for contents of the dataset ["none", "top", "sim", "full"]
        logLevel: log mode for contents of the dataset ["none", "top", "sim", "full"]
    """
    transform: Transforms
    name:str
    dataDirs:List[str]
    filterTop:List[str]
    excludeFilterTop:bool
    filterSim:List[Tuple[int, int]]
    excludefilterSim:bool
    filterFrame:List[Tuple[int, int]]
    sequenceLength:List[Tuple[int, int]]
    randSeqOffset:bool
    simFields:List[str]
    simParams:List[str]
    printLevel:str="none"
    logLevel:str="sim"

    def __init__(self, name:str, dataDirs:List[str], filterTop:List[str], excludeFilterTop:bool=False, filterSim:List[Tuple[int, int]]=[],
                excludefilterSim:bool=False, filterFrame:List[Tuple[int, int]]=[], sequenceLength:List[Tuple[int, int]]=[],
                randSeqOffset:bool=False, simFields:List[str]=[], simParams:List[str]=[], printLevel:str="none", logLevel:str="sim"):

        assert (len(filterSim) in [0,1,len(filterTop)]), "Sim filter is not set up correctly. Use len=0 for all; len=1 for the same everywhere, len=len(filterTop) to adjust for each top filter"
        assert (len(filterFrame) in [1,len(filterTop)]), "Frame filter is not set up correctly. Use len=1 for the same everywhere, len=len(filterTop) to adjust for each top filter"
        assert (len(sequenceLength) == len(filterFrame)), "Sequence length is not set up correctly, it should match the frame filter."
        if excludeFilterTop:
            assert (len(filterSim) <= 1), "Excluded top filter and adjust sim filtering is not supported!"
            assert (len(filterFrame) <= 1), "Excluded top filter and adjust frame filtering is not supported!"
        assert (printLevel in ["none", "top", "sim", "full"]), "Invalid print level!"

        self.transform = None
        self.name = name
        self.dataDirs = dataDirs
        self.filterTop = filterTop
        self.excludeFilterTop = excludeFilterTop
        self.filterSim = filterSim
        self.excludefilterSim = excludefilterSim
        self.filterFrame = filterFrame
        self.sequenceLength = sequenceLength
        self.randSeqOffset = randSeqOffset
        self.simFields = ["velocity"]
        if "velZ" in simFields:
            self.simFields += ["velocityZ"]
        if "dens" in simFields:
            self.simFields += ["density"]
        if "pres" in simFields:
            self.simFields += ["pressure"]

        self.simParams = simParams
        self.printLevel = printLevel
        self.logLevel = logLevel

        self.summaryPrint = []
        self.summaryLog = []
        self.summaryPrint += ["Dataset " + name + " at " + str(dataDirs)]
        self.summaryLog   += ["Dataset " + name + " at " + str(dataDirs)]
        self.summaryPrint += [self.getFilterInfoString()]
        self.summaryLog   += [self.getFilterInfoString()]

        # BUILD FULL FILE LIST
        self.dataPaths = []
        self.dataPathModes = []

        for dataDir in dataDirs:
            topDirs = os.listdir(dataDir)
            topDirs.sort()

            # top level folders
            for topDir in topDirs:
                if filterTop:
                    # continue when excluding or including according to filter
                    if excludeFilterTop == any( item in topDir for item in filterTop ):
                        continue

                match = -1
                # compute matching top filter for according sim or frame filtering
                if len(filterSim) > 1 or len(filterFrame) > 1:
                    for i in range(len(filterTop)):
                        if filterTop[i] in topDir:
                            match = i
                            break
                    assert (match >= 0), "Match computation error"

                simDir = os.path.join(dataDir, topDir)
                sims = os.listdir(simDir)
                sims.sort()

                if printLevel == "top":
                    self.summaryPrint += ["Top folder loaded: " + simDir.replace(dataDir + "/", "")]
                if logLevel == "top":
                    self.summaryLog   += ["Top folder loaded: " + simDir.replace(dataDir + "/", "")]

                # sim_000001 folders
                for sim in sims:
                    currentDir = os.path.join(simDir, sim)
                    if not os.path.isdir(currentDir):
                        continue

                    if len(filterSim) > 0:
                        simNum = int(sim.split("_")[1])
                        if len(filterSim) == 1:
                            if type(filterSim[0]) is tuple:
                                inside = simNum >= filterSim[0][0] and simNum < filterSim[0][1]
                            elif type(filterSim[0]) is list:
                                inside = simNum in filterSim[0]
                        else:
                            if type(filterSim[match]) is tuple:
                                inside = simNum >= filterSim[match][0] and simNum < filterSim[match][1]
                            elif type(filterSim[match]) is list:
                                inside = simNum in filterSim[match]
                        # continue when excluding or including according to filter
                        if inside == excludefilterSim:
                            continue

                    if printLevel == "sim":
                        self.summaryPrint += ["Sim loaded: " + currentDir.replace(dataDir + "/", "")]
                    if logLevel == "sim":
                        self.summaryLog   += ["Sim loaded: " + currentDir.replace(dataDir + "/", "")]

                    # individual simulation frames
                    minFrame = filterFrame[0][0] if len(filterFrame) == 1 else filterFrame[match][0]
                    maxFrame = filterFrame[0][1] if len(filterFrame) == 1 else filterFrame[match][1]
                    print(sequenceLength)
                    seqLength = sequenceLength[0][0] if len(sequenceLength) == 1 else sequenceLength[match][0]
                    seqSkip   = sequenceLength[0][1] if len(sequenceLength) == 1 else sequenceLength[match][1]
                    for seqStart in range(minFrame, maxFrame, seqLength*seqSkip):
                        validSeq = True
                        for frame in range(seqStart, seqStart+seqLength*seqSkip, seqSkip):
                            # discard incomplete sequences at simulation end
                            if seqStart+seqLength*seqSkip > maxFrame:
                                validSeq = False
                                break

                            for field in self.simFields:
                                currentField = os.path.join(currentDir, "%s_%06d.npz" % (field, frame))
                                if not os.path.isfile(currentField):
                                    raise FileNotFoundError("Could not load %s file: %s" % (field, currentField))

                        # imcomplete sequence means there are no more frames left
                        if not validSeq:
                            break

                        if printLevel == "full":
                            self.summaryPrint += ["Frames %s loaded: %s/%s_%06d-%06d(%03d).npz" % ("-".join(self.simFields),
                                        currentDir.replace(dataDir + "/", ""), "-".join(self.simFields), seqStart, seqStart + seqLength*(seqSkip-1), seqSkip)]
                        if logLevel == "full":
                            self.summaryLog   += ["Frames %s loaded: %s/%s_%06d-%06d(%03d).npz" % ("-".join(self.simFields),
                                        currentDir.replace(dataDir + "/", ""), "-".join(self.simFields), seqStart, seqStart + seqLength*(seqSkip-1), seqSkip)]

                        self.dataPaths.append((currentDir, seqStart, seqStart + seqLength*seqSkip, seqSkip))

        self.summaryPrint += ["Dataset Length: %d\n" % len(self.dataPaths)]
        self.summaryLog   += ["Dataset Length: %d\n" % len(self.dataPaths)]


    def __len__(self) -> int:
        return len(self.dataPaths)


    def __getitem__(self, idx:int) -> dict:
        # sequence indexing
        basePath, seqStart, seqEnd, seqSkip = self.dataPaths[idx]
        seqLen = int((seqEnd - seqStart) / seqSkip)
        if self.randSeqOffset:
            halfSeq = int((seqEnd-seqStart) / 2)
            offset = torch.randint(-halfSeq, halfSeq+1, (1,)).item()
            if seqStart + offset >= self.filterFrame[0][0] and seqEnd + offset < self.filterFrame[0][1]:
                seqStart = seqStart + offset
                seqEnd = seqEnd + offset

        # loading simulation parameters
        with open(os.path.join(basePath, "src", "description.json")) as f:
            loadedJSON = json.load(f)

            loadNames = ["Reynolds Number", "Mach Number", "Drag Coefficient", "Lift Coefficient", "Z Slice"]
            loadedParams = {}
            for loadName in loadNames:
                loadedParam = np.zeros(seqLen, dtype=np.float32)
                if loadName in loadedJSON:
                    temp = loadedJSON[loadName]
                    if isinstance(temp, int) or isinstance(temp, float):
                        temp = np.array(temp, dtype=np.float32)
                        loadedParam[0:] = np.repeat(temp, seqLen)
                    elif isinstance(temp, list):
                        loadedParam[0:] = temp[seqStart:seqEnd:seqSkip]
                    else:
                        raise ValueError("Invalid simulation parameter data type")
                loadedParams[loadName] = loadedParam

            if "rey" in self.simParams and "mach" in self.simParams:
                simParameters = np.stack([loadedParams["Reynolds Number"], loadedParams["Mach Number"]], axis=1)
            elif "rey" in self.simParams:
                simParameters = np.reshape(loadedParams["Reynolds Number"], (-1,1))
            elif "mach" in self.simParams:
                simParameters = np.reshape(loadedParams["Mach Number"], (-1,1))
            elif "zslice" in self.simParams:
                simParameters = np.reshape(loadedParams["Z Slice"], (-1,1))
            elif not self.simParams:
                simParameters ={}
            else:
                raise ValueError("Invalid specification of simulation parameters")

        # loading obstacle mask
        if os.path.isfile(os.path.join(basePath, "obstacle_mask.npz")):
            obsMask = np.load(os.path.join(basePath, "obstacle_mask.npz"))['arr_0']
        else:
            obsMask = None

        # loading fields and combining them with simulation parameters
        loaded = {}
        for field in self.simFields:
            loaded[field] = []

        for frame in range(seqStart, seqEnd, seqSkip):
            for field in self.simFields:
                loadedArr = np.load(os.path.join(basePath, "%s_%06d.npz" % (field,frame)))['arr_0']
                loaded[field] += [loadedArr.astype(np.float32)]

        loadedFields = []
        for field in self.simFields:
            loadedFields += [np.stack(loaded[field], axis=0)]

        if type(simParameters) is not dict:
            vel = loadedFields[0]
            if vel.ndim == 4:
                simParExpanded = simParameters[:,:,np.newaxis,np.newaxis]
                simParExpanded = np.repeat(np.repeat(simParExpanded, vel.shape[2], axis=2), vel.shape[3], axis=3)
            elif vel.ndim == 5:
                simParExpanded = simParameters[:,:,np.newaxis,np.newaxis,np.newaxis]
                simParExpanded = np.repeat(np.repeat(np.repeat(simParExpanded, vel.shape[2], axis=2), vel.shape[3], axis=3), vel.shape[4], axis=4)
            else:
                raise ValueError("Invalid input shape when loading samples!")
            loadedFields += [simParExpanded]

        data = np.concatenate(loadedFields, axis=1) # ORDER (fields): velocity (x,y), velocity z / density, pressure, ORDER (params): rey, mach, zslice

        # output
        dataPath = "%s/%s_%06d-%06d(%03d).npz" % (basePath, "-".join(self.simFields), seqStart, seqEnd - seqSkip, seqSkip)
        sample = {"data" : data, "simParameters" : simParameters, "allParameters" : loadedParams, "path" : dataPath}
        if obsMask is not None:
            sample["obsMask"] = obsMask

        if self.transform:
            sample = self.transform(sample)
        else:
            print("WARNING: no data transformations are employed!")

        return sample


    def printDatasetInfo(self):
        if self.transform:
            s  = "%s - Data Augmentations: %s\n" % (self.name, len(self.transform.p_d.augmentations))
            s += "\tactivate augmentations: [%s]\n" % ", ".join(self.transform.p_d.augmentations)
            if self.transform.crop:
                s += "\tcrop settings: outputSize (%d, %d)\n" % (self.transform.outputSize[0], self.transform.outputSize[1])
            if self.transform.resize:
                s += "\tresize settings: outputSize (%d, %d)\n" % (self.transform.outputSize[0], self.transform.outputSize[1])

            self.summaryPrint += [s]
            self.summaryLog   += [s]

        print('\n'.join(self.summaryPrint))
        logging.info('\n'.join(self.summaryLog))

    def getFilterInfoString(self) -> str:
        s  = "%s - Data Filter Setup: \n" % (self.name)
        s += "\tdataDirs: %s\n" % (str(self.dataDirs))
        s += "\tfilterTop: %s  exlude: %s\n" % (str(self.filterTop), self.excludeFilterTop)
        s += "\tfilterSim: %s  exlude: %s\n" % (str(self.filterSim), self.excludefilterSim)
        s += "\tfilterFrame: %s\n" % (str(self.filterFrame))
        s += "\tsequenceLength: %s\n" % (str(self.sequenceLength))
        return s


import xarray as xr

class WeatherBenchDataset(Dataset):
    """
    Dataset for WeatherBench data (NetCDF format).
    Assumes data is stored in individual .nc files per variable or combined.
    Loads selected years into memory for efficient training.
    
    Structure: (Time, Channels, Lat, Lon)
    
    Args:
        dataDir (str): Path to the directory containing .nc files.
        vars (List[str]): List of variable names to load (e.g., ['z', 't']). 
                          These must match the variable names inside the NetCDF files.
        years (List[str]): List of years to load (e.g., ['2016', '2017']).
        sequenceLength (int): Length of the time sequence to return.
        levels (List[int], optional): Pressure levels to select if data has a level dimension 
                                      (e.g., [500, 850]). If None, assumes single level or surface data.
        file_pattern (str): F-string pattern to locate files. Default assumes WeatherBench standard 
                            like "{var}_{resolution}.nc" or just "{var}.nc".
        resolution (str): Optional resolution string used in file naming (e.g. "5.625deg").
    """
    def __init__(self, dataDir: str, vars: List[str], years: List[str], sequenceLength: int, 
                 levels: Optional[List[int]] = None, 
                 file_pattern: str = "{var}.nc", 
                 resolution: str = "") -> None:
        super().__init__()
        self.dataDir = dataDir
        self.vars = vars
        self.years = years
        self.levels = levels
        self.seq_len = sequenceLength
        self.resolution = resolution
        
        data_arrays = []

        print(f"Loading WeatherBench data from {dataDir} for years {years}...")

        # 1. Iterate over requested variables (e.g., geopotential, temperature)
        for var_name in self.vars:
            # Construct filename
            fname = file_pattern.format(var=var_name, resolution=self.resolution)
            fpath = os.path.join(dataDir, fname)
            
            if not os.path.isfile(fpath):
                raise FileNotFoundError(f"Could not find WeatherBench file: {fpath}")

            # Open dataset using Xarray
            # We use chunks=None to load lazily initially, then we select years
            ds = xr.open_dataset(fpath)
            
            # 2. Select specific Years
            # WeatherBench uses standard datetime64 indexing
            try:
                ds_subset = ds.sel(time=ds.time.dt.year.isin([int(y) for y in self.years]))
            except Exception as e:
                raise ValueError(f"Error slicing years {self.years} for variable {var_name}. Ensure data has 'time' coordinate. Error: {e}")

            if ds_subset.time.size == 0:
                raise ValueError(f"No data found for years {self.years} in file {fpath}")

            # 3. Select Pressure Levels (if applicable)
            da = ds_subset[var_name] # Extract the DataArray
            
            if 'level' in da.dims and self.levels is not None:
                # If the variable has levels (e.g., z at 500, 850), slice them
                # This might result in (Time, Level, Lat, Lon)
                da = da.sel(level=self.levels)
            elif 'level' in da.dims and self.levels is None:
                # If levels exist but none specified, warn or select all? 
                # Usually better to error out to prevent massive memory usage
                print(f"Warning: Variable {var_name} has levels but no levels specified. Loading all levels.")
            
            # 4. Homogenize Dimensions
            # We want final shape: (Time, Channels, Lat, Lon)
            # If we have (Time, Level, Lat, Lon), we merge Level into Channels conceptually
            # If we have (Time, Lat, Lon), we add a Channel dim
            
            # Load actual data into numpy/RAM here
            vals = da.values # Shape e.g. (8760, 32, 64) or (8760, 2, 32, 64)
            
            # Handle NaNs (common in some raw datasets)
            if np.isnan(vals).any():
                print(f"Warning: NaNs found in {var_name}, replacing with 0.")
                vals = np.nan_to_num(vals)

            tensor_vals = torch.from_numpy(vals).float()

            if len(tensor_vals.shape) == 3: 
                # (Time, Lat, Lon) -> (Time, 1, Lat, Lon)
                tensor_vals = tensor_vals.unsqueeze(1)
            elif len(tensor_vals.shape) == 4:
                # (Time, Level, Lat, Lon) -> (Time, Level, Lat, Lon) 
                # We will concatenate along dim 1 later
                pass
            
            data_arrays.append(tensor_vals)
            ds.close()

        # 5. Concatenate all variables along the channel dimension
        # Result shape: (Total_Time, Total_Channels, Lat, Lon)
        self.data = torch.cat(data_arrays, dim=1)
        
        self.n_time_steps = self.data.shape[0]
        self.n_samples = self.n_time_steps - self.seq_len
        
        print(f"WeatherBench Loaded. Shape: {self.data.shape}. Total valid sequences: {self.n_samples}")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        # Returns a sequence of frames
        # Shape: (Sequence_Length, Channels, Lat, Lon)
        
        # Check bounds
        if idx >= self.n_samples:
            raise IndexError
            
        # Slice the time dimension
        data_slice = self.data[idx : idx + self.seq_len]
        
        # WeatherBench typically doesn't use simParameters like Reynolds number, 
        # but we return an empty dict to match the interface of the other datasets
        return {
            "data": data_slice, 
            "simParameters": {}
        }