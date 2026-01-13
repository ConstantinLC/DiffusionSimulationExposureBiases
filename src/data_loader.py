import copy

from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, SubsetRandomSampler
from src.dataset import KolmogorovDataset_Rozet, TurbulenceDataset
from src.data_transformations import DataParams, Transforms

def get_data_loaders(data_params):
    """
    Initializes and returns the data loaders for training and validation.
    """
    if data_params["dataset_name"] == "KolmogorovFlow":
        
        trainSet = KolmogorovDataset_Rozet(
            "Test",
            data_params["data_path"],
            mode="train",
            resolution=data_params["resolution"],
            sequenceLength=data_params["sequence_length"],
            framesPerTimeStep=data_params["frames_per_time_step"],
            limit_trajectories=data_params["limit_trajectories_train"]
        )
        valSet = KolmogorovDataset_Rozet(
            "Test",
            data_params["data_path"],
            mode="valid",
            resolution=data_params["resolution"],
            limit_trajectories=data_params["limit_trajectories_val"],
            sequenceLength=data_params["sequence_length"],
            framesPerTimeStep=data_params["frames_per_time_step"]
        )
        trajSet = KolmogorovDataset_Rozet(
            "Test",
            data_params["data_path"],
            mode="valid",
            resolution=data_params["resolution"],
            limit_trajectories=data_params["limit_trajectories_val"],
            sequenceLength=data_params["trajectory_sequence_length"],
            framesPerTimeStep=data_params["frames_per_time_step"]
        )

        train_loader = DataLoader(trainSet, batch_size=data_params["batch_size"], shuffle=True, num_workers=0)
        val_loader = DataLoader(valSet, batch_size=data_params["val_batch_size"], shuffle=False, num_workers=0)
        traj_loader = DataLoader(trajSet, batch_size=data_params["batch_size"], shuffle=False, num_workers=0)

    elif data_params["dataset_name"] == "TransonicFlow":
        
        trainSet = TurbulenceDataset("Training", [data_params["data_path"]], filterTop=['128_tra'], filterSim=[[0,1,2,14,15,16,17,18]], excludefilterSim=True, filterFrame=[(0,1000)],
                    sequenceLength=[data_params["sequence_length"]], randSeqOffset=True, simFields=["dens", "pres"], simParams=["mach"], printLevel="sim")
        
        print(len(trainSet))
        
        testSet = TurbulenceDataset("Test Interpolate Mach 0.66-0.68", [data_params["data_path"]], filterTop=['128_tra'], filterSim=[(16,19)],
                        filterFrame=[(500,750)], sequenceLength=[[60,2]], randSeqOffset=False, simFields=["dens", "pres"], simParams=["mach"], printLevel="sim")
        
        p_d = DataParams(batch=64, augmentations=["normalize"], sequenceLength=trainSet.sequenceLength, randSeqOffset=True,
                 dataSize=[128,64], dimension=2, simFields=["dens","pres"], simParams=["mach"], normalizeMode="machMixed")

        p_d_test = copy.deepcopy(p_d)
        p_d_test.augmentations = ["normalize"]
        p_d_test.sequenceLength = testSet.sequenceLength
        p_d_test.randSeqOffset = False
        p_d_test.batch = 4

        ### TRANSFORMS
        transTrain = Transforms(p_d)
        trainSet.transform = transTrain
        trainSet.printDatasetInfo()
        trainSampler = RandomSampler(trainSet)
        
        transTest = Transforms(p_d_test)
        testSet.transform = transTest
        testSet.printDatasetInfo()
        testSampler = SequentialSampler(testSet)


        ### DATA LOADERS
        train_loader = DataLoader(trainSet, sampler=trainSampler,
                    batch_size=p_d.batch, drop_last=True, num_workers=4)
        val_loader = None
        traj_loader = DataLoader(testSet, sampler=testSampler,
                        batch_size=p_d_test.batch, drop_last=False, num_workers=4)

        

    return train_loader, val_loader, traj_loader