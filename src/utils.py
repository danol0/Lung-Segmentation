import pydicom
import os
import numpy as np
import torch
import pickle


def DSC(pred, target, binary=True):
    """
    Calculate the dice similarity coefficient for a batch of predictions and targets.

    Args:
        pred (torch.Tensor): Logit predictions, shape (N, 1, H, W)
        target (torch.Tensor): Binary targets, shape (N, 1, H, W)
        binary (bool): Whether to threshold the predictions at 0.5

    Returns:
        DSC (torch.Tensor): Dice similarity coefficient, shape (N,)
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float() if binary else pred
    numer = (pred * target).sum(dim=(2, 3))
    denom = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return (2 * numer) / (denom + 1e-8)


def binary_accuracy(pred, target):
    """
    Calculate the binary accuracy for a batch of predictions and targets.

    Args:
        pred (torch.Tensor): Logit predictions, shape (N, 1, H, W)
        target (torch.Tensor): Binary targets, shape (N, 1, H, W)

    Returns:
        accuracy (torch.Tensor): Binary accuracy, shape (N,)
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    correct = (pred == target).sum(dim=(2, 3))
    return correct / (target.shape[2] * target.shape[3])


def load_data_splits(fname, data_dir="Dataset"):
    """
    Loads the dataset from a file or creates it if it does not exist.

    Args:
        fname (str): File name of the pkl file containing the data splits.
        data_dir (str): The path to the directory containing the DICOM files and segmentation files.

    Returns:
        splits (dict): A dictionary containing the raw data splits. The keys are "train" and "test".
    """

    fname = os.path.join(data_dir, fname)
    # Check dataset exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found at {data_dir}")
    # Check if splits exist
    if not os.path.exists(fname):
        print(f"No split file found at {fname}, creating data splits...")
        aligned_data = create_aligned_dataset(data_dir)
        splits = create_raw_splits(aligned_data)
        with open(fname, "wb") as f:
            pickle.dump(splits, f)

    # Load dataset
    splits = pickle.load(open(fname, "rb"))
    print(
        f"Loaded data splits with {len(splits['train'])} train cases and {len(splits['test'])} test cases"
    )
    return splits


def dicom2dict(data_dir, deidentify=False, verbose=False, info=False):
    """
    Reads DICOM files from a directory and converts them to a numpy array.
    The function optionally de-identifies and re-saves the files.

    Args:
        data_dir (str): The path to the directory containing the DICOM files.
        deidentify (bool): Whether to de-identify the DICOM files.
        verbose (bool): Whether to print the shape of the numpy arrays for each case.
        info (bool): Whether to return a selection of metadata for each case.

    Returns:
        img_data (dict): A dictionary containing the numpy arrays for each case. The keys are the case IDs.
    """

    # Read the DICOM files
    case_dirs = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)])
    print(f"Identified {len(case_dirs)} cases")
    all_fpaths = []
    for case_dir in case_dirs:
        sorted_fnames = sorted(
            os.listdir(case_dir), key=lambda x: int(x.split("-")[1].split(".")[0])
        )
        all_fpaths.append([os.path.join(case_dir, f) for f in sorted_fnames])
    print(f"Identified {sum([len(x) for x in all_fpaths])} dicom files")

    # De-identify the DICOM files
    if deidentify:
        print("De-identifying the DICOM files...")
        for case in all_fpaths:
            case_id = case[0].split("/")[-2]
            for dicom_file in case:
                ds = pydicom.dcmread(dicom_file)
                ds.PatientID = case_id
                ds.PatientName = ""
                ds.PatientBirthDate = ""
                try:
                    del ds["PatientBirthTime"]
                except KeyError:
                    pass
                ds.save_as(dicom_file)

    # Create a numpy array from the DICOM files for each case
    img_data = {}
    info_array = []
    for case in all_fpaths:
        flag = False
        case_array = []
        case_id = case[0].split("/")[-2]
        for dicom_file in case:
            ds = pydicom.dcmread(dicom_file)
            if not flag:
                info_array.append(
                    [
                        case_id,
                        ds.PatientSex,
                        ds.PatientPosition,
                        ds.Modality,
                        ds.BodyPartExamined,
                        ds.Manufacturer,
                        ds.ManufacturerModelName,
                        ds.KVP,
                        ds.SliceThickness,
                        ds.RescaleSlope,
                        ds.RescaleIntercept,
                    ]
                )
                flag = True
            sex = ds.PatientSex
            image_array = ds.pixel_array
            # Use int16 to save memory
            image_array = image_array.astype(np.int16)
            # convert to HU
            image_array = image_array * ds.RescaleSlope + ds.RescaleIntercept
            slice_id = int(ds.InstanceNumber)
            case_array.append((slice_id, image_array))
        case_array.sort(key=lambda x: x[0])
        case_array = np.array([x[1] for x in case_array])

        img_data[f"{case_id}_{sex}"] = case_array
        print(f"Case {case_id} has shape {case_array.shape}") if verbose else None
    return img_data if not info else (img_data, info_array)


def seg2dict(data_dir, verbose=False):
    """
    Reads the segmentation files from a directory and converts them to a dictionary.

    Args:
        data_dir (str): The path to the directory containing the segmentation files.
        verbose (bool): Whether to print the shape of the segmentation masks for each case.

    Returns:
        seg_data (dict): A dictionary containing the segmentation masks for each case. The keys are the case IDs.
    """

    # Read the segmentation files
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Segmentation directory not found at {data_dir}")
    seg_fpaths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    print(f"Identified {len(seg_fpaths)} segmentation files")
    seg_fpaths.sort()

    # Create a dictionary from the segmentation files
    seg_data = {}
    for seg_fpath in seg_fpaths:
        case_id_seg = seg_fpath.split("/")[-1].split(".")[0]
        case_id = "_".join(case_id_seg.split("_")[:-1])
        seg = np.load(seg_fpath)
        seg = seg["masks"].astype(np.int16)
        seg = seg[::-1]
        seg_data[case_id] = seg
        print(f"Case {case_id} has shape {seg.shape}") if verbose else None

    return seg_data


def create_aligned_dataset(data_dir):
    """
    Creates a dataset containing the images aligned with corresponding segmentations.

    Args:
        data_dir (str): The path to the directory containing the DICOM files and segmentation files.

    Returns:
        aligned_data (dict): A dictionary containing the aligned images and segmentations for each case.
            The keys are the case IDs.
    """

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found at {data_dir}")

    # Read the DICOM files and segmentations
    img_data = dicom2dict(os.path.join(data_dir, "Images"))
    seg_data = seg2dict(os.path.join(data_dir, "Segmentations"))

    # Align the images and segmentations
    aligned_data = {}
    assert sorted(["_".join(k.split("_")[:-1]) for k in img_data.keys()]) == sorted(
        list(seg_data.keys())
    ), "Case IDs do not match between images and segmentations"
    for case_id in img_data.keys():
        # Images are indexed by case_id_sex, segmentations are indexed by case_id
        img = img_data[case_id]
        seg = seg_data["_".join(case_id.split("_")[:-1])]
        assert (
            img.shape[0] == seg.shape[0]
        ), f"Number of images and segmentations do not match for case {case_id}"
        assert (
            img.shape == seg.shape
        ), f"Shape of images and segmentations do not match for case {case_id}"
        aligned_data[case_id] = {"imgs": img, "segs": seg}

    return aligned_data


def create_raw_splits(aligned_data, test_size=0.34, random_state=42):
    """
    Unpacks the aligned dataset into raw splits containing the images and segmentations.

    Args:
        aligned_data (dict): A dictionary containing the aligned images and segmentations for each case. The keys are the case IDs.
        test_size (float): The proportion of cases to include in the test split.
        random_state (int): The random seed for reproducibility.

    Returns:
        output (dict): A dictionary containing the raw data splits. The keys are "train" and "test",
            the values are dictionaries containing the images and segmentations for each case.
    """
    np.random.seed(random_state)

    train_data = {}
    test_data = {}

    cases = list(aligned_data.keys())
    male_cases = [case for case in cases if case.split("_")[-1] == "M"]
    female_cases = [case for case in cases if case.split("_")[-1] == "F"]

    test_cases = []
    for c in [male_cases, female_cases]:
        n_test = int(test_size * len(c))
        test_cases.extend(np.random.choice(c, n_test, replace=False))
    print(f"Test cases: {test_cases}")
    for case_id_sex in cases:
        imgs = aligned_data[case_id_sex]["imgs"]
        segs = aligned_data[case_id_sex]["segs"]
        # Remove sex from case_id
        case_id = "_".join(case_id_sex.split("_")[:-1])
        if case_id_sex in test_cases:
            for i in range(imgs.shape[0]):
                test_data[f"{case_id}_{i + 1}"] = {"img": imgs[i], "seg": segs[i]}
        else:
            for i in range(imgs.shape[0]):
                train_data[f"{case_id}_{i + 1}"] = {"img": imgs[i], "seg": segs[i]}

    output = {"train": train_data, "test": test_data}

    return output
