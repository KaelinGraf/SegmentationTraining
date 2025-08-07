import pickle
import os
import robi_loader_detectron
from detectron2.data import DatasetCatalog, MetadataCatalog

# These are the classes defined in your training script
ROBI_CLASSES = ["Chrome_screw_synthetic","DIN_connector_synthetic","DSub_connector_synthetic","Eye_bolt_synthetic","Gear_synthetic","Tube_fitting_synthetic","Zigzag_synthetic"]

# Create a mapping from class name to class ID
CLASS_TO_ID = {cls: idx for idx, cls in enumerate(ROBI_CLASSES)}

# Path to your data
DataPath = "/home/kaelin/Documents/BinPicking/SegmentationModel/Data/"

def create_serialized_dataset():
    """
    Loads the ROBI dataset for all classes and serializes it to a pickle file.
    """
    all_datasets_as_dict = {}
    for class_name in ROBI_CLASSES:
        print(f"Loading data for class: {class_name}")
        # Call your robiLoader function to get the data for the current class
        dataset_dicts = robi_loader_detectron.robiLoader(class_name, CLASS_TO_ID[class_name], DataPath)
        all_datasets_as_dict[{class_name}] = dataset_dicts
        print(f"Finished loading {len(dataset_dicts)} samples for class: {class_name}")

    # Define the output file path
    output_file = "robi_dataset.pkl"

    print(f"\nSerializing {len(all_datasets_as_dict)} total samples to {output_file}...")

    # Serialize and save the dataset
    with open(output_file, 'wb') as f:
        pickle.dump(all_datasets_as_dict, f)

    print("Serialization complete.")

if __name__ == '__main__':
    create_serialized_dataset()