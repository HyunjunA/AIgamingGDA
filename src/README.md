# Grand Driving Auto

## Data Collection:
Must have game running in a 800x600 window in the top left corner of the screen

    python get_data.py <output_dir>

Data will be saved in the output dir, in files called training_data-{file_num}.npy
if files are already present in the output dir, the new data will be stored at the next file_num increment.
500 images per training data file

## Check Data/Train a simple CNN Model:

    python check_data.py <data_dir>
## Check Data/Train a AlexNet or CNN:    
    python check_data_various_models.py <model_name> <data_dir>
    e.g.
    If you want to train AlexNet 
    python check_data_various_models.py AlexNet <data_dir>
    If you want to train CNN
    python check_data_various_models.py CNN <data_dir>


    python check_data_various_models_v3.py <model_name> <data_dir> <epochs> <batchsize>


## Model Testing
Must have game running in a 800x600 window in the top left corner of the screen
    
    python test_model.py <absolute_path_to_saved_model>