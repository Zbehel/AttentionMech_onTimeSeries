import mne
import numpy as np
import pandas as pd
from typing import List, Optional
import random

from EpilepsyNet_model import *

def extract_random_segment(raw: mne.io.Raw, duration: float = 60.0, 
                          random_state: Optional[int] = None) -> mne.io.Raw:
    """
    Extract a random segment of specified duration from a raw MNE file.
    
    Parameters:
    -----------
    raw : mne.io.Raw
        The raw MNE object
    duration : float
        Duration of the segment to extract in seconds
    random_state : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    mne.io.Raw
        A cropped raw object containing only the random segment
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    # Get the total duration of the raw file
    total_duration = raw.times[-1]
    
    # Ensure the raw file is long enough
    if total_duration <= duration:
        raise ValueError(f"Raw file duration ({total_duration:.2f}s) is shorter than requested segment duration ({duration:.2f}s)")
    
    # Generate a random start time
    max_start = total_duration - duration
    start_time = np.random.uniform(0, max_start)
    end_time = start_time + duration
    
    # Create a copy and crop to the random segment
    raw_segment = raw.copy().crop(tmin=start_time, tmax=end_time)
    
    return raw_segment


def segment_to_epochs(raw_segment: mne.io.Raw, n_segments: int = 12) -> mne.Epochs:
    """
    Convert a raw segment into fixed-length epochs.
    
    Parameters:
    -----------
    raw_segment : mne.io.Raw
        The raw segment to convert to epochs
    n_segments : int
        Number of segments to create
    
    Returns:
    --------
    mne.Epochs
        Epoch object containing the segmented data
    """
    # Calculate duration of each epoch based on total duration and number of segments
    total_duration = raw_segment.times[-1]
    epoch_duration = total_duration / n_segments
    
    # Create fixed-length epochs
    epochs = mne.make_fixed_length_epochs(
        raw_segment, 
        duration=epoch_duration,
        preload=True,
        reject_by_annotation=True
    )
    
    return epochs


def process_raw_files(raw_file: mne.io.Raw, 
                      eeg_cols: List[str],
                      segment_duration: float = 60.0,
                      n_segments_per_file: int = 12,
                      samples_per_segment: int = 1250,
                      random_state: Optional[int] = None) -> np.ndarray:
    """
    Process a list of raw MNE files into a batch of epochs with specific EEG channels.
    
    Parameters:
    -----------
    raw_files : mne.io.Raw
        The raw MNE object to make preds on
    eeg_cols : List[str]
        List of EEG channel names to keep
    segment_duration : float
        Duration of random segment to extract from each file in seconds
    n_segments_per_file : int
        Number of segments to create per file
    samples_per_segment : int
        Number of time samples per segment
    random_state : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    np.ndarray
        Array of shape (len(raw_files), n_segments_per_file, len(eeg_cols), samples_per_segment)
    """
    # Initialize the output array
    X = np.zeros((n_segments_per_file, len(eeg_cols), samples_per_segment))
    
    # Define duration of each epoch based on number of segment and total duration
    epoch_duration = segment_duration / n_segments_per_file
    

    try:
        # Set different random seed for each file if random_state is provided
        file_random_state = None if random_state is None else random_state
        
        # Pick only the specified EEG channels
        available_channels = raw_file.ch_names
        print('Num of availbable ch :', len(available_channels))
        channels_to_use = [ch for ch in available_channels if ch.replace('-REF','').replace('-LE','') in eeg_cols]
        if not channels_to_use:
            raise ValueError(f"None of the specified EEG channels found in file")
        
        if len(channels_to_use) < len(eeg_cols):
            print(f"Warning: Only {len(channels_to_use)}/{len(eeg_cols)} EEG channels found in file")
            
        # Select only the required channels
        raw_eeg = raw_file.copy().pick_channels(channels_to_use)
        
        # Resample to 250Hz
        current_sfreq = int(raw_eeg.info['sfreq'])
        if current_sfreq != 250:
            print(f"🔁 Resample : {current_sfreq} Hz → {250} Hz")
            raw_eeg.resample(250)

        # Extract random segment
        raw_segment = extract_random_segment(
            raw_eeg, 
            duration=segment_duration,
            random_state=file_random_state
        )
        
        # Convert to epochs
        epochs = segment_to_epochs(raw_segment, n_segments=n_segments_per_file)
        
        # Get the data as array
        epoch_data = epochs.get_data()
        
        # Ensure the data has the correct number of time samples
        if epoch_data.shape[2] != samples_per_segment:
            # Resample if necessary
            resampling_freq = samples_per_segment / (epoch_duration / n_segments_per_file)
            raw_segment.resample(resampling_freq)
            epochs = segment_to_epochs(raw_segment, n_segments=n_segments_per_file)
            epoch_data = epochs.get_data()
        
        # Store in the output array
        X[:, :len(channels_to_use), :] = epoch_data
        
    except Exception as e:
        print(f"Error processing file {str(e)}")
        # Keep zeros in the output array for this file
    
    return X

# Standardize the data per channel :
def standardize_data(X: np.ndarray) -> np.ndarray:
    """
    Standardize the data along the last axis (time samples).
    
    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_segments, n_channels, n_time_samples)
    
    Returns:
    --------
    np.ndarray
        Standardized data
    """
    # Compute mean and std for each channel across all segments and samples
    mean = np.mean(X, axis=(1), keepdims=True)
    std = np.std(X, axis=(1), keepdims=True)
    print(X.shape)
    
    # Standardize the data
    X_standardized = (X - mean) / std
    
    return X_standardized


# Compute Correlation Matrix
def compute_correlation_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute the correlation matrix for the data.
    
    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_segments, n_channels, n_time_samples)
    
    Returns:
    --------
    np.ndarray
        Correlation matrices of shape (n_samples, n_segments, n_channels, n_channels)
    """
    # Declare corr_matrix np array of shape (n_samples, n_segments,n_channels, n_channels)
    corr_matrix = np.zeros((X.shape[0], X.shape[1], X.shape[1]))
    for j in range(X.shape[0]): # for each segment 5 secs
        # Compute the correlation matrix
        temp = np.corrcoef(X[j])
        corr_matrix[j] = np.nan_to_num(temp)
    
    return corr_matrix



# Discard bottom triangle from the matrix:
def discard_bottom_triangle(matrix):
    """
    Discard the bottom triangle of a square matrix.

    Parameters:
    - matrix (numpy.ndarray): The input square matrix.

    Returns:
    - numpy.ndarray: The matrix with the bottom triangle discarded.
    """
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    
    # Apply the mask to the matrix
    upper_triangle = np.where(mask, matrix, 0)
    
    return upper_triangle

def extract_upper_triangle(corr_matrices):
    """
    Extract upper triangles from correlation matrices
    
    Args:
        corr_matrices: numpy array of shape (n_sample, n_segments, n_channels, n_channels)
        
    Returns:
        numpy array of shape (n_segments, n_features) where n_features = n_channels*(n_channels-1)/2
    """
    n_segments, n_channels, = corr_matrices.shape[0], corr_matrices.shape[1]
    n_features = n_channels * (n_channels - 1) // 2
    
    flattened = np.zeros((n_segments, n_features))
    
    for j in range(n_segments):
        # Get upper triangle indices (excluding diagonal)
        upper_indices = np.triu_indices(n_channels, k=1)
        # Extract values
        flattened[j] = corr_matrices[j][upper_indices]

    return flattened

def main():

    # EEG channels used to prediction
    eeg_cols = ['EEG FP1', 'EEG FP2', 'EEG F3', 'EEG F4', 
                'EEG C3', 'EEG C4', 'EEG P3', 'EEG P4', 
                'EEG O1', 'EEG O2', 'EEG F7', 'EEG F8', 
                'EEG T3', 'EEG T4', 'EEG T5', 'EEG T6', 
                'EEG T1', 'EEG T2', 'EEG FZ', 'EEG CZ',
                'EEG PZ']
                
            
    # eeg_cols = ['EEG FP1-LE','EEG FP2-LE','EEG F3-LE', 'EEG F4-LE',
    #             'EEG C3-LE', 'EEG C4-LE', 'EEG T1-LE', 'EEG T2-LE'
    #             'EEG P3-LE', 'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE',
    #             'EEG F7-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG T4-LE', 
    #             'EEG T5-LE', 'EEG T6-LE', 'EEG FZ-LE', 'EEG CZ-LE', 
    #             'EEG PZ-LE']
                

    parameters = {
        'eeg_cols':eeg_cols,
        'segment_duration':60.0,        # 60 second segments
        'n_segments_per_file':12,       # Split into 12 epochs (5 sec each)
        'samples_per_segment':1250,     # 1250 samples per segment (250 Hz sampling rate)
        'random_state':42  
        }

    df_test = pd.read_csv('test_path.csv')
    paths = [
            'EEG_Epilepsy/00_epilepsy/aaaaamnj/s012_2016/01_tcp_ar/aaaaamnj_s012_t005.edf',
            'EEG_Epilepsy/00_epilepsy/aaaaamnj/s012_2016/01_tcp_ar/aaaaamnj_s012_t011.edf',           
            'EEG_Epilepsy/00_epilepsy/aaaaamnj/s012_2016/01_tcp_ar/aaaaamnj_s012_t010.edf',
            'EEG_Epilepsy/00_epilepsy/aaaaamnj/s012_2016/01_tcp_ar/aaaaamnj_s012_t004.edf',
            'EEG_Epilepsy/00_epilepsy/aaaaamnj/s012_2016/01_tcp_ar/aaaaamnj_s012_t012.edf',
            'EEG_Epilepsy/00_epilepsy/aaaaamnj/s012_2016/01_tcp_ar/aaaaamnj_s012_t006.edf',
            'EEG_Epilepsy/00_epilepsy/aaaaamnj/s012_2016/01_tcp_ar/aaaaamnj_s012_t007.edf',
            'EEG_Epilepsy/00_epilepsy/aaaaamnj/s012_2016/01_tcp_ar/aaaaamnj_s012_t009.edf',
            'EEG_Epilepsy/00_epilepsy/aaaaamnj/s012_2016/01_tcp_ar/aaaaamnj_s012_t008.edf',
            'EEG_Epilepsy/00_epilepsy/aaaaanwn/s003_2014/01_tcp_ar/aaaaanwn_s003_t001.edf',

            'EEG_Epilepsy/01_no_epilepsy/aaaaamey/s001_2011/02_tcp_le/aaaaamey_s001_t000.edf',
            'EEG_Epilepsy/01_no_epilepsy/aaaaapkt/s001_2013/01_tcp_ar/aaaaapkt_s001_t000.edf',
            'EEG_Epilepsy/01_no_epilepsy/aaaaapkt/s001_2013/01_tcp_ar/aaaaapkt_s001_t001.edf',
            'EEG_Epilepsy/01_no_epilepsy/aaaaaocx/s001_2012/01_tcp_ar/aaaaaocx_s001_t000.edf',
            'EEG_Epilepsy/01_no_epilepsy/aaaaaooz/s001_2013/01_tcp_ar/aaaaaooz_s001_t000.edf',
            'EEG_Epilepsy/01_no_epilepsy/aaaaalpi/s001_2011/02_tcp_le/aaaaalpi_s001_t004.edf',
            'EEG_Epilepsy/01_no_epilepsy/aaaaalpi/s001_2011/02_tcp_le/aaaaalpi_s001_t002.edf',
            'EEG_Epilepsy/01_no_epilepsy/aaaaalpi/s001_2011/02_tcp_le/aaaaalpi_s001_t003.edf',
            'EEG_Epilepsy/01_no_epilepsy/aaaaalpi/s001_2011/02_tcp_le/aaaaalpi_s001_t001.edf',
            'EEG_Epilepsy/01_no_epilepsy/aaaaalpi/s002_2011/02_tcp_le/aaaaalpi_s002_t001.edf'
            ]

    predictions, probas = [], []
    correct=0

    datas = df_test.sample(200)
    total = len(datas)
    for row in datas.itertuples():
        path = row.edf_path
        y_true = row.epilepsy
        raw = mne.io.read_raw_edf(path,
                                preload=True,
                                verbose='ERROR')
        X = process_raw_files(
            raw_file=raw,
            eeg_cols=eeg_cols,
            segment_duration=parameters['segment_duration'],
            n_segments_per_file=parameters['n_segments_per_file'],
            random_state=parameters['random_state']
            )

        X_std = standardize_data(X)
        corr_matrix = compute_correlation_matrix(X_std)
        # print('Correlation matrix shape :',corr_matrix.shape)

        upper_triangle_matrix = extract_upper_triangle(corr_matrix)
        # print('Upper Triangle shape :',upper_triangle_matrix.shape)

        X_tensor = torch.tensor(upper_triangle_matrix, dtype=torch.float32)
        X_tensor = X_tensor.unsqueeze(0)
        
        # Model parameters
        input_dim = 210  # Size of flattened upper triangle (21*20/2)
        embed_dim = 256  # Embedding dimension
        num_heads = 16    # Number of attention heads

        
        model = TimeSeriesAttentionClassifier(input_dim, embed_dim, num_heads)
        model.load_state_dict(torch.load('Weights_model/EpilepsyNet_0904.pth'))
        model.eval()
        print('¨'*50)
        print('Model Prediction :')

        outputs, _ = model(X_tensor)
       
        # For binary classification with sigmoid, prediction is 1 if output > 0.5
        predicted = (outputs > 0.5).float()       
        if predicted == y_true:
            correct+=1
        print('¨'*50)
        predictions.append(int(predicted))
        
        print(path)
    print(f'Accuracy:{round(100*correct/total,3)}%')
if __name__ == "__main__":
    main()




