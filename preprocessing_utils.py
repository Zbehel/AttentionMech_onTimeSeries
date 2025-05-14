import mne
import numpy as np

from typing import List, Optional

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score


### 1. Loading Funtions

def Load_raw_labeled(df):
    """
    Load the raw EEG data and labels from the dataframe.
    """
    raw_list = []
    labels = []
    for index, row in df.iterrows():
        # Load the raw EEG data
        raw = mne.io.read_raw_edf(row['edf_path'], preload=True, verbose='ERROR')
        # Append the loaded data and label to the lists
        if raw.duration>60:
            raw_list.append(raw)
            labels.append(row['epilepsy'])
        else:
            print(f"Skipping {row['edf_path']} due to insufficient duration. Duration: {raw.duration} seconds")
    return raw_list, labels


### 2. Preprocessing on Data :

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


def process_raw_files(raw_files: List[mne.io.Raw], 
                      eeg_cols: List[str],
                      segment_duration: float = 60.0,
                      n_segments_per_file: int = 12,
                      samples_per_segment: int = 1250,
                      random_state: Optional[int] = None,
                      epoch_duration: Optional[float] = 5.) -> np.ndarray:
    """
    Process a list of raw MNE files into a batch of epochs with specific EEG channels.
    
    Parameters:
    -----------
    raw_files : List[mne.io.Raw]
        List of raw MNE objects
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
    X = np.zeros((len(raw_files), n_segments_per_file, len(eeg_cols), samples_per_segment))
    
    for i, raw in enumerate(raw_files):
        try:
            # Set different random seed for each file if random_state is provided
            file_random_state = None if random_state is None else random_state + i
            
            # Pick only the specified EEG channels
            available_channels = raw.ch_names
            channels_to_use = [ch for ch in available_channels if ch.replace('-REF','').replace('-LE','') in eeg_cols]
            
            if not channels_to_use:
                raise ValueError(f"None of the specified EEG channels found in file {i}")
            
            if len(channels_to_use) < len(eeg_cols):
                print(f"Warning: Only {len(channels_to_use)}/{len(eeg_cols)} EEG channels found in file {i}")
                
            # Select only the required channels
            raw_eeg = raw.copy().pick_channels(channels_to_use)
            
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
            X[i, :, :len(channels_to_use), :] = epoch_data
            
        except Exception as e:
            print(f"Error processing file {i}: {str(e)}")
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
    mean = np.mean(X, axis=(0, 1, 3), keepdims=True)
    std = np.std(X, axis=(0, 1, 3), keepdims=True)
    
    # Standardize the data
    X_standardized = (X - mean) / std
    
    return X_standardized

def compute_pearson_correlation(eeg_data):
  """
  Calcule la matrice de corrélation de Pearson pour une acquisition EEG.

  Paramètres :
  - eeg_data (numpy.ndarray) : Données EEG de forme (21, 1250), où 21 est le nombre de canaux
    et 1250 est le nombre d'échantillons temporels.

  Retourne :
  - corr_matrix (numpy.ndarray) : Matrice de corrélation de taille (21,21),
    où chaque canal a une matrice de corrélation de Pearson entre les échantillons temporels.
  """
  eeg_data = eeg_data.copy()

  corr_matrix = np.corrcoef(eeg_data)

  return corr_matrix


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
    corr_matrix = np.zeros((X.shape[0], X.shape[1], X.shape[2], X.shape[2]))
    for i in range(X.shape[0]): # for each acquisition
        for j in range(X.shape[1]): # for each segment 5 secs
            # Compute the correlation matrix
            temp = np.corrcoef(X[i][j])
            corr_matrix[i][j] = np.nan_to_num(temp)
    
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
    n_samples, n_segments, n_channels, = corr_matrices.shape[0], corr_matrices.shape[1], corr_matrices.shape[2]
    n_features = n_channels * (n_channels - 1) // 2
    
    flattened = np.zeros((n_samples, n_segments, n_features))
    
    for i in range(n_samples):
        for j in range(n_segments):
            # Get upper triangle indices (excluding diagonal)
            upper_indices = np.triu_indices(n_channels, k=1)
            # Extract values
            flattened[i] = corr_matrices[i][j][upper_indices]
    
    return flattened


### 3. Evaluation

from sklearn.metrics import f1_score, precision_score, recall_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate F1 score, sensitivity (recall), and precision.

    Parameters:
    - y_true: Ground truth labels
    - y_pred: Predicted labels

    Returns:
    - metrics: Dictionary containing F1 score, sensitivity, and precision
    """
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)  # Sensitivity

    metrics = {
        'F1 Score': f1,
        'Precision': precision,
        'Sensitivity (Recall)': recall
    }
    return metrics

