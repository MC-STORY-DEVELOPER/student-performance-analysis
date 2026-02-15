import kagglehub
import os
import shutil

def download_dataset():
    print("Downloading dataset...")
    # Download latest version
    path = kagglehub.dataset_download("ayeshasiddiqa123/student-perfirmance")
    
    print("Path to dataset files:", path)
    
    # List files in the downloaded directory
    files = os.listdir(path)
    print("Files in dataset:", files)
    
    # Look for the CSV file
    csv_file = None
    for f in files:
        if f.endswith('.csv'):
            csv_file = os.path.join(path, f)
            break
            
    if csv_file:
        print(f"Found CSV file: {csv_file}")
        # Copy to local analysis directory for easier access
        destination = os.path.join(os.path.dirname(__file__), 'student_performance.csv')
        shutil.copy(csv_file, destination)
        print(f"Copied dataset to: {destination}")
        return destination
    else:
        print("No CSV file found in the dataset.")
        return None

if __name__ == "__main__":
    download_dataset()
