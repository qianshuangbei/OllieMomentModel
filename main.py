import chardet

def detect_file_encoding(file_path):
    # Read the raw bytes from the file
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    
    # Detect the encoding
    result = chardet.detect(raw_data)
    
    print(f"File: {file_path}")
    print(f"Detected encoding: {result['encoding']}")
    print(f"Confidence: {result['confidence']}")
    
    # Show first few bytes for debugging
    print(f"First few bytes (hex): {raw_data[:20].hex(' ')}")

if __name__ == "__main__":
    file_path = "data/train/sv4_000062_aug_04.jpg_pose.txt"
    detect_file_encoding(file_path)
    with open(file_path, 'r', encoding='ascii') as pf:
        lines = pf.readlines()