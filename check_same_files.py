import os
import sys
#To check if same files exist in different directories! run by python3 /path of dir1 /path of dir2
def check_common_files(folder1, folder2):
    files1 = [f for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))]
    files2 = [f for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f))]
    common_files = set(files1) & set(files2)
    
    if not common_files:
        print("No same files were found!")
    else:
        N =0
        for file in common_files:
            N += 1
            count = files1.count(file) + files2.count(file)
            print(f"{file}")
        print(f"Common files found: {N} ")
            

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python check_common_files.py <folder1> <folder2>")
    else:
        folder1 = sys.argv[1]
        folder2 = sys.argv[2]
        if not os.path.isdir(folder1) or not os.path.isdir(folder2):
            print("Both arguments must be valid folder paths.")
        else:
            check_common_files(folder1, folder2)
