# Dowload project code-net
# tar -xvz file
# Then run the script to get all OCaml and Fortran subfolders.

import os
import shutil

# Set the root directory containing the subfolders like p0004, p0005, etc.
root_dir = 'data'

# Iterate over each subfolder in the root directory
for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)
    if os.path.isdir(subdir_path):  # Check if it's a directory
        keep = []  # List to keep track of special subfolders
        # Iterate over each sub-subfolder (like C, C#, etc.) in the current subfolder
        for folder in os.listdir(subdir_path):
            folder_path = os.path.join(subdir_path, folder)
            if folder in ['OCaml', 'Fortran']:  # Check if it's a special folder
                keep.append(folder)
            else:
                if os.path.isdir(folder_path):  # Ensure it's a directory before deleting
                    print("Deleting", folder_path)
                    shutil.rmtree(folder_path)  # Delete the sub-subfolder

        # If neither OCaml nor Fortran directories were found, delete the subfolder
        if not keep:
            print("Deleting", subdir_path)
            shutil.rmtree(subdir_path)


# Set the destination directories for OCaml and Fortran files
ocaml_dest_dir = 'Combined_OCaml'
fortran_dest_dir = 'Combined_Fortran'

# Create the destination directories if they don't already exist
os.makedirs(ocaml_dest_dir, exist_ok=True)
os.makedirs(fortran_dest_dir, exist_ok=True)

# Function to copy files from a source directory to a destination directory
def copy_files(src_dir, dst_dir):
    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(dst_dir, item)
        if os.path.isfile(s):
            shutil.copy2(s, d)  # Use copy2 to preserve metadata

# Iterate over each subfolder in the root directory
for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)
    if os.path.isdir(subdir_path):  # Check if it's a directory
        # Check for OCaml and Fortran sub-subfolders and copy their contents if they exist
        for language in ['OCaml', 'Fortran']:
            language_path = os.path.join(subdir_path, language)
            if os.path.isdir(language_path):  # Check if the language folder exists
                if language == 'OCaml':
                    copy_files(language_path, ocaml_dest_dir)
                elif language == 'Fortran':
                    copy_files(language_path, fortran_dest_dir)
