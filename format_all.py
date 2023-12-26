
import os
import shutil
import pathlib

def format_directory(dir:str, ext:list[str], recursive:bool=False):
    """
    Format all files in a directory with a given extension using clang-format
    """
    # check clang-format is installed
    if not shutil.which("clang-format"):
        raise RuntimeError("clang-format not installed")
    
    # check directory exists
    if not os.path.isdir(dir):
        raise RuntimeError("directory does not exist")

    # go through all files in directory
    for file in os.listdir(dir):
        # check file has correct extension
        need_process = False
        if(os.path.isfile(f"{dir}/{file}")):
            p = pathlib.Path(file)
            for e in ext:
                # check extension
                if p.suffix == e:
                    need_process = True
                    break
        
        if need_process:
            # format file
            print(f"-Formatting {dir}/{file}")
            os.system(f"clang-format -i {dir}/{file}")

        
        # check if recursive
        if recursive:
            # check if file is a directory
            if os.path.isdir(f"{dir}/{file}"):
                # format directory
                format_directory(f"{dir}/{file}", ext, recursive)

if __name__ == "__main__":
    # format all files in current directory
    exts = [".cpp", ".cu", ".h", ".inl"]
    format_directory("./benchmarks/", exts, True)
    format_directory("./cuMat/", [".cpp", ".cu", ".h", ".inl", ""], True)
    format_directory("./tests/", exts, True)
    format_directory("./demos/", exts, True)
