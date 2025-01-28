from pathlib import Path
from typing import List


def already_processed(dir_path: Path) -> bool:
    """
    This method checks if the directory has already been processed. 
    """
    translated_pathlist = list(dir_path.glob('MOUSE_*_*_*.nxs'))
    if len(translated_pathlist) == 0:
        return False
    else:
        return translated_pathlist[0].is_file()

def len_files_in_path(dir_path: Path, globstring:str = '*') -> int:
    return len(list(dir_path.glob(globstring)))


def processing_possible(dir_path: Path, return_list:bool = False) -> bool:
    """
    This method checks if the processing is needed for a given repetition. 
    """
    missing_list = []

    if not (len_files_in_path(dir_path, '*/eiger_*_master.h5')==2):
        # missing direct beam and/or direct beam through sample files
        missing_list.append('*/eiger_*_master.h5')

    if not (len_files_in_path(dir_path, '*/im_craw.nxs')==2):
        # missing im_craw for direct beam and/or direct beam through sample files
        missing_list.append('*/im_craw.nxs')

    if not (len_files_in_path(dir_path, 'eiger_*_master.h5') == 1):
        # missing or too many eiger files. 
        missing_list.append('eiger_*_master.h5')

    if not (len_files_in_path(dir_path, 'im_craw.nxs') == 1):
        # missing im_craw files. 
        missing_list.append('im_craw.h5')

    if return_list:
        return missing_list
    elif len(missing_list) > 0:
        return False

    return True