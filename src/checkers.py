from pathlib import Path


def already_processed(dir_path: Path) -> bool:
    """
    This method checks if the directory has already been processed. 
    """
    translated_path = dir_path / 'translated.nxs'
    return translated_path.is_file()

def processing_possible(dir_path: Path) -> bool:
    """
    This method checks if the processing is needed for a given repetition. 
    """
    craw_path = dir_path / 'im_craw.nxs'
    translated_path = dir_path / 'translated.nxs'

    if not craw_path.is_file():
        return False

    if not (len(craw_path.glob('*/eiger_*_master.h5'))==2):
        # missing direct beam and/or direct beam through sample files
        return False

    if not (len(craw_path.glob('*/im_craw.nxs'))==2):
        # missing im_craw for direct beam and/or direct beam through sample files
        return False

    if translated_path.is_file():
        # already translated
        return False

    if not len(list(dir_path.glob('eiger_*_master.h5'))) == 1:
        # missing or too many eiger files. 
        return False

    return True