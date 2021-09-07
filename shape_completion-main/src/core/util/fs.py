import os
import pickle
import shutil
import sys
import tempfile as tmp
from contextlib import contextmanager
from pathlib import Path
from typing import Union

import psutil


# ----------------------------------------------------------------------------------------------------------------------
#                                             File System
# ----------------------------------------------------------------------------------------------------------------------

def restart_program():
    """Restarts the current program, with file objects and descriptors
       cleanup
    """

    try:
        p = psutil.Process(os.getpid())
        for handler in p.open_files() + p.connections():
            os.close(handler.fd)
    except Exception as e:
        print(e)

    python = sys.executable
    os.execl(python, python, *sys.argv)


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def assert_new_dir(dp, parents=False):
    if dp.is_dir():
        shutil.rmtree(dp, ignore_errors=True)

    e = None
    for retry in range(100):
        try:
            dp.mkdir(parents=parents)
            break
        except OSError as e:
            pass
    else:
        raise e


@contextmanager
def tempfile(suffix='', dir=None):
    """ Context for temporary file.

    Will find a free temporary filename upon entering
    and will try to delete the file on leaving, even in case of an exception.

    Parameters
    ----------
    suffix : string
        optional file suffix
    dir : string
        optional directory to save temporary file in
    """

    tf = tmp.NamedTemporaryFile(delete=False, suffix=suffix, dir=dir)
    tf.file.close()
    try:
        yield tf.name
    finally:
        try:
            os.remove(tf.name)
        except OSError as e:
            if e.errno == 2:
                pass
            else:
                raise


@contextmanager
def open_atomic(filepath, *args, **kwargs):
    """ Open temporary file object that atomically moves to destination upon
    exiting.

    Allows reading and writing to and from the same filename.

    The file will not be moved to destination in case of an exception.

    Parameters
    ----------
    filepath : string
        the file path to be opened
    fsync : bool
        whether to force write the file to disk
    *args : mixed
        Any valid arguments for :code:`open`
    **kwargs : mixed
        Any valid keyword arguments for :code:`open`
    """
    fsync = kwargs.get('fsync', False)

    with tempfile(dir=os.path.dirname(os.path.abspath(filepath))) as tmppath:
        with open(tmppath, *args, **kwargs) as file:
            try:
                yield file
            finally:
                if fsync:
                    file.flush()
                    os.fsync(file.fileno())
        os.rename(tmppath, filepath)


def file_extension(fp):
    return os.path.splitext(str(fp))[1][1:].lower()


def align_file_extension(fp, tgt):
    full_tgt = tgt if tgt[0] == '.' else f'.{tgt}'
    fp = str(fp)  # Support for pathlib.Path
    if fp.endswith(full_tgt):
        return fp
    else:
        return fp + full_tgt


def assert_is_dir(d):
    assert os.path.isdir(d), f"Directory {d} is invalid"


def get_exp_version(cache_dir):
    last_version = -1
    try:
        for f in os.listdir(cache_dir):
            if 'version_' in f:
                file_parts = f.split('_')
                version = int(file_parts[-1])
                last_version = max(last_version, version)
    except:  # No such dir # TODO - detect the explict error
        pass

    return last_version + 1


def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def pkl_load(fp: Union[Path, str]):
    if not isinstance(fp, Path):
        fp = Path(fp)
    with fp.open(mode='rb') as f:
        return pickle.load(f)


def pkl_dump(fp: Union[Path, str], obj):
    if not isinstance(fp, Path):
        fp = Path(fp)
    fp.parents[0].mkdir(exist_ok=True, parents=True)
    with fp.open(mode='wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def next_cache_version(cache_dir: Path):
    if not cache_dir.is_dir():  # TODO - Anti-pattern to place this here
        cache_dir.mkdir(parents=True)
        return 0
    last_version = -1
    for f in cache_dir.glob('*'):
        name = f.stem
        if 'version_' in name:  # File adhering to the name convention
            file_parts = name.split('_')
            version = int(file_parts[-1])  # Presuming version is the last thing to appear before the extension dot
            last_version = max(last_version, version)

    return last_version + 1
