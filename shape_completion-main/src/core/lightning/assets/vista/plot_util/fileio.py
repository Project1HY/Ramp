import os
import pickle
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Union
import hashlib

# TODO -
#  * Use the `safer` package to install atomic_read, atomic_write
#    https://github.com/rec/safer
#    https://github.com/untitaker/python-atomicwrites
#  * Maybe it is better to simply use the numpy.load with pickle_enabled=True?
# ----------------------------------------------------------------------------------------------------------------------
#                                                      Output Streams
# ----------------------------------------------------------------------------------------------------------------------


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def prepend_text(file, text, after=None):
    """ Prepend file with given raw text """
    f_read = open(file, 'r')
    buff = f_read.read()
    f_read.close()
    f_write = open(file, 'w')
    inject_pos = 0
    if after:
        pattern = after
        inject_pos = buff.find(pattern) + len(pattern)
    f_write.write(buff[:inject_pos] + text + buff[inject_pos:])
    f_write.close()


def append_line(file, line):
    """ Append a line to a file """
    if not os.path.exists(file):
        print("Can't append to file '%s'. File does not exist." % file)
        return
    f = open(file, 'a')
    f.write(line + '\n')
    f.close()


def replace_line(file, line, replacement):
    """ Replace a single line """
    f_read = open(file, 'r')
    lines = f_read.readlines()
    f_read.close()
    for i in range(len(lines)):
        if line.rstrip() == lines[i].rstrip():
            lines[i] = replacement.rstrip() + '\n'
            break
    f_write = open(file, 'w')
    f_write.write(''.join(lines))
    f_write.close()


def replace_all(file, match, replacement):
    """ Replaces all occurences of matching string with the replacement string """
    f_read = open(file, 'r')
    buff = f_read.read()
    f_read.close()
    f_write = open(file, 'w')
    f_write.write(buff.replace(match, replacement))
    f_write.close()


def make_file(file, content):
    """ Creates a new file with specific content """
    f = open(file, 'a')
    f.write(content)
    if not content.endswith('\n'):
        f.write('\n')
    f.close()


# ----------------------------------------------------------------------------------------------------------------------
#                                                         Queries
# ----------------------------------------------------------------------------------------------------------------------
def file_extension(fp):
    return os.path.splitext(str(fp))[1][1:].lower()


def align_file_extension(fp, tgt):
    full_tgt = tgt if tgt[0] == '.' else f'.{tgt}'
    fp = str(fp)  # Support for pathlib.Path
    if fp.endswith(full_tgt):
        return fp
    else:
        return fp + full_tgt


def file_parts(filename):
    """
    Splits the input file name string into path, name and extension

    Parameters
    ----------
    filename : str
        a file name

    Returns
    -------
    (str,str,str)
        the path, name and extension of the input file
    """

    path, name = os.path.split(filename)
    name, ext = os.path.splitext(name)
    return path, name, ext


def is_module(path):
    """
    Returns True if the given path is a Python module, False otherwise

    Parameters
    ----------
    path : str
        the path to check

    Returns
    -------
    bool
        True if the given path is a Python module, False otherwise
    """

    return os.path.isfile(path) and path.endswith('.py')


def is_package(path):
    """
    Returns True if the given path is a Python package, False otherwise

    Parameters
    ----------
    path : str
        the path to check

    Returns
    -------
    bool
        True if the given path is a Python package, False otherwise
    """

    return os.path.isdir(path) and ('__init__.py' in os.listdir(path))


def distance_to_end(file_obj):
    """
    For an open file object how far is it to the end
    Parameters
    ------------
    file_obj: open file-like object
    Returns
    ----------
    distance: int, bytes to end of file
    """
    position_current = file_obj.tell()
    file_obj.seek(0, 2)
    position_end = file_obj.tell()
    file_obj.seek(position_current)
    distance = position_end - position_current
    return distance


# ----------------------------------------------------------------------------------------------------------------------
#                                             File System
# ----------------------------------------------------------------------------------------------------------------------

def crawl_directory(directory, ignore_hidden=True):
    """
    Given a directory, return a dict representing the tree of files under that directory.

    :param directory: A string representing a directory.
    :return: A dict<file_or_dir_name: content> where:
        file_or_dir_name is the name of the file within the parent directory.
        content is either
        - An absolute file path (for files) or
        - A dictionary containing the output of crawl_directory for a subdirectory.
    """
    contents = os.listdir(directory)
    if ignore_hidden:
        contents = [c for c in contents if not c.startswith('.')]
    this_dir = {}
    for c in contents:
        full_path = os.path.join(directory, c)
        if os.path.isdir(full_path):
            this_dir[c] = crawl_directory(full_path)
        else:
            this_dir[c] = full_path
    return this_dir


def next_cache_version(cache_dir: Path, prelim_mkdir=True, version_prefix='version_'):
    if prelim_mkdir and not cache_dir.is_dir():
        cache_dir.mkdir(parents=True)
        return 0

    last_version = -1
    for f in cache_dir.glob('*'):
        name = f.stem
        if version_prefix in name:  # File adhering to the name convention
            version = int(name.split('_')[-1])  # Presuming version is the last thing to appear before the extension dot
            last_version = max(last_version, version)

    return last_version + 1


def assert_is_dir(d):
    assert os.path.isdir(d), f"Directory {d} is invalid"


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
def cd(newdir, cleanup=lambda: True):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)
        cleanup()


class DirectoryCrawler(object):

    def __init__(self, directory, ignore_hidden=True):
        if isinstance(directory, (list, tuple)):
            directory = os.path.join(*directory)
        assert os.path.exists(directory), 'Directory "{}" does not exist.'.format(directory)
        self.directory = directory
        self._contents = None
        self.ignore_hidden = ignore_hidden

    def refresh(self):
        self._contents = None

    def listdir(self, refresh=False, sortby=None, end_dirs_with_slash=True):
        if isinstance(sortby, str) and sortby.startswith('-'):
            sortby = sortby[1:]
            reverse = True
        else:
            reverse = False
        if refresh or self._contents is None:
            self._contents = os.listdir(self.directory)
            if sortby == 'mtime':
                self._contents = sorted(self._contents, key=lambda item: os.path.getmtime(self.get_path(item)))
            elif sortby == 'name':
                self._contents = sorted(self._contents)
            elif sortby is not None:
                raise AssertionError('Invalid value for sortby: {}'.format(sortby))
            if reverse:
                self._contents = self._contents[::-1]
            if end_dirs_with_slash:
                self._contents = [c + os.sep if self.isdir(c) else c for c in self._contents]
            if self.ignore_hidden:
                self._contents = [item for item in self._contents if not item.startswith('.')]
        return self._contents

    def isdir(self, item):
        return os.path.isdir(self.get_path(item))

    def __iter__(self):
        for item in self.listdir():
            yield item

    def values(self):
        for item in self:
            yield self[item]

    def items(self):
        for item, full_path in zip(self, self.values()):
            yield item, full_path

    def subdirs(self):
        for item in self:
            full_path = os.path.join(self.directory, item)
            if os.path.isdir(full_path):
                yield full_path

    def get_path(self, item):
        return os.path.join(self.directory, item)

    def __getitem__(self, item):
        full_path = os.path.normpath(os.path.join(self.directory, item))
        if os.path.isdir(full_path):
            return DirectoryCrawler(full_path, ignore_hidden=self.ignore_hidden)
        elif os.path.exists(full_path):
            return full_path
        else:
            raise Exception('No such file or directory: "{}"'.format(full_path))

    def __str__(self):
        return '{}({}, ignore_hidden={})'.format(self.__class__.__name__, self.directory, self.ignore_hidden)


# ---------------------------------------------------------------------------------------------------------------------#
#                                                              Saves to Disk
# ---------------------------------------------------------------------------------------------------------------------#

def pickle_load(fp: Union[Path, str]):
    if not isinstance(fp, Path):
        fp = Path(fp)
    with fp.open(mode='rb') as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            return pickle.load(f, encoding='latin1')


def pickle_dump(fp: Union[Path, str], obj):
    fp = align_file_extension(fp, 'pkl')
    if not isinstance(fp, Path):
        fp = Path(fp)
    fp.parents[0].mkdir(exist_ok=True, parents=True)
    with fp.open(mode='wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return fp


@contextmanager
def temp_dir():
    # with temp_dir() as dirpath:
    dirpath = tempfile.mkdtemp()

    def cleanup():
        shutil.rmtree(dirpath)
        return True

    with cd(dirpath, cleanup):
        yield dirpath


@contextmanager
def temp_file(suffix='', dp=None):
    """ Context for temporary file.

    Will find a free temporary filename upon entering
    and will try to delete the file on leaving, even in case of an exception.

    Parameters
    ----------
    suffix : string
        optional file suffix
    dp : string
        optional directory to save temporary file in
    """

    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=dp)
    tf.close()
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


# ----------------------------------------------------------------------------------------------------------------------
#                                                         Misc
# ----------------------------------------------------------------------------------------------------------------------

def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return f"%3.1f %s" % (num, x)
        num /= 1024.0


def get_unnamed_file_hash(url):
    """
    Hash the url into a random filename, preserving the extension if any.
    :param url: A URL
    :return: A hashed filename based on the url.
    """
    if url is not None:
        _, ext = os.path.splitext(url)
    else:
        import random
        import string
        elements = string.ascii_uppercase + string.digits
        url = ''.join(random.choice(elements) for _ in range(256))
        ext=''

    hasher = hashlib.md5()
    hasher.update(url.encode('utf-8'))
    filename = os.path.join('temp', hasher.hexdigest()) + ext
    return filename
