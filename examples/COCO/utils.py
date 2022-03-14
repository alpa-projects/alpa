# import pytz
# from datetime import datetime
import os
import sys
import shutil
import subprocess
import urllib.request as urllib
import numpy as np
import tqdm
import multiprocessing
import threading
import tensorflow as tf
from functools import reduce
import colorsys
import random

class ThreadWithReturnValue(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        if sys.version_info >= (3, 0):
            threading.Thread.join(self, *args)
        else:
            threading.Thread.join(self)

        return self._return


def parallize_v2(f, args, desc='threading'):
    threads = multiprocessing.cpu_count()
    print(f"number of threads: {threads}")
    # threads = 1
    return parallize(f, args, threads=threads, desc=desc)


def parallize(f, args, threads=None, desc='threading'):
    """
        Args:
            - f: function
            - args: list or list(tuple), list when threads not None
    """
    def parse_arg(arg):
        if type(arg) == list or type(arg) == set or type(arg) == tuple:
            return tuple(arg)
        elif type(arg) == dict:
            return arg
        else:
            return (arg, )

    if threads is not None:

        results = []
        for i in tqdm.trange(0, len(args), threads, desc=desc):
            func_args = [parse_arg(arg) for arg in args[i:i + threads]]
            if len(func_args) == 0:
                continue

            if type(func_args[0]) == dict:
                active_threads = [ThreadWithReturnValue(
                    target=f, kwargs=arg) for arg in func_args]
            else:
                active_threads = [ThreadWithReturnValue(
                    target=f, args=arg) for arg in func_args]
            [thread.start() for thread in active_threads]
            results += [thread.join() for thread in active_threads]
        return results
    else:
        if len(args) == 0:
            return []

        if type(args[0]) == dict:
            active_threads = [ThreadWithReturnValue(
                target=f, kwargs=arg) for arg in args]
        else:
            args = [parse_arg(arg) for arg in args]
            active_threads = [ThreadWithReturnValue(
                target=f, args=arg) for arg in args]

        [thread.start() for thread in active_threads]
        return [thread.join() for thread in active_threads]


def parallize_v3(f, args, n_processes=None, desc='parallize_v3'):
    if n_processes == None:
        n_processes = multiprocessing.cpu_count()

    with multiprocessing.Pool(n_processes) as pool:
        results = [r for r in tqdm.tqdm(pool.imap(f, args), desc=desc, total=len(args))]

    return results


def get_colors(N, shuffle=False, bright=True):
    """
    https://github.com/pedropro/TACO/blob/master/detector/visualize.py

    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    colors = [[0, 0, 0]]
    if N == 1:
        return colors

    brightness = 1.0 if bright else 0.7
    hsv = [(i / (N - 1), 1, brightness) for i in range(N - 1)]
    colors.extend(list(map(lambda c: list(map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*c))), hsv)))

    if shuffle:
        random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image


class LOG:
    def info(self, s):
        print("[info]: "+s)
    def debug(self, s):
        print("[debug]: "+s)
    def warning(self, s):
        print("[warning]: "+s)

logger = LOG()

class ExtractException(Exception):
    pass

def tf_version_gt_eq(version):
    # checks if the tensorflow version is greater or equal to `version`
    for r_part, part in zip(map(int, version.split(".")), map(int, tf.__version__.split("."))):
        if r_part > part:
            return False
    return True


def run(cmd, block=True):
    logger.debug(cmd)
    output = b''
    if not block:
        p = subprocess.Popen(cmd, shell=True)
        stdout, stderr = p.communicate()
    else:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)

        stdout, stderr = process.stdout, process.stderr

    if stdout:
        for line in iter(stdout.readline, ""):
            line = line.replace(b'\r', b'').replace(b'\n', b'').strip()
            if line == b"":
                break
            output += line + b'\n'
            logger.debug("%s" % str(line, 'utf-8'))
    if stderr:
        for line in iter(stderr.readline, ""):
            line = line.replace(b'\r', b'').replace(b'\n', b'').strip()
            if line == b"":
                break
            output += line + b'\n'
            logger.debug("%s" % str(line, 'utf-8'))

    return str(output, encoding='utf-8')


def kill_start_tensorboard(logdir, port=6006):

    def target():
        try:
            kill(port)
        except Exception as e:
            logger.warning("could not kill tensorboard, %s" % str(e))
        run(['tensorboard', '--logdir=%s' % logdir, '--port=%d' % port])

    t = threading.Thread(target=target)
    t.start()
    return t


def kill(port: int):
    try:
        pid = str(subprocess.check_output(['lsof', '-t', '-i:%d' % port]), 'utf-8')[:-1]
    except subprocess.CalledProcessError:
        logger.warn("no process on port %d" % port)
        return

    if pid:
        pid = int(pid)

    logger.info("kill process with pid %d on port %d" % (pid, port))
    cmd = ['kill', '%d' % pid]
    return call_for_ret_code(cmd)


def get_files(directory, extensions=None):
    files = []
    extensions = [ext.lower() for ext in extensions] if extensions else None
    for root, _, filenames in sorted(os.walk(directory)):
        if extensions:
            files += [os.path.join(root, name) for name in sorted(filenames)
                      if any(name.lower().endswith("." + ext) for ext in extensions)]
        else:
            files += list(map(lambda filename: os.path.join(root,
                                                            filename), filenames))

    return sorted(files)#[:len(files)//10]


def extract(archive, destination, silent=False, remove_archive_on_success=True):
    import zipfile
    import gzip

    if os.path.exists(archive):
        if not silent:
            logger.info("[*] extracting " + archive + " to " + destination)
        ret = 0
        if archive.endswith(".zip"):
            ret = extract_zip(archive, destination)

        elif archive.endswith(".rar"):
            ret = unrar(archive, destination)
            if ret != 0:
                raise ExtractException(
                    "[*] could not extract rar, please install unrar")
        elif archive.endswith(".gz") and not archive.endswith("tar.gz"):
            with gzip.open(archive, 'rb') as f_in, open(os.path.join(destination, os.path.basename(archive)[:-3]), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            ret = extract_tar(archive, destination, silent=silent)

        if ret != 0:
            raise Exception("[*] could not extract tar %s" % archive)

        if not silent:
            logger.info("[*] removing archive %s" % archive)

        if remove_archive_on_success:
            os.remove(archive)
    else:
        logger.info("[*] already extracted")


def extract_tar(archive, destination, silent=False):
    import tarfile
    try:
        tarfile.TarFile(archive).extractall(destination)
        return 0

    except tarfile.ReadError:
        if not silent:
            logger.info(
                "[*] could extract tar %s via python, trying system call" % archive)

    return call_for_ret_code(["tar", "xfv", archive, "-C", destination], silent=silent)


def extract_zip(archive, destination, silent=False):
    import zipfile
    try:
        zipfile.ZipFile(archive).extractall(destination)
        return 0
    except zipfile.BadZipFile:
        if not silent:
            logger.info("[*] could not extract zip %s via python, trying system call" % archive)

    return call_for_ret_code(['unzip', archive, '-d', destination], silent=silent)


def unrar(archive, destination):

    if call_for_ret_code(["unrar"]) < 0:
        raise ExtractException("unrar not found. Please install unrar.")

    args = ["unrar", "x", archive, "-o", destination]
    return call_for_ret_code(args, silent=False)


def call_for_ret_code(args, silent=False):
    """
        Calls the subprocess and returns the return code
        :param args: list, arguments to fed into subprocess.call
        :param silent: bool, whether to display the ouput of the call
                       in stdout
        :returns int, 1 for failure and 0 for success, -1 for not found
    """
    if not silent:
        print("[+] " + reduce(lambda x, y: str(x) + " " + str(y), args))
    try:
        if silent:
            return subprocess.call(args, stdout=open(os.devnull, 'w'),
                                   stderr=open(os.devnull, 'w'))
        else:
            return subprocess.call(args)
    except IOError:
        return -1
    except OSError:
        return -1


def download_file(url, destination_dir=None, file_name=None, silent=False, auth=None):
    """url lib downloader
    Downloads content of url to destination dir with or without a given file name
    Supports basic authentication and report hooking

    Args:
        url: str, download path
        destination_dir: str, optional if specified file will be downloaded to destination
        file_name : str, optional: file name of the downloaded data
        auth: dict, user authentication {"username": your_user_name, "password": your_password}

    Returns:
        str: file path or None
    """
    os.makedirs(destination_dir, exist_ok=True)

    if not file_name:
        file_name = url.split('/')[-1]

    if not destination_dir:
        destination = file_name
    else:
        destination = os.path.join(destination_dir, file_name)

    if os.path.exists(destination):
        return destination

    if not silent:
        print('[*] downloading ' + url + " to " + destination)

    if auth and "username" in auth and "password" in auth:
        pass_manager = urllib.HTTPPasswordMgrWithDefaultRealm()
        pass_manager.add_password(
            None, url, auth["username"], auth["password"])
        urllib.install_opener(urllib.build_opener(
            urllib.HTTPBasicAuthHandler(pass_manager)))

    try:
        response = urllib.urlopen(url)
        _chunk_read(response, destination,
                    reporthook=None if silent else _download_reporthook)
    except urllib.URLError as error:
        if not silent:
            print("[*] error downloading", url, ":", error.reason)
        return None

    return destination


def _download_reporthook(bytes_count, block_size, total_size):
    if (bytes_count // block_size) % 10 == 0:
        sys.stdout.write('[*] downloaded %02.02f/%02.02f MB \r' % (
            bytes_count / 1000.0 / 1000.0,
            total_size / 1000.0 / 1000.0))
        sys.stdout.flush()
    if bytes_count >= total_size and total_size > 0.0:
        sys.stdout.write('[*] download finished \n')
        sys.stdout.flush()


def _chunk_read(response, filename, chunk_size=8192, reporthook=None):
    try:
        content_length = response.info().getheader('Content-Length')
    except AttributeError:
        content_length = response.headers['Content-Length']  # python3 fix
    if content_length is not None:
        total_size = content_length.strip()
        total_size = int(total_size)
    else:
        total_size = -1
    bytes_so_far = 0

    with open(filename, 'wb') as handler:
        with tqdm.tqdm(total=total_size, unit='B', mininterval=1, unit_scale=True) as tq:
            while 1:
                chunk = response.read(chunk_size)
                handler.write(chunk)
                bytes_so_far += len(chunk)
                if not chunk:
                    break

                tq.update(len(chunk))

                # if reporthook:
                #    reporthook(bytes_so_far, chunk_size, total_size)

    return bytes_so_far


def download_from_google_drive(drive_id, destination_dir, filename, block_size=32 * 1024, silent=False):
    import requests
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': drive_id}, stream=True)
    logger.debug("download from google drive response: %d" % response.status_code)
    token = _google_drive_get_confirm_token(response)
    logger.debug("download from google drive token: %s" % token)
    if token:
        params = {'id': drive_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    destination = os.path.join(destination_dir, filename)
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, "wb") as f:
        for count, chunk in enumerate(response.iter_content(block_size)):
            if not silent:
                _download_reporthook(count * block_size,
                                     block_size, total_size)
            if chunk:
                f.write(chunk)
    return destination


def _google_drive_get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def download_and_extract(url, destination_dir, chk_exists=True, overwrite=False, silent=False, auth=None, file_name=None, cache_dir='/tmp/extracted', remove_archive_on_success=True):
    """ Download and extract rar, zip and tar

    Args:
        url: str, download dataset url or {name, drive_id} for google drive download
        destination_dir: str, directory path to extract the dataset to
        chk_exists: bool, check whether destination exists and download and extract if it does not
        silent: bool, do not print anything during downloading if true
        file_name: file name of the downloaded file (f.e. filename too long)
        auth: dict, user authentification {"username": your_user_name, "password": your_password}
        remove_archive_on_success: bool, remove archive when the extraction was successful

    Returns:
        str: folder where the dataset is extracted to
    """
    if not os.path.exists(destination_dir) or not chk_exists or overwrite:

        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(destination_dir, exist_ok=True)

        if type(url) == str:
            archive = download_file(
                url, destination_dir=cache_dir, file_name=file_name, silent=silent, auth=auth)
        else:
            logger.info("downloading from google drive...")
            name, drive_id = url
            archive = download_from_google_drive(
                drive_id=drive_id, destination_dir=cache_dir, filename=name, silent=silent)

        extract(archive, destination_dir, remove_archive_on_success=remove_archive_on_success)

    return destination_dir


def get_gpu_stats():
    cmd = ["nvidia-smi", 
        "--query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used",
        "--format=csv"]
    headers = "timestamp name pci.bus_id driver_version pstate pcie.link.gen.max pcie.link.gen.current temperature.gpu utilization.gpu utilization.memory memory.total memory.free memory.used".split(" ")
    outputs = str(subprocess.check_output(cmd), 'utf-8')
    values = outputs.split("\n")[-2].split(",")
    values = list(map(lambda x: x.strip(), values))
    return dict(zip(headers, values))

class DataType:
    TRAIN, TEST, VAL = 'train', 'test', 'val'

    @staticmethod
    def get():
        return list(map(lambda x: DataType.__dict__[x], list(filter(lambda k: not k.startswith("__") and type(DataType.__dict__[k]) == str, DataType.__dict__))))


########################################
##### Other Utilities
########################################
from jax.tree_util import tree_map, tree_flatten, PyTreeDef
GB = 1 << 30  # Gigabyte
MB = 1 << 20  # Megabyte

def map_to_shape(array_pytree: PyTreeDef):
    """Map a PyTree of jax arrays to their shapes."""
    return tree_map(lambda x: getattr(x, "shape", None), array_pytree)


def compute_bytes(pytree: PyTreeDef):
    """Compute the total bytes of arrays in a pytree."""
    flatten_args, _ = tree_flatten(pytree)
    ret = 0
    for x in flatten_args:
        if hasattr(x, "shape"):
            ret += np.prod(x.shape) * x.dtype.itemsize
    return ret


def compute_param_number(pytree: PyTreeDef):
    """Compute the total number of elements in a pytree."""
    flatten_args, _ = tree_flatten(pytree)
    ret = 0
    for x in flatten_args:
        if hasattr(x, "shape"):
            ret += np.prod(x.shape)
    return ret

if __name__ == "__main__":
    from pprint import pprint
    pprint(get_gpu_stats())

