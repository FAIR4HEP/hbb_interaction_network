import sys
import urllib.request
from pathlib import Path

from tqdm import tqdm

# prepare the dir for raw data
cwd = Path.cwd()
data_raw_dir = cwd.parent.parent.joinpath("data/raw")
data_raw_dir.mkdir(parents=True, exist_ok=True)

# list the online raw files
base_url = "http://opendata.cern.ch/eos/opendata/cms/datascience/HiggsToBBNtupleProducerTool/test"
rawFileUrls = ["{base_url}/ntuple_merged_{i}".format(base_url=base_url, i=i) for i in range(0, 9)]

# prepare target file objects
fileNames = [url.rsplit("/", 1)[-1] for url in rawFileUrls]
targetFiles = [data_raw_dir.joinpath(f) for f in fileNames]


# progress bar class and fn for each file
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


testLimit = 2
counter = 0
# download raw url files to the destinations
for url, f in zip(rawFileUrls, targetFiles):
    download_url(url, f)
    if counter < testLimit:
        counter += 1
        continue
    else:
        sys.exit(1)
