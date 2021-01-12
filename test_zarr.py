import dask.array as da
from utils import define_path
import matplotlib.pyplot as plt

local = True
path = define_path(local)
z1 = da.from_zarr(path+"80982_image.zarr")

h,bins = da.histogram(z1[:,:,2],6500,(0,65000))
h.compute()

plt.plot(bins[1:],h)
plt.loglog()
plt.show()