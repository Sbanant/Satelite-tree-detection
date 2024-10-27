import os
import rasterio
from rasterio.enums import Resampling

fp_in = '/Users/anant/Desktop/internship mynzo /jp2 images'

bandList = [band for band in os.listdir(fp_in) if band.endswith('.jp2')]

for band in bandList:
    with rasterio.open(fp_in + "/"+band) as src:
         profile = src.profile
         profile.update(
            dtype=rasterio.float32,  # Update data type if needed
            driver='GTiff'
        )
        
         fp_tif = fp_in + band[:-4] + '.tif'
        
         with rasterio.open(fp_tif, 'w', **profile) as dst:
            for i in range(1, src.count + 1):
                band_data = src.read(i, resampling=Resampling.nearest)
                dst.write(band_data, i)