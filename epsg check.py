import rasterio 
clipped_img_path = r"/Users/anant/Desktop/internship mynzo /2020images/merged2020.tif"
with rasterio.open(clipped_img_path) as src:
    img = src.read(1)
    meta =src.meta
    print (img)
    print (img.shape)