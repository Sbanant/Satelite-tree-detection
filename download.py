
from datetime import date, timedelta
import os
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape

copernicus_user = os.getenv("fallengod2008@gmail.com") # copernicus User
copernicus_password = os.getenv("$Helloanant123") # copernicus Password
ft = "POLYGON ((82.85970261 18.21578985, 82.74225631 18.21688235, 82.74220912 18.21689718, 82.74216486 18.21691688, 82.74211206 18.21694144, 82.74206663 18.21696348, 82.73417615 18.22155073, 82.677033 18.262006, 82.67396584 18.26461846, 82.67392267 18.26465702, 82.64700294 18.28903947, 82.64323619 18.29348506, 82.64452524 18.30063793, 82.80571201 18.44401298, 82.80598199 18.44406503, 82.80602398 18.44406503, 82.80605902 18.44406202, 82.80820001 18.44350596, 82.80823899 18.44349104, 82.80834803 18.44344896, 82.82010198 18.43670798, 82.92949327 18.37023455, 82.9320438 18.3686015, 82.93208119 18.36857602, 82.99587233 18.31536309, 82.99587644 18.31534163, 82.99593084 18.31504349, 82.99595514 18.31484836, 82.99595573 18.31472766, 82.99594349 18.31469958, 82.85972617 18.21580561, 82.85970261 18.21578985, 82.85970261 18.21578985))" 
data_collection = "SENTINEL-2" # Sentinel satellite

today =  date.today()
today_string = today.strftime("2024-7-15")
yesterday = today - timedelta(days=1)
yesterday_string = yesterday.strftime("2024-7-14")



def get_keycloak(username: str, password: str) -> str:
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    try:
        r = requests.post(
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
            data=data,
        )
        r.raise_for_status()
    except Exception as e:
        raise Exception(
            f"Keycloak token creation failed. Reponse from the server was: {r.json()}"
        )
    return r.json()["access_token"]


json_ = requests.get(
    f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq '{data_collection}' and OData.CSC.Intersects(area=geography'SRID=4326;{ft}') and ContentDate/Start gt {yesterday_string}T00:00:00.000Z and ContentDate/Start lt {today_string}T00:00:00.000Z&$count=True&$top=1000"
).json()  
p = pd.DataFrame.from_dict(json_["value"]) # Fetch available dataset
if p.shape[0] > 0 :
    p["geometry"] = p["GeoFootprint"].apply(shape)
    productDF = gpd.GeoDataFrame(p).set_geometry("geometry") # Convert PD to GPD
    productDF = productDF[~productDF["Name"].str.contains("L1C")] # Remove L1C dataset
    print(f" total L2A tiles found {len(productDF)}")
    productDF["identifier"] = productDF["Name"].str.split(".").str[0]
    allfeat = len(productDF) 

    if allfeat == 0:
        print("No tiles found for today")
    else:
        ## download all tiles from server
        for index,feat in enumerate(productDF.iterfeatures()):
            try:
                session = requests.Session()
                keycloak_token = get_keycloak(copernicus_user,copernicus_password)
                session.headers.update({"Authorization": f"Bearer {keycloak_token}"})
                url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({feat['properties']['Id']})/$value"
                response = session.get(url, allow_redirects=False)
                while response.status_code in (301, 302, 303, 307):
                    url = response.headers["Location"]
                    response = session.get(url, allow_redirects=False)
                print(feat["properties"]["Id"])
                file = session.get(url, verify=False, allow_redirects=True)

                with open(
                    f"{feat['properties']['identifier']}.zip", #location to save zip from copernicus 
                    "wb",
                ) as p:
                    print(feat["properties"]["Name"])
                    p.write(file.content)
            except:
                print("problem with server")
else :
    print('no data found')