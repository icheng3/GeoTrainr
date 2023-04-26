import os
from selenium import webdriver
from selenium.webdriver.common.by import By
import urllib.request
import pandas as pd
import numpy as np

IMG_SIZE = 256
EU_CODES = ['AL', 'AD', 'AT', 'BY', 'BE', 'BA', 'BG', 'HR',
            'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR',
            'HU', 'IS', 'IE', 'IT', 'XK', 'LV', 'LI', 'LT',
            'LU', 'MT', 'MD', 'MC', 'ME', 'NL', 'MK', 'NO',
            'PL', 'PT', 'RO', 'RU', 'SM', 'RS', 'SK', 'SI',
            'ES', 'SE', 'CH', 'UA', 'GB', 'VA']
API_ENDPOINT = 'https://maps.googleapis.com/maps/api/streetview'
# change this to your API key
API_KEY = "AIzaSyCScWJEy0uXU8KCVOzJ2yW7y3pkz1ZnYF8"
# number of images to be obtained
NUM_IMAGES = 10000

# load data
loc_data = pd.read_csv('country_data.csv')
num_rows = loc_data.shape[0]
random_locs = np.random.choice(num_rows, num_rows, replace=False)

# set this to the path of the driver you download @ https://chromedriver.storage.googleapis.com/index.html
os.environ['PATH'] += r"/Documents/CS1430_Projects/chromedriver_mac_arm64/chromedriver"
driver = webdriver.Chrome()

hits = 0
seen = 0
total = 0

for i in random_locs:
    if hits == NUM_IMAGES:
        break

    row = loc_data.iloc[i]
    loc = f'{row[0]},{row[1]}'
    country_code = row[2]

    if os.path.exists(f'data/{country_code}_{loc}.png'):
        seen += 1
        pass

    metadata_request = f'{API_ENDPOINT}/metadata?size={IMG_SIZE}x{IMG_SIZE}&location={loc}&fov=80&heading=70&pitch=0&key={API_KEY}'
    driver.get(metadata_request)
    
    try:
        text_element = driver.find_element(By.XPATH, "//*[contains(text(), 'ZERO_RESULTS')]")
    except:
        api_request = f'{API_ENDPOINT}?size={IMG_SIZE}x{IMG_SIZE}&location={loc}&fov=80&heading=70&pitch=0&key={API_KEY}'
        driver.get(api_request)
        image_url = driver.find_element(By.TAG_NAME, 'img').get_attribute('src')
        urllib.request.urlretrieve(image_url, f'data/{country_code}_{loc}.png')
        hits += 1
    
    total += 1

print("seen:", seen)
print(f"hit rate: {hits}/{total} =", hits / total)