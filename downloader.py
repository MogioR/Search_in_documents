# -*- coding: utf-8 -*-
import json
from Modules.Xml_river.google_xml_river import GoogleXmlRiver
from Modules.url_to_text import url_to_text
CONFIG = {
    "XMLRiver": {
        "xml_river_user": "1660",
        "xml_river_key": "9d9ea875799adf551c8329d0a6dcf50ed168f9b8",
        "group_by": 20,
        "Google": {
            "default_country_id": 2112,
            "default_loc_id": 1001493,
            "default_language_id": "RU",
            "default_device": "desktop",
            "default_use_language": False
        },
        "Yandex": {
            "default_loc_id": 4,
            "default_language_id": "ru",
            "default_device": "desktop",
            "default_use_language": False
        }
    }
}

key = 'массаж детский'
urls = []

# river = GoogleXmlRiver(CONFIG)
#
# river.set_region('MOSCOW')
# results = river.get_query_items_with_params(key, relatives=False, questions=False)
# for site in results['sites']:
#     urls.append(site.url)
# river.set_region('MINSK')
# results = river.get_query_items_with_params(key, relatives=False, questions=False)
# for site in results['sites']:
#     urls.append(site.url)
# river.set_region('EKATERINBURG')
# results = river.get_query_items_with_params(key, relatives=False, questions=False)
# for site in results['sites']:
#     urls.append(site.url)
# river.set_region('SANKT-PETERBURG')
# results = river.get_query_items_with_params(key, relatives=False, questions=False)
# for site in results['sites']:
#     urls.append(site.url)
# river.set_region('VELIKII-NOVGOROD')
# results = river.get_query_items_with_params(key, relatives=False, questions=False)
# for site in results['sites']:
#     urls.append(site.url)
# with open("urls_backup.json", "w") as write_file:
#     json.dump(urls, write_file)


import warnings
warnings.filterwarnings("ignore")


with open('urls_backup.json', 'r') as f:
    data = json.load(f)

url_text = []
with open('Data/detskiy_masaj.txt', 'w') as f:
    for url in data:
        try:
            f.write(url_to_text(url) + '\n')
            f.write(url+ '\n')
        except Exception as e:
            print("Url", url, "Exeption:", e)
