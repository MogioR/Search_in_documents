import os
import sys
import gc
import json
import urllib3
import pickle

urllib3.disable_warnings()
API_PATH = os.path.dirname(os.path.realpath(__file__)) + '/Modules/local_modules'
print(API_PATH)
sys.path.append(str(API_PATH))

from Modules.RelatedPhrasesService import RelatedPhrasesService
from Modules.RelatedPhrasesService import RelatedPhrasesServiceNew
from Modules.local_modules.serp.api_xmlriver_v2 import BaseSerp, get_data, serp_parser
from Modules.url_to_text import url_to_text

from tqdm import tqdm
from multiprocessing import Pool
from nltk.stem.snowball import SnowballStemmer

MAX_URLS = 100000000
URLS_PROCESSED = 0
NUM_THREADS = 8

if __name__ == '__main__':
    # # Load serps
    # base = BaseSerp()
    # base.load()
    #
    # # Get urls
    # urls_documents = dict()
    # for key, query in base.base_query.items():
    #     if key.lower().find('ресниц') != -1:
    #         for serp in query:
    #             for i, url_item in enumerate(serp.urls):
    #                 document_name = serp.query + '_' + serp.location + '_' + str(i)
    #                 if url_item.url in urls_documents.keys():
    #                     urls_documents[url_item.url].append(document_name)
    #                 else:
    #                     urls_documents[url_item.url] = [document_name]
    #                     URLS_PROCESSED += 1
    #
    #                 if URLS_PROCESSED >= MAX_URLS:
    #                     break
    #             if URLS_PROCESSED >= MAX_URLS:
    #                 break
    #         if URLS_PROCESSED >= MAX_URLS:
    #             break
    #
    # del base
    # gc.collect()
    # # Get sites
    # service = RelatedPhrasesServiceNew()
    # pool = Pool(NUM_THREADS)
    # results = []
    # for text in tqdm(pool.imap(url_to_text, list(urls_documents.keys())), total=len(urls_documents.keys())):
    #     if len(service.clear_text(text).strip()) > 0 and len(service.clear_text(text)) >= len(text)/2:
    #         results.append(text + '\n')
    #     else:
    #         results.append('')
    # print(len([r for r in results if r != '']))
    #
    # with open('Reports/results_download.json', 'w', encoding='utf-8') as file:
    #     json.dump(results, file, ensure_ascii=False, indent=4)
    #
    # with open('Reports/urls_documents.json', 'w', encoding='utf-8') as file:
    #     json.dump(urls_documents, file, ensure_ascii=False, indent=4)

    with open('Reports/urls_documents.json', 'r', encoding='utf-8') as file:
        urls_documents = json.load(file)

    with open('Reports/results_download.json', 'r', encoding='utf-8') as file:
        results = json.load(file)

    sizes = []
    for line in results:
        sizes.append(len(str(line)))

    lol = 0

    # with open('Reports/results_download_3.txt', 'w', encoding='utf-8') as file:
    #     for line in results:
    #         file.write(line)

    # values = list(urls_documents.values())
    # for i, result in enumerate(tqdm(results)):
    #     if len(result) > 0:
    #         service.add_document(result, values[i])

    # service.set_phrases_bank(0, safe=False)
    # service.phrases_banks_count = 14
    # service.set_good_phrases_bank(0, safe=False)
    # service.good_phrases_banks_count = 1
    # service.count_of_documents = 500

    service.sum_phrase_banks()
    print('Mark good')
    service.mark_good()
    # service.set_good_phrases_bank(0)
    # service.good_phrases_count = len(service.current_good_phrase_bank)
    print('Generate g')
    service.generate_g_matrix()

    print(service.good_phrases_banks_count)

    # print('Mark bad')
    # service.mark_bad()
    #
    # service.save()
    #
    # print('Get_related')
    # service.get_related()
    #
    # print('Get reindex')
    # service.reindex()
    #
    # print('Sort goods')
    # service.sort_goods()
    #
    # service.save()
#############################################################
    # service_2 = RelatedPhrasesServiceNew()
    # service_2.load()
    #
    # stemmer = SnowballStemmer("russian")
    #
    # query = 'Штукатурные работы Минск'
    # results = service_2.search(' '.join([stemmer.stem(word) for word in query.lower().split()]))
    # print('Запрос: ', query)
    # for result in results:
    #     print(service_2.documents_names[result[0]], 'счет: ', result[1])
    #
    # lol = 1

    # service.mark_good()
    # phrases_strings = service.print_phrases(True)
    # with open('Reports/phrases_good.txt', 'w') as file:
    #     file.writelines(phrases_strings)
    #
    # # with open('Backups/phrases_good.json', 'w') as file:
    # #     json.dump(service.good_phrases, file)
    #
    # print('G matrix')
    # service.generate_g_matrix()
    # # phrases_strings = service.print_g_matrix(service.g_matrix_new, service.good_phrases)
    # # with open('Reports/global.txt', 'w') as file:
    # #     file.write('----------------Global------------------')
    # #     file.write('\n')
    # #     file.writelines(phrases_strings)
    #
    # service.mark_bad()
    # phrases_strings = service.print_phrases(False)
    # with open('Reports/phrases_all.txt', 'w') as file:
    #     file.writelines(phrases_strings)
    #
    # phrases_strings = service.print_phrases(True)
    # with open('Reports/phrases_good.txt', 'w') as file:
    #     file.writelines(phrases_strings)
    #
    # del phrases_strings
    #
    # print('Get_related')
    # service.get_related()
    # service.print_related('Reports/can_predict.txt')
    #
    # # print('Get cluster_prediction')
    # # service.get_clusters_prediction()
    # # service.print_clusters_prediction('Reports/clusters_predict.txt')
    # print('Get reindex')
    # service.reindex()
    # service.print_reindex_results('Reports/reindex.txt')
    # print('Sort goods')
    # service.sort_goods()
    #
    # with open('google_cluster.pickle', 'wb') as f:
    #     pickle.dump(service, f)
    #
    # print(service.search('детск массаж'))
