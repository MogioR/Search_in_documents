# -*- coding: utf-8 -*-

import os
import json
from Modules.RelatedPhrasesService import RelatedPhrasesService

service = RelatedPhrasesService()

file1 = open("Data/detskiy_masaj.txt", "r", encoding='utf-8')
lines = file1.readlines()
file1.close()

for i, line in enumerate(lines):
    if i % 2 == 0:
        service.add_document(line.strip(), i)

service.mark_good()
phrases_strings = service.print_phrases(True)
with open('Reports/phrases_good.txt', 'w') as file:
    file.writelines(phrases_strings)

# with open('Backups/phrases_good.json', 'w') as file:
#     json.dump(service.good_phrases, file)


print('G matrix')
service.generate_g_matrix()
# phrases_strings = service.print_g_matrix(service.g_matrix_new, service.good_phrases)
# with open('Reports/global.txt', 'w') as file:
#     file.write('----------------Global------------------')
#     file.write('\n')
#     file.writelines(phrases_strings)

service.mark_bad()
phrases_strings = service.print_phrases(False)
with open('Reports/phrases_all.txt', 'w') as file:
    file.writelines(phrases_strings)

phrases_strings = service.print_phrases(True)
with open('Reports/phrases_good.txt', 'w') as file:
    file.writelines(phrases_strings)

del phrases_strings

print('Get_related')
service.get_related()
service.print_related('Reports/can_predict.txt')

# print('Get cluster_prediction')
# service.get_clusters_prediction()
# service.print_clusters_prediction('Reports/clusters_predict.txt')
print('Get reindex')
service.reindex()
service.print_reindex_results('Reports/reindex.txt')
print('Sort goods')
service.sort_goods()
print(service.search('детск массаж'))
