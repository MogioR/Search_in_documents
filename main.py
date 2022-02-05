import os
from Modules.RelatedPhrasesService import RelatedPhrasesService

service = RelatedPhrasesService()

# text1 = 'Ремонт комода и направляющего сделан быстро, качественно и аккуратно. Очень понравился Александр. ' \
#         'Он- хороший сотрудник, добросовестный и ответственный.'
#
# text2 = 'В комоде нужно было заменить направляющие. По моему мнению, тут даже лишних слов не надо: быстро, ' \
#         'своевременно, аккуратно, качественно, без огрехов, недорого и красиво.'
#
# service.add_document(text1, 0)
# service.add_document(text2, 1)

# получим объект файла
file1 = open("Data/orders.txt", "r", encoding='utf-8')
lines = file1.readlines()
file1.close()

for i, line in enumerate(lines):
    service.add_document(line.strip(), i)

service.mark_good()
service.generate_g_matrix()
phrases_strings = service.print_g_matrix(service.g_matrix, service.good_phrases)
with open('Reports/global.txt', 'w') as file:
    file.write('----------------Global------------------')
    file.write('\n')
    file.writelines(phrases_strings)

service.mark_bad()
phrases_strings = service.print_phrases(False)
with open('Reports/phrases_all.txt', 'w') as file:
    file.writelines(phrases_strings)

phrases_strings = service.print_phrases(True)
with open('Reports/phrases_good.txt', 'w') as file:
    file.writelines(phrases_strings)

service.get_related()
service.print_related('Reports/can_predict.txt')

service.get_clusters_prediction()
service.print_clusters_prediction('Reports/clusters_predict.txt')

service.reindex()
service.print_reindex_results('Reports/reindex.txt')

service.sort_goods()
print(service.search('массаж при тонус мышц'))
