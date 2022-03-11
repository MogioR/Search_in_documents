# -*- coding: utf-8 -*-
import json

with open('Backups/results.json', 'r') as f:
    results = json.load(f)
with open('Data/detskiy_masaj.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    f.close()

pages = []
for i, line in enumerate(lines):
    if i % 2 == 1:
        sep = line.index(':')
        num_in_serp = int(line[0:sep])
        document_id = i-1
        pages.append([document_id, num_in_serp])

print(results['results'])
print(pages)

stats = {}

for page in pages:
    for result in results['results']:
        if result[0] == page[0]:
            if page[1] in stats.keys():
                stats[page[1]][0].append(result[1])
                stats[page[1]][1] += result[1]
            else:
                stats[page[1]] = [[result[1]], result[1]]

for key in sorted(stats):
    print('Позиция в выдаче: {0:2} Средний результат: {1:10.3f} Результаты: {2}'.format(
        key+1, stats[key][1]/len(stats[key][0]), stats[key][0])
    )
    # print('Позиция в выдаче:', key+1, 'Средний результат:', stats[key][1]/len(stats[key][0]), 'Результаты:', stats[key][0])