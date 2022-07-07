import re
import sys
import struct
from .phrase import Phrase
from Modules.MemoryArray import MemoryArray
from nltk.stem.snowball import SnowballStemmer

WINDOW_ONE_SIZE = 5
WINDOW_TWO_SIZE = 30

P_THRESHOLD = 0.1
S_THRESHOLD = 0.2

I_THRESHOLD = 1.1
RELATED_THRESHOLD = 1.2

ALPHABET = ["а", "б", "в", "г", "д", "е", "ё", "ж", "з", "и", "й", "к", "л", "м", "н", "о", " ",
            "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я",
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q",
            "r", "s", "t", "u", "v", "w", "x", "y", "z", ".", "1", "2", "3", "4", "5", "6", "7",
            "8", "9", "0"]

stemmer = SnowballStemmer("russian")

class RelatedPhrasesService:
    def __init__(self):
        self.phrases: dict = dict()
        self.documents_names: dict = dict()
        self.documents_count: int = 0
        self.g_matrix_new = None

        self.good_phrases = list()
        self.sorted_goods_indexes = None
        self.related = None
        self.predictions = None
        self.predictions_maps = None

    def __getstate__(self):
        attributes = self.__dict__.copy()
        return attributes

    # Add document to phrases
    def add_document(self, text: str, document: int):
        self.documents_count += 1

        sentences = RelatedPhrasesService.clear_text(text).split('.')
        word_num = 0
        for sentence in sentences:
            words = sentence.strip().split(' ')
            for begin in range(len(words)):
                for end in range(WINDOW_ONE_SIZE):
                    phrase_words = words[begin:begin + end + 1]
                    phrase_words_stemmed = [stemmer.stem(word) for word in phrase_words]
                    phrase_text = ' '.join(phrase_words_stemmed)

                    if len(phrase_text) > 0:
                        if len(self.phrases) == 0 or phrase_text not in self.phrases.keys():
                            self.phrases[phrase_text] = Phrase(phrase_text, document, word_num)
                        else:
                            self.phrases[phrase_text].add(document, word_num)

                    if begin + end + 1 == len(words):
                        break
                word_num += 1

    def add_document_next(self, text: str):
        self.add_document(text,  self.documents_count)

    # Mark good phrases
    def mark_good(self):
        for item in self.phrases.values():
            if item.p / self.documents_count >= P_THRESHOLD and item.s / self.documents_count >= S_THRESHOLD:
                item.status = 'good_phrase'
                item.e = item.p / self.documents_count
                self.good_phrases.append(item)

    # Generate co-occurrence matrix
    def generate_g_matrix(self):
        matrix_new = MemoryArray('g_matrix.bin', [len(self.good_phrases), len(self.good_phrases)], 4)

        # matrix = []
        for i, first_good in enumerate(self.good_phrases):
            buf = []
            for j, second_good in enumerate(self.good_phrases):
                # Documents with a both of goods
                intersection_documents = set(first_good.position_in_documents.keys()
                                             ).intersection(set(second_good.position_in_documents.keys()))
                # Count of intersection
                R = 0
                for document in intersection_documents:
                    for first_pos in first_good.position_in_documents[document]:
                        for second_pos in second_good.position_in_documents[document]:
                            if abs(first_pos - second_pos) <= WINDOW_TWO_SIZE:
                                R += 1
                # Add to matrix
                E = first_good.e * second_good.e
                A = R / self.documents_count
                I = A / E
                matrix_new.set([i, j], struct.pack('>f', float(I)))
                # buf.append(I)

                if I > I_THRESHOLD and i != j:
                    first_good.predicts = True
                    second_good.can_predict.add(i)

            # matrix.append(buf)

        self.g_matrix_new = matrix_new

    # Mark not_match_I
    def mark_bad(self):
        for phrase in self.good_phrases:
            if not phrase.predicts:
                phrase.status = 'bad_phrase'
                phrase.bad_status = 'not_match_I'

    # Del included predictions
    def short_goods(self):
        for good in self.good_phrases:
            short = True
            for predictable in good.can_predict:
                if self.good_phrases[predictable].text.find(good.text) != 0:
                    short = False
                    break

            if short is True:
                good.status = 'bad_phrase'
                good.bad_status = 'incomplete_phrases'

    # Del not good phrases from good_phrases
    def clear_goods(self):
        to_delete = list()
        for i, phrase in enumerate(self.good_phrases):
            if phrase.status != 'good_phrase':
                to_delete.append(i)

        new_good_phrase_count = len(self.good_phrases) - len(to_delete)
        matrix_new = MemoryArray('g_matrix_2.bin', [new_good_phrase_count, new_good_phrase_count], 4)

        true_i = 0
        true_j = 0

        for i in range(self.g_matrix_new.directions[0]):
            if i not in to_delete:
                for j in range(self.g_matrix_new.directions[0]):
                    if j not in to_delete:
                        matrix_new.set([i, j], self.g_matrix_new.get([true_i, true_j]))
                        true_j += 1
                true_i += 1
            else:
                self.good_phrases.pop(i)

        self.g_matrix_new = matrix_new

    # Return text with only letters and points at the end of sentences
    @staticmethod
    def clear_text(text):
        text = text.lower()
        text = text.replace('!', '.')
        text = text.replace('?', '.')
        text = ''.join([letter if letter in ALPHABET else ' ' for letter in text])
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[.]+', '.', text)
        return text

    @staticmethod
    def print_g_matrix(matrix, goods):
        strings = []
        string = ''.join([' ' * 24])
        for g in goods:
            string += "{0:33}".format(str(g.text))
            string += ' '
        string += '\n'
        strings.append(string)

        for i, row in enumerate(matrix):
            string = '{0:22}'.format(goods[i].text)
            string += ' '
            for item in row:
                string += '[{0:22.3f}]'.format(item)
                string += ' '
            string += '\n'
            strings.append(string)
        return strings

    def print_phrases(self, good_only=False):
        strings = []
        for phrase in self.phrases.values():
            if good_only:
                if phrase.status == 'good_phrase':
                    strings.append(str(phrase) + '\n')
            else:
                strings.append(str(phrase) + '\n')

        return strings

    def get_related(self):
        goods_related = list()
        for i, good in enumerate(self.good_phrases):
            related_buf = list()
            for j in range(len(self.good_phrases)):
                I = struct.unpack('>f', self.g_matrix_new.get([j, i]))[0]
                if I >= RELATED_THRESHOLD and i != j:
                    related_buf.append([j, I])

            related_buf.sort(key=lambda x: x[1], reverse=True)

            goods_related.append([_[0] for _ in related_buf])

        self.related = goods_related

    def print_related(self, file_name):
        with open(file_name, 'w') as file:
            for i, prediction in enumerate(self.related):
                string = '{0:20}: '.format(self.good_phrases[i].text)
                buf_prediction = list()
                for p in prediction:
                    buf_prediction.append('{0:20}, \t'.format(self.good_phrases[p].text))
                string += ''.join(buf_prediction) + '\n'
                file.write(string)

    def get_clusters_prediction(self):
        maps = list()
        predictions = list()
        for good_index, related_goods in enumerate(self.related):
            print(good_index)
            # Find relations
            main_related = related_goods
            already_related = list([good_index] + main_related)

            current_predictions = []
            if len(main_related) > 0:
                current_predictions = [[good_index] + list(main_related)]
                for main_r in main_related:
                    related_ = self.related[main_r]
                    add_cluster = False
                    for sub_r in related_:
                        if sub_r not in main_related and sub_r != good_index:
                            RelatedPhrasesService.add_related(current_predictions, self.related, already_related,
                                                              good_index, main_r, 0)
                            current_predictions.append([good_index, main_r] + self.related[main_r])
                            add_cluster = True
                            break

                    if not add_cluster and len(main_related) > 1 and main_r != good_index:
                        current_predictions.append([good_index, main_r])

            # Convert to vectors
            predictions.append([])
            maps.append(list(already_related))
            for prediction in current_predictions:
                buf = [0 for _ in range(len(already_related))]
                for prediction_el in prediction:
                    buf[maps[-1].index(prediction_el)] = 1
                predictions[-1].append(buf)

        self.predictions = predictions
        self.predictions_maps = maps

    @staticmethod
    def add_related(current_predictions: list, global_related: list, already_related: list, good_index: int,
                    related_index: int, deep: int):
        already_related.append(related_index)
        related = global_related[related_index]
        for sub_r in related:
            if sub_r not in already_related:
                current_predictions.append([related_index, sub_r])
                if deep < 500:
                    RelatedPhrasesService.add_related(current_predictions, global_related, already_related, good_index,
                                                      sub_r, deep+1)

    # def get_document_relative_stats(self):
    #     phrases_documents = list()
    #     for good in self.good_phrases:
    #         phrases_documents.append([])
    #         for d in good.documents_s.keys():
    #             phrases_documents[-1].append(d)

    # Generate file with clusters_prediction
    def print_clusters_prediction(self, file_name):
        with open(file_name, 'w') as file:
            for index, good_cluster in enumerate(self.predictions):
                if len(good_cluster) > 0:
                    file.write('-------------------- ' + self.good_phrases[index].text + ' ------------------\n')
                    for phrase_index in self.predictions_maps[index]:
                        file.write('{0:20}\t'.format(self.good_phrases[phrase_index].text))
                    file.write(' \n')

                    for prediction in good_cluster:
                        for prediction_el in prediction:
                            file.write('{0:20}\t'.format(str(prediction_el)))
                        file.write(' \n')

    # Remake index for good_phrases
    def reindex(self):
        for i, good_phrase in enumerate(self.good_phrases):
            if good_phrase.text == 'тонус':
                l = 1000

            documents_index = {}
            documents = list(good_phrase.position_in_documents.keys())
            related = self.related[i]

            for document in documents:
                related_phrase_counts = []
                related_bit_vector = []
                positions_in_document_main = good_phrase.position_in_documents[document]
                for r in related:
                    count = 0
                    bit_vector = [0, 0]
                    if document in list(self.good_phrases[r].position_in_documents.keys()):
                        for pos_in_document_r in self.good_phrases[r].position_in_documents[document]:
                            for pos_in_document_m in positions_in_document_main:
                                if abs(pos_in_document_r - pos_in_document_m) < WINDOW_TWO_SIZE:
                                    count += 1
                        bit_vector[0] = 1

                    break_flag = False

                    for sub_rel in self.related[r]:
                        if document in list(self.good_phrases[sub_rel].position_in_documents.keys()):
                            for pos_in_document_m in positions_in_document_main:
                                for pos_in_document_sr in self.good_phrases[sub_rel].position_in_documents[document]:
                                    if abs(pos_in_document_m - pos_in_document_sr) < WINDOW_TWO_SIZE:
                                        bit_vector[1] = 1
                                        break_flag = True
                                        break
                                if break_flag:
                                    break
                        if break_flag:
                            break

                    related_phrase_counts.append(count)
                    related_bit_vector.append(bit_vector)

                documents_index[document] = [related_phrase_counts, related_bit_vector]

            good_phrase.documents_index = documents_index

    def deep_reindex(self):
        pass

    def print_reindex_results(self, file_name: str):
        with open(file_name, 'w') as file:
            for index, good_phrase in enumerate(self.good_phrases):
                file.write('-------------------- ' + self.good_phrases[index].text + ' ------------------\n')
                file.write('{0:20}\t'.format(' '))
                for phrase_index in self.related[index]:
                    file.write('{0:20}\t'.format(self.good_phrases[phrase_index].text))
                file.write(' \n')

                for document in list(good_phrase.documents_index.keys()):
                    if len(good_phrase.documents_index[document][0]) > 0:
                        file.write('{0:20}\t'.format(str(document)))
                        for phrase_index in range(len(self.related[index])):
                            file.write('{0:20}\t'.format(str(good_phrase.documents_index[document][0][phrase_index]) +
                                                         ' ' +
                                                         str(good_phrase.documents_index[document][1][phrase_index])))
                        file.write(' \n')

    # Generate sorted_goods_indexes array with sorted goods indexes by frequency
    def sort_goods(self):
        sorted_goods = [[i, good] for i, good in enumerate(self.good_phrases)]
        sorted_goods.sort(key=lambda x: x[1].e, reverse=True)

        sorted_indexes = list(range(len(self.good_phrases)))
        for i, goods in enumerate(sorted_goods):
            sorted_indexes[goods[0]] = i

        self.sorted_goods_indexes = sorted_indexes

    # Make search query
    def search(self, query: str):
        # Get possible phrases
        possible_phrases = set()
        sentences = RelatedPhrasesService.clear_text(query).split('.')
        for sentence in sentences:
            words = sentence.strip().split(' ')
            for begin in range(len(words)):
                for end in range(WINDOW_ONE_SIZE):
                    phrase_text = ' '.join(words[begin:begin + end + 1])
                    if len(phrase_text) > 0:
                        possible_phrases.add(phrase_text)

        # Get candidate phrases
        candidate_phrases = []
        print(possible_phrases)
        for possible in possible_phrases:
            good_phrase = self.check_good_phrase(possible)
            if good_phrase != -1:
                candidate_phrases.append([good_phrase, self.sorted_goods_indexes[good_phrase],
                                          self.good_phrases[good_phrase].text])

        candidate_phrases.sort(key=lambda x: x[1], reverse=True)
        print(candidate_phrases)

        # Get Qp list
        Qp = []
        while len(candidate_phrases) > 0:
            Qp.append(candidate_phrases[0])
            candidate_phrases = [phrase for phrase in candidate_phrases if Qp[-1][2].find(phrase[2]) == -1]

        print(Qp)
        # Intersection documents
        current_documents = set(self.good_phrases[Qp[0][0]].documents_index.keys())

        for x_index in range(len(Qp) - 1):
            for y_index in range(x_index, len(Qp)):
                # x_index related with y_index
                if Qp[y_index][0] in self.related[Qp[x_index][0]]:
                    related_index = self.related[Qp[x_index][0]].index(Qp[y_index][0])
                    to_del = set()
                    for document in current_documents:
                        if self.good_phrases[Qp[x_index][0]].documents_index[document][1][related_index][0] == 0:
                            to_del.add(document)
                    current_documents = current_documents.difference(to_del)

                # x_index and y_index has common related
                elif len(set(self.related[Qp[x_index][0]]).intersection(set(self.related[Qp[y_index][0]]))) != 0:
                    current_documents = current_documents.intersection(
                        set(self.good_phrases[Qp[x_index][0]].documents_index.keys())
                    )
                    current_documents = current_documents.intersection(
                        set(self.good_phrases[Qp[y_index][0]].documents_index.keys())
                    )
                # x_index and y_index don't related
                else:
                    current_documents = current_documents.intersection(
                        set(self.good_phrases[Qp[x_index][0]].documents_index.keys())
                    )
                    current_documents = current_documents.intersection(
                        set(self.good_phrases[Qp[y_index][0]].documents_index.keys())
                    )
        result = list(current_documents)

        ranks = self.ranking_a(Qp, result)
        ranks_results = []
        for i in range(len(result)):
            ranks_results.append([result[i], ranks[i]])

        ranks_results.sort(key=lambda x: x[1], reverse=True)
        return ranks_results

    # Return index of good phrase by text, or -1 if phrase not good
    def check_good_phrase(self, phrase_text):
        for i, phrase in enumerate(self.good_phrases):
            if phrase.text == phrase_text:
                return i
        return -1

    def ranking_a(self, Qp, results):
        results_score = [0 for _ in results]
        phrases = [_[0] for _ in Qp]

        for i, result in enumerate(results):
            for phrase in phrases:
                index_bin_vector = []
                for document_index in self.good_phrases[phrase].documents_index[result][1]:
                    index_bin_vector += document_index
                results_score[i] += self.bin_to_num_simple(index_bin_vector)

        return results_score

    # Return bin_num_vector in 10 number system
    @staticmethod
    def bin_num_to_num(bin_num_vector: list) -> int:
        result = 0
        for i, x in enumerate(reversed(bin_num_vector)):
            result += pow(2, i) * x
        return result

    # Return rating by second ranking system
    @staticmethod
    def bin_to_num_simple(bin_num_vector: list) -> int:
        result = 0
        for i, x in enumerate(reversed(bin_num_vector)):
            result += i * x
        return result
