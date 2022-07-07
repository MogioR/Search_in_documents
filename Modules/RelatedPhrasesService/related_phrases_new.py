import os
import sys
import re
import gc
import pickle
import struct
import copy

from tqdm import tqdm
from nltk.stem.snowball import SnowballStemmer

from .phrase import Phrase
from Modules.MemoryArray import MemoryArray

WINDOW_ONE_SIZE = 5
WINDOW_TWO_SIZE = 30

P_THRESHOLD = 0.1
S_THRESHOLD = 0.2

I_THRESHOLD = 1.1
RELATED_THRESHOLD = 1.2

MAX_PHRASES_IN_BANK = 1000000
ROOT_TO_SAVE = 'processed_data/'

ALPHABET = ["а", "б", "в", "г", "д", "е", "ё", "ж", "з", "и", "й", "к", "л", "м", "н", "о", " ",
            "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я",
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q",
            "r", "s", "t", "u", "v", "w", "x", "y", "z", ".", "1", "2", "3", "4", "5", "6", "7",
            "8", "9", "0"]

stemmer = SnowballStemmer("russian")
print(stemmer.stem("не"))

class RelatedPhrasesServiceNew:
    def __init__(self):
        # self.phrases: dict = dict()
        self.documents_names: dict = dict()
        self.count_of_documents: int = 0

        self.phrases_banks_count: int = 1
        self.current_phrase_bank_id: int = 0
        self.current_phrase_bank: dict = dict()

        self.good_phrases_banks_count: int = 1
        self.current_good_phrase_bank_id: int = 0
        self.current_good_phrase_bank: list = list()
        self.good_phrases_count: int = 0

        self.g_matrix = None

        self.sorted_goods_indexes = None
        self.related = None
        # self.predictions = None
        # self.predictions_maps = None

    def set_phrases_bank(self, bank_id: int, safe=True):
        root_ = ROOT_TO_SAVE + 'phrase_banks/'

        if safe:
            with open(root_ + 'phrase_bank_' + str(self.current_phrase_bank_id) + '.pickle', 'wb') as f:
                pickle.dump(self.current_phrase_bank, f)

            gc.collect()

        with open(root_ + 'phrase_bank_' + str(bank_id) + '.pickle', 'rb') as f:
            del self.current_phrase_bank
            self.current_phrase_bank = pickle.load(f)

        gc.collect()
        self.current_phrase_bank_id = bank_id

    def add_phrases_bank(self):
        root_ = ROOT_TO_SAVE + 'phrase_banks/'

        with open(root_ + 'phrase_bank_' + str(self.current_phrase_bank_id) + '.pickle', 'wb') as f:
            pickle.dump(self.current_phrase_bank, f)

        self.current_phrase_bank_id = self.phrases_banks_count
        self.phrases_banks_count += 1
        self.current_phrase_bank = dict()
        gc.collect()

    def set_good_phrases_bank(self, bank_id: int, safe=True):
        root_ = ROOT_TO_SAVE + 'good_phrase_banks/'

        if safe:
            with open(root_ + 'good_phrase_bank_' + str(self.current_good_phrase_bank_id) + '.pickle', 'wb') as f:
                pickle.dump(self.current_good_phrase_bank, f)

            gc.collect()

        with open(root_ + 'good_phrase_bank_' + str(bank_id) + '.pickle', 'rb') as f:
            self.current_good_phrase_bank = pickle.load(f)
        self.current_good_phrase_bank_id = bank_id

    def add_good_phrases_bank(self):
        root_ = ROOT_TO_SAVE + 'good_phrase_banks/'

        with open(root_ + 'good_phrase_bank' + str(self.current_good_phrase_bank_id) + '.pickle', 'wb') as f:
            pickle.dump(self.current_good_phrase_bank, f)

        self.current_good_phrase_bank_id = self.good_phrases_banks_count
        self.good_phrases_banks_count += 1
        self.current_good_phrase_bank = list()
        gc.collect()

    def add_document(self, text: str, document_names: list):
        """
        Add document to phrases
        :param text: text of document
        :param document_names: names of document
        """

        document_id = len(self.documents_names)
        self.documents_names[document_id] = document_names
        self.count_of_documents += 1
        sentences = self.clear_text(text).split('.')
        word_num = 0
        for sentence in sentences:
            words = [stemmer.stem(word) for word in sentence.strip().split(' ')]
            for begin in range(len(words)):
                for end in range(WINDOW_ONE_SIZE):
                    phrase_words = words[begin:begin + end + 1]
                    phrase_text = ' '.join(phrase_words)

                    if len(phrase_text) > 0:
                        if len(self.current_phrase_bank) == 0 or phrase_text not in self.current_phrase_bank.keys():
                            self.current_phrase_bank[phrase_text] = Phrase(phrase_text, document_id, word_num)
                        else:
                            self.current_phrase_bank[phrase_text].add(document_id, word_num)

                    if len(self.current_phrase_bank) >= MAX_PHRASES_IN_BANK:
                        self.add_phrases_bank()

                    if begin + end + 1 == len(words):
                        break
                word_num += 1

    def sum_phrase_banks(self):
        for i in range(0, self.phrases_banks_count-1):
            self.set_phrases_bank(i)
            main_bank = copy.deepcopy(self.current_phrase_bank)
            for j in range(i+1, self.phrases_banks_count):
                self.set_phrases_bank(j)
                duplicate_keys = set(main_bank.keys()).intersection(set(self.current_phrase_bank))
                for key in duplicate_keys:
                    main_bank[key] += self.current_phrase_bank[key]
                    del self.current_phrase_bank[key]
                    lol = 1

    # Mark good phrases
    def mark_good(self):
        for i in tqdm(range(0, self.phrases_banks_count)):
            self.set_phrases_bank(i)
            for item in self.current_phrase_bank.values():
                if (item.p / self.count_of_documents >= P_THRESHOLD and
                        item.s / self.count_of_documents >= S_THRESHOLD):
                    item.status = 'good_phrase'
                    item.e = item.p / self.count_of_documents
                    self.current_good_phrase_bank.append(item)

                    if len(self.current_good_phrase_bank) >= MAX_PHRASES_IN_BANK:
                        self.good_phrases_count += len(self.current_good_phrase_bank)
                        self.add_good_phrases_bank()
        self.good_phrases_count += len(self.current_good_phrase_bank)
        print('Good phrases: ', self.good_phrases_count)

    # Generate co-occurrence matrix
    def generate_g_matrix(self):
        matrix_new = MemoryArray('g_matrix.bin', [self.good_phrases_count, self.good_phrases_count], 4)
        matrix_new.create()

        shift_i = 0
        shift_j = 0

        for i_bank in range(0, self.good_phrases_banks_count):
            self.set_good_phrases_bank(i_bank)
            main_bank = copy.deepcopy(self.current_good_phrase_bank)
            for j_bank in range(0, self.good_phrases_banks_count):
                self.set_good_phrases_bank(j_bank)
                for i, first_good in enumerate(main_bank):
                    for j, second_good in enumerate(self.current_good_phrase_bank):
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
                        A = R / self.count_of_documents
                        I = A / E
                        matrix_new.set([shift_i+i, shift_j+j], struct.pack('>f', float(I)))
                        # buf.append(I)

                        if I > I_THRESHOLD and i != j:
                            first_good.predicts = True
                            second_good.can_predict.add(i)

                shift_i += len(main_bank)
                shift_j += len(self.current_good_phrase_bank)

        self.g_matrix = matrix_new

        # # matrix = []
        # for i, first_good in enumerate(self.good_phrases):
        #     buf = []
        #     for j, second_good in enumerate(self.good_phrases):
        #         # Documents with a both of goods
        #         intersection_documents = set(first_good.position_in_documents.keys()
        #                                      ).intersection(set(second_good.position_in_documents.keys()))
        #         # Count of intersection
        #         R = 0
        #         for document in intersection_documents:
        #             for first_pos in first_good.position_in_documents[document]:
        #                 for second_pos in second_good.position_in_documents[document]:
        #                     if abs(first_pos - second_pos) <= WINDOW_TWO_SIZE:
        #                         R += 1
        #         # Add to matrix
        #         E = first_good.e * second_good.e
        #         A = R / len(self.documents_names.keys())
        #         I = A / E
        #         matrix_new.set([i, j], struct.pack('>f', float(I)))
        #         # buf.append(I)
        #
        #         if I > I_THRESHOLD and i != j:
        #             first_good.predicts = True
        #             second_good.can_predict.add(i)
        #
        #     # matrix.append(buf)
        #
        # self.g_matrix = matrix_new

    # Mark not_match_I
    def mark_bad(self):
        for phrase in self.good_phrases:
            if not phrase.predicts:
                phrase.status = 'bad_phrase'
                phrase.bad_status = 'not_match_I'

    # Del not good phrases from good_phrases
    def clear_goods(self):
        to_delete = list()
        for i, phrase in enumerate(self.good_phrases):
            if phrase.status != 'good_phrase':
                to_delete.append(i)

        new_good_phrase_count = len(self.good_phrases) - len(to_delete)
        matrix_new = MemoryArray('g_matrix_2.bin', [new_good_phrase_count, new_good_phrase_count], 4)
        matrix_new.create()

        true_i = 0
        true_j = 0

        for i in range(self.g_matrix.directions[0]):
            if i not in to_delete:
                for j in range(self.g_matrix.directions[0]):
                    if j not in to_delete:
                        matrix_new.set([i, j], self.g_matrix.get([true_i, true_j]))
                        true_j += 1
                true_i += 1
            else:
                self.good_phrases.pop(i)

        self.g_matrix = matrix_new

    def get_related(self):
        goods_related = list()
        for i, good in enumerate(self.good_phrases):
            related_buf = list()
            for j in range(len(self.good_phrases)):
                I = struct.unpack('>f', self.g_matrix.get([j, i]))[0]
                if I >= RELATED_THRESHOLD and i != j:
                    related_buf.append([j, I])

            related_buf.sort(key=lambda x: x[1], reverse=True)

            goods_related.append([_[0] for _ in related_buf])

        self.related = goods_related

    # Remake index for good_phrases
    def reindex(self):
        for i, good_phrase in enumerate(self.good_phrases):
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
        sentences = self.clear_text(query).split('.')
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

    def save(self, root=None):
        root_ = ROOT_TO_SAVE
        if root:
            root_ = root

        with open(root_ + 'related_phrases_phrases.pickle', 'wb') as f:
            pickle.dump(self.phrases, f)
        with open(root_ + 'related_phrases_good_phrases.pickle', 'wb') as f:
            pickle.dump(self.good_phrases, f)
        with open(root_ + 'related_phrases_document_names.pickle', 'wb') as f:
            pickle.dump(self.documents_names, f)
        with open(root_ + 'related_phrases_related.pickle', 'wb') as f:
            pickle.dump(self.related, f)
        with open(root_ + 'related_phrases_sorted_goods_indexes.pickle', 'wb') as f:
            pickle.dump(self.sorted_goods_indexes, f)

    def load(self, root=None):
        root_ = ROOT_TO_SAVE
        if root:
            root_ = root

        with open(root_ + 'related_phrases_phrases.pickle', 'rb') as f:
            self.phrases = pickle.load(f)
        with open(root_ + 'related_phrases_good_phrases.pickle', 'rb') as f:
            self.good_phrases = pickle.load(f)
        with open(root_ + 'related_phrases_document_names.pickle', 'rb') as f:
            self.documents_names = pickle.load(f)
        with open(root_ + 'related_phrases_related.pickle', 'rb') as f:
            self.related = pickle.load(f)
        with open(root_ + 'related_phrases_sorted_goods_indexes.pickle', 'rb') as f:
            self.sorted_goods_indexes = pickle.load(f)

        if os.path.exists(root_+'g_matrix_2.bin'):
            self.g_matrix = MemoryArray('g_matrix_2.bin', [len(self.good_phrases), len(self.good_phrases)], 4)
            self.g_matrix.load(root_)
        elif os.path.exists(root_+'g_matrix.bin'):
            self.g_matrix = MemoryArray('g_matrix.bin', [len(self.good_phrases), len(self.good_phrases)], 4)
            self.g_matrix.load(root_)

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
