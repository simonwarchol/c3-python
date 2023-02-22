import json
import math
from pathlib import Path

import colorutil
import numpy as np
from line_profiler_pycharm import profile
from sklearn.neighbors import BallTree
from tqdm import tqdm
from scipy.spatial.distance import squareform

class c3:
    # @profile
    def __init__(self, data=(Path(__file__).parent / 'data' / 'xkcd' / 'c3_data.json').resolve()):
        self.minE = -4.5
        self.maxE = 0
        self.data = json.load(open(data))
        self.color = np.array(self.data['color']).reshape(-1, 3)
        self.C = len(self.color)
        # parse terms
        self.terms = self.data['terms']
        self.W = len(self.terms)
        # parse count table
        self.T = {}
        self.tempT = np.array(self.data['T']).reshape(-1, 2)
        for i in range(len(self.data['T']) // 2):
            self.T[self.data['T'][i * 2]] = self.data['T'][i * 2 + 1]

        self.colorCount = [0 for i in range(self.C)]
        self.termsCount = [0 for i in range(self.W)]
        self.cosine_matrix = squareform(np.load((Path(__file__).parent / 'data' / 'xkcd' / 'cosine_distances_square.npy').resolve()))
        for t in self.T:
            c = math.floor(t / self.W)
            w = math.floor(t % self.W)
            v = 0
            if t in self.T:
                v = self.T[t]
            self.colorCount[c] += v
            self.termsCount[w] += v
        self.A = self.data['A']
        self.tree = BallTree(np.array(self.color), metric='euclidean')
        self.map = {}
        for c in range(self.C):
            x = self.color[c]
            test = ",".join(x.astype(str))
            s = ','.join([str(x[0]), str(x[1]), str(x[2])])
            self.map[s] = c

    def color_entropy(self, c):
        H = 0
        for w in range(self.W):
            # (T[c*W+w]||0) / tcount[w];
            val = c * self.W + w
            p = 0
            if val in self.T:
                p = self.T[val]
            p = p / self.colorCount[c]
            if p > 0:
                H += (p * math.log(p) / math.log(2))
        return H

    # @profile
    def color_related_terms(self, c, limit=None, minCount=None, salience_threshold=None):
        cc = c * self.W
        _list = []
        _sum = 0
        for w in range(self.W):
            if cc + w in self.T:
                _sum += self.T[cc + w]
                _list.append({'index': w, 'score': self.T[cc + w]})
        _list = [{'score': (x['score'] / _sum), 'index': x['index']} for x in _list]
        if salience_threshold is not None:
            _list = list(filter(lambda x: x['score'] > salience_threshold, _list))
        if minCount is not None:
            _list = list(filter(lambda x: self.termsCount[x.index] > minCount >= minCount, _list))
        _list.sort(key=lambda x: x['score'], reverse=True)
        if limit is not None:
            _list = _list[:limit]
        return _list

    def color_cosine(self, a, b):
        sa = 0
        sb = 0
        sc = 0
        for w in range(self.W):
            ta = 0
            tb = 0
            if a * self.W + w in self.T:
                ta = self.T[a * self.W + w]
            if b * self.W + w in self.T:
                tb = self.T[b * self.W + w]
            sa += ta * ta
            sb += tb * tb
            sc += ta * tb
        return sc / math.sqrt(sa * sb)

    def parse_palette(self, palette):
        if all(isinstance(s, str) for s in palette):
            def hex_to_rgb(hex):
                if hex.startswith('#'):
                    hex = hex[1:]
                return np.array(tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))) / 255.0

            palette = [hex_to_rgb(c) for c in palette]
        return palette

    def color_index(self, _c):
        return self.tree.query(_c.reshape(1, -1), k=1)[1][0][0]

        # def index(_c):
        #     x = colorutil.srgb_to_lab(_c)
        #

    # palette is list of hex colors
    def analyze_palette(self, palette, color_term_limit=1):

        # If list of string, convert to RGB list


        def color(_x, color_term_limit=1):
            c = self.color_index(_x)
            h = (self.color_entropy(c) - self.minE) / (self.maxE - self.minE)
            t = self.color_related_terms(c, limit=color_term_limit)
            return {"x": _x, "c": c, "h": h, 'terms': t}

        data = [color(x, color_term_limit=color_term_limit) for x in palette]
        return data

    def palette_indices(self, palette):
        palette = self.parse_palette(palette)
        return [{'c': self.color_index(x)} for x in palette]

    # @profile
    def analyze(self, palette, color_term_limit=1):
        data = self.analyze_palette(palette, color_term_limit)
        color_name_distance_matrix = self.compute_color_name_distance_matrix(data)
        cleaned_data = self.cleanup_data(data, palette)
        return {'palette_data': cleaned_data, 'color_name_distance_matrix': color_name_distance_matrix}

    def cleanup_data(self, data, palette):
        clean_data = []
        for i in range(0, len(data)):
            obj = {}
            obj['color'] = palette[i]
            obj['rgb'] = [int(x * 255) for x in data[i]['x'].tolist()]
            obj['salience'] = data[i]['h']
            obj['terms'] = [{'p': x['score'], 'name': self.terms[x['index']]} for x in data[i]['terms']]
            clean_data.append(obj)
        return clean_data

    def color_name_distance_matrix(self, palette):
        data = self.palette_indices(palette)
        return self.compute_color_name_distance_matrix(data)

    def compute_color_name_distance_matrix(self, data):

        matrix = np.zeros((len(data), len(data)))
        for i in range(0, len(data)):
            for j in range(0, i):
                cosine_distance = self.cosine_matrix[data[i]['c'], data[j]['c']]
                matrix[i, j] = matrix[j, i] = cosine_distance
        return matrix

    # @profile
    def get_color_salience_dict(self, salience_threshold=None, limit=10):
        color_term_salience = {}
        all_colors_lab = [[int(_c) for _c in c.split(',')] for c in self.map]
        all_colors_srgb = colorutil.lab_to_srgb(np.array(all_colors_lab))
        for i, c in tqdm(enumerate(self.map)):
            color_index = self.map[c]
            lab_array = np.array([int(c) for c in c.split(',')])
            terms = self.color_related_terms(color_index, salience_threshold=salience_threshold, limit=10)
            for t in terms:
                term = self.terms[t['index']]
                if term not in color_term_salience:
                    color_term_salience[term] = []
                color_term_salience[term].append(
                    {'srgb_color': all_colors_srgb[i].tolist(), 'score': t['score'], 'lab_color': lab_array.tolist()})
        for term in color_term_salience:
            color_term_salience[term].sort(key=lambda x: x['score'], reverse=True)
        return color_term_salience

    def get_most_salient_colors(self, salience_threshold=None, limit=10):
        color_term_salience = self.get_color_salience_dict(salience_threshold=salience_threshold, limit=limit)
        color_term_salience_most_salient = [[k, np.mean([x['score'] for x in v[0:10]])] for k, v in
                                            color_term_salience.items()]
        sorted_color_term_salience_most_salient = sorted(color_term_salience_most_salient, key=lambda x: x[1],
                                                         reverse=True)
        return sorted_color_term_salience_most_salient


    def precompute_cosine_distances(self):
        self.cosine_distances = np.zeros((self.C, self.C))
        for i in tqdm(range(0, self.C)):
            for j in range(0, i):
                cosine_distance = 1 - self.color_cosine(i, j)
                self.cosine_distances[i, j] = self.cosine_distances[j, i] = cosine_distance
        np.save('cosine_distances.npy', self.cosine_distances)

    def load_cosine_dist(self):
        matrix = np.load('cosine_distances.npy')
        # Save as 16bit float
        matrix = matrix.astype(np.float32)
        np.save('cosine_distances_32.npy', matrix)
        matrix = matrix.astype(np.float16)
        np.save('cosine_distances_16.npy', matrix)
        tosquare = squareform(matrix)
        np.save('cosine_distances_square.npy', tosquare)
        fromsquare = squareform(tosquare)
        test = ''
