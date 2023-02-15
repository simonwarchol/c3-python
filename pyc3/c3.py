import json
import math
from pathlib import Path

import colorutil
import numpy as np
# from line_profiler_pycharm import profile
from tqdm import tqdm


class c3:
    def __init__(self, data=(Path(__file__).parent / 'data' / 'xkcd' / 'c3_data.json').resolve()):
        self.minE = -4.5
        self.maxE = 0
        self.data = json.load(open(data))
        self.color = []
        for i in range(len(self.data['color']) // 3):
            lab_color = {
                'L': self.data['color'][i * 3],
                'a': self.data['color'][i * 3 + 1],
                'b': self.data['color'][i * 3 + 2]
            }
            self.color.append(lab_color)
        self.C = len(self.color)
        # parse terms
        self.terms = self.data['terms']
        self.W = len(self.terms)
        # parse count table
        self.T = {}
        for i in range(len(self.data['T']) // 2):
            self.T[self.data['T'][i * 2]] = self.data['T'][i * 2 + 1]

        self.colorCount = [0 for i in range(self.C)]
        self.termsCount = [0 for i in range(self.W)]
        for t in self.T:
            c = math.floor(t / self.W)
            w = math.floor(t % self.W)
            v = 0
            if t in self.T:
                v = self.T[t]
            self.colorCount[c] += v
            self.termsCount[w] += v
        self.A = self.data['A']
        self.map = {}
        for c in range(self.C):
            x = self.color[c]
            s = ','.join([str(x['L']), str(x['a']), str(x['b'])])
            self.map[s] = c
        test = ''

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

    # palette is list of hex colors
    def analyze_palette(self, palette, color_term_limit=1):

        # If list of string, convert to RGB list

        palette = self.parse_palette(palette)

        def index(_c):
            x = colorutil.srgb_to_lab(_c)

            L = 5 * round(x[0] / 5)
            a = 5 * round(x[1] / 5)
            b = 5 * round(x[2] / 5)
            s = ",".join([str(x) for x in [L, a, b]])
            if s in self.map:
                return self.map[s]
            return None

        def color(_x, color_term_limit=1):
            c = index(_x)
            h = (self.color_entropy(c) - self.minE) / (self.maxE - self.minE)
            t = self.color_related_terms(c, limit=color_term_limit)
            z = colorutil.srgb_to_lab(_x)
            z = str(math.trunc(z[0])) + ', ' + str(math.trunc(z[1])) + ', ' + str(math.trunc(z[2]))
            return {"x": _x, "c": c, "h": h, 'terms': t, "z": z}

        data = [color(x, color_term_limit=color_term_limit) for x in palette]
        return data

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
            obj['lab'] = [int(x.strip()) for x in data[i]['z'].split(',')]
            obj['salience'] = data[i]['h']
            obj['terms'] = [{'p': x['score'], 'name': self.terms[x['index']]} for x in data[i]['terms']]
            clean_data.append(obj)
        return clean_data

    def color_name_distance_matrix(self, palette):
        data = self.analyze_palette(palette)
        return self.compute_color_name_distance_matrix(data)

    def compute_color_name_distance_matrix(self, data):

        matrix = np.zeros((len(data), len(data)))
        for i in range(0, len(data)):
            for j in range(0, i):
                cosine_distance = 1 - self.color_cosine(data[i]['c'], data[j]['c'])
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
