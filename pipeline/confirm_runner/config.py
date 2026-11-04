"""Small v2 configuration interface consumed by the unchanged arms functions."""
from itertools import product


class V2Config:
    def __init__(self, doc):
        self.doc = doc
        self.draw_rng = (doc['designs']['P']['draw_fold_multiplier'], doc['designs']['P']['draw_seed_offset'])
        self.inner_rng = (self.draw_rng[0], doc['inner']['seed_offset'])
        self.inner_frac = doc['inner']['fraction']
        self.y = dict(heads=dict(hist_gbm=doc['models']['hist_gbm'], logistic=doc['models']['logistic_descriptive']))

    def head_grid(self, kind):
        if kind == 'hist_gbm':
            s = self.doc['models']['hist_gbm']
            return [dict(learning_rate=lr, max_leaf_nodes=k) for lr, k in product(s['learning_rate'], s['max_leaf_nodes'])]
        if kind == 'logistic':
            return [dict(C=c) for c in self.doc['models']['logistic_descriptive']['C']]
        raise ValueError(kind)
