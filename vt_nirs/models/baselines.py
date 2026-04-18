import numpy as np


class CustomRegressionTree:

    def __init__(self, max_depth=6, min_samples_leaf=20,
                 max_features=None, honest=False):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.honest = honest
        self.tree = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if self.honest:
            n = len(X)
            idx = np.random.permutation(n)
            mid = n // 2
            split_idx = idx[:mid]
            est_idx = idx[mid:]

            self.tree = self._build(X[split_idx], y[split_idx], depth=0)
            self._honest_reestimate(self.tree, X[est_idx], y[est_idx])
        else:
            self.tree = self._build(X, y, depth=0)

        return self

    def _build(self, X, y, depth):
        """Recursively build tree nodes."""
        n_samples = len(y)
        node = {'value': np.mean(y), 'n': n_samples}

        if (depth >= self.max_depth or
                n_samples < 2 * self.min_samples_leaf or
                np.var(y) < 1e-10):
            node['leaf'] = True
            return node

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]
        if self.max_features is not None:
            feature_indices = np.random.choice(
                n_features, size=min(self.max_features, n_features),
                replace=False)
        else:
            feature_indices = np.arange(n_features)

        total_var = np.var(y) * n_samples

        for feat in feature_indices:
            col = X[:, feat]
            unique_vals = np.unique(col)
            if len(unique_vals) <= 20:
                thresholds = unique_vals
            else:
                thresholds = np.percentile(col, np.linspace(5, 95, 20))

            for thresh in thresholds:
                left_mask = col <= thresh
                right_mask = ~left_mask
                n_left = left_mask.sum()
                n_right = right_mask.sum()

                if (n_left < self.min_samples_leaf or
                        n_right < self.min_samples_leaf):
                    continue

                var_left = np.var(y[left_mask]) * n_left
                var_right = np.var(y[right_mask]) * n_right
                gain = total_var - (var_left + var_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat
                    best_threshold = thresh

        if best_feature is None:
            node['leaf'] = True
            return node

        left_mask = X[:, best_feature] <= best_threshold
        node['leaf'] = False
        node['feature'] = best_feature
        node['threshold'] = best_threshold
        node['left'] = self._build(X[left_mask], y[left_mask], depth + 1)
        node['right'] = self._build(X[~left_mask], y[~left_mask], depth + 1)

        return node

    def _honest_reestimate(self, node, X, y):
        if node.get('leaf', False) or 'feature' not in node:
            if len(y) > 0:
                node['value'] = np.mean(y)
            node['n'] = len(y)
            return

        left_mask = X[:, node['feature']] <= node['threshold']
        self._honest_reestimate(node['left'], X[left_mask], y[left_mask])
        self._honest_reestimate(node['right'], X[~left_mask], y[~left_mask])

    def predict(self, X):
        """Predict by traversing tree for each sample."""
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _predict_one(self, x, node):
        if node.get('leaf', False) or 'feature' not in node:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])

    def get_leaf_id(self, x, node=None):
        """Return leaf ID for a single sample (for Causal Forest weighting)."""
        if node is None:
            node = self.tree
        if node.get('leaf', False) or 'feature' not in node:
            return id(node)
        if x[node['feature']] <= node['threshold']:
            return self.get_leaf_id(x, node['left'])
        else:
            return self.get_leaf_id(x, node['right'])


class CustomTLearner:

    def __init__(self, n_trees=100, max_depth=6, min_samples_leaf=20,
                 max_features_frac=0.5, subsample_frac=0.8, random_state=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features_frac = max_features_frac
        self.subsample_frac = subsample_frac
        self.random_state = random_state
        self.forest_0 = []
        self.forest_1 = []

    def fit(self, X, W, Y):
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        X_0, Y_0 = X[W == 0], Y[W == 0]
        X_1, Y_1 = X[W == 1], Y[W == 1]

        n_features = X.shape[1]
        max_features = max(1, int(n_features * self.max_features_frac))

        print(f'  Fitting μ_0 forest ({self.n_trees} trees, '
              f'{len(X_0)} control samples)...')
        self.forest_0 = self._build_forest(
            X_0, Y_0, max_features)

        print(f'  Fitting μ_1 forest ({self.n_trees} trees, '
              f'{len(X_1)} treated samples)...')
        self.forest_1 = self._build_forest(
            X_1, Y_1, max_features)

        print(f'  T-Learner fit complete.')
        return self

    def _build_forest(self, X, Y, max_features):
        """Build a random forest (list of CustomRegressionTree)."""
        forest = []
        n = len(X)
        n_subsample = max(1, int(n * self.subsample_frac))

        for t in range(self.n_trees):
            boot_idx = np.random.choice(n, size=n_subsample, replace=True)
            tree = CustomRegressionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
            )
            tree.fit(X[boot_idx], Y[boot_idx])
            forest.append(tree)
        return forest

    def predict_ite(self, X):
        X = np.asarray(X, dtype=np.float64)

        mu_0 = np.mean([tree.predict(X) for tree in self.forest_0], axis=0)
        mu_1 = np.mean([tree.predict(X) for tree in self.forest_1], axis=0)
        ite = mu_1 - mu_0

        return ite, mu_0, mu_1


class CustomCausalForest:
    def __init__(self, n_trees=100, max_depth=5, min_samples_leaf=30,
                 max_features_frac=0.5, subsample_frac=0.8, random_state=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features_frac = max_features_frac
        self.subsample_frac = subsample_frac
        self.random_state = random_state
        self.trees = []

    def fit(self, X, W, Y):
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        n = len(X)
        n_sub = max(1, int(n * self.subsample_frac))
        n_features = X.shape[1]
        max_features = max(1, int(n_features * self.max_features_frac))

        print(f'  Fitting Causal Forest ({self.n_trees} trees, '
              f'{n} samples)...')

        self.trees = []
        for t in range(self.n_trees):
            sub_idx = np.random.choice(n, size=n_sub, replace=True)
            X_sub = X[sub_idx]
            W_sub = W[sub_idx]
            Y_sub = Y[sub_idx]

            perm = np.random.permutation(n_sub)
            mid = n_sub // 2
            struct_idx = perm[:mid]
            est_idx = perm[mid:]

            tree = self._build_causal_tree(
                X_sub[struct_idx], W_sub[struct_idx], Y_sub[struct_idx],
                max_features, depth=0)

            self._honest_causal_reestimate(
                tree, X_sub[est_idx], W_sub[est_idx], Y_sub[est_idx])

            self.trees.append(tree)

        print(f'  Causal Forest fit complete.')
        return self

    def _build_causal_tree(self, X, W, Y, max_features, depth):
        n = len(Y)
        n_treated = (W == 1).sum()
        n_control = (W == 0).sum()

        tau = 0.0
        if n_treated > 0 and n_control > 0:
            tau = Y[W == 1].mean() - Y[W == 0].mean()

        node = {
            'tau': tau,
            'n': n,
            'n_treated': int(n_treated),
            'n_control': int(n_control),
        }

        if (depth >= self.max_depth or
                n < 2 * self.min_samples_leaf or
                n_treated < 2 or n_control < 2):
            node['leaf'] = True
            return node

        best_score = -np.inf
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]
        if max_features is not None:
            feat_idx = np.random.choice(
                n_features, size=min(max_features, n_features), replace=False)
        else:
            feat_idx = np.arange(n_features)

        for feat in feat_idx:
            col = X[:, feat]
            unique_vals = np.unique(col)
            if len(unique_vals) <= 20:
                thresholds = unique_vals
            else:
                thresholds = np.percentile(col, np.linspace(5, 95, 20))

            for thresh in thresholds:
                left = col <= thresh
                right = ~left

                n_left = left.sum()
                n_right = right.sum()

                if (n_left < self.min_samples_leaf or
                        n_right < self.min_samples_leaf):
                    continue

                if ((W[left] == 1).sum() < 1 or (W[left] == 0).sum() < 1 or
                        (W[right] == 1).sum() < 1 or (W[right] == 0).sum() < 1):
                    continue

                tau_left = (Y[left & (W == 1)].mean() -
                            Y[left & (W == 0)].mean())
                tau_right = (Y[right & (W == 1)].mean() -
                             Y[right & (W == 0)].mean())

                score = (n_left * tau_left ** 2 +
                         n_right * tau_right ** 2)

                if score > best_score:
                    best_score = score
                    best_feature = feat
                    best_threshold = thresh

        if best_feature is None:
            node['leaf'] = True
            return node

        left_mask = X[:, best_feature] <= best_threshold
        node['leaf'] = False
        node['feature'] = best_feature
        node['threshold'] = best_threshold
        node['left'] = self._build_causal_tree(
            X[left_mask], W[left_mask], Y[left_mask],
            max_features, depth + 1)
        node['right'] = self._build_causal_tree(
            X[~left_mask], W[~left_mask], Y[~left_mask],
            max_features, depth + 1)

        return node

    def _honest_causal_reestimate(self, node, X, W, Y):
        if node.get('leaf', False) or 'feature' not in node:
            n_treated = (W == 1).sum()
            n_control = (W == 0).sum()
            if n_treated > 0 and n_control > 0:
                node['tau'] = Y[W == 1].mean() - Y[W == 0].mean()
            elif n_treated > 0:
                node['tau'] = Y[W == 1].mean()
            elif n_control > 0:
                node['tau'] = -Y[W == 0].mean()
            else:
                node['tau'] = 0.0
            node['n'] = len(Y)
            node['n_treated'] = int(n_treated)
            node['n_control'] = int(n_control)
            return

        left_mask = X[:, node['feature']] <= node['threshold']
        self._honest_causal_reestimate(
            node['left'], X[left_mask], W[left_mask], Y[left_mask])
        self._honest_causal_reestimate(
            node['right'], X[~left_mask], W[~left_mask], Y[~left_mask])

    def predict_ite(self, X):
        X = np.asarray(X, dtype=np.float64)
        tree_preds = np.array([
            self._predict_tree(X, tree) for tree in self.trees
        ])
        ite = np.mean(tree_preds, axis=0)
        return ite

    def _predict_tree(self, X, tree):
        """Predict τ for each sample by traversing one causal tree."""
        return np.array([self._predict_one(x, tree) for x in X])

    def _predict_one(self, x, node):
        if node.get('leaf', False) or 'feature' not in node:
            return node['tau']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])


def run_baselines(X_baseline, W, Y, random_state=42):
    results = {}

    print('Running T-Learner baseline...')
    tl = CustomTLearner(
        n_trees=100, max_depth=6, min_samples_leaf=20,
        max_features_frac=0.5, subsample_frac=0.8,
        random_state=random_state,
    )
    tl.fit(X_baseline, W, Y)
    ite_tl, mu0_tl, mu1_tl = tl.predict_ite(X_baseline)
    results['T-Learner'] = {
        'ite': ite_tl, 'mu_0': mu0_tl, 'mu_1': mu1_tl,
        'model': tl,
    }
    print(f'  T-Learner ATE: {ite_tl.mean():.4f}')

    print('\nRunning Causal Forest baseline...')
    cf = CustomCausalForest(
        n_trees=100, max_depth=5, min_samples_leaf=30,
        max_features_frac=0.5, subsample_frac=0.8,
        random_state=random_state,
    )
    cf.fit(X_baseline, W, Y)
    ite_cf = cf.predict_ite(X_baseline)
    results['Causal Forest'] = {
        'ite': ite_cf,
        'model': cf,
    }
    print(f'  Causal Forest ATE: {ite_cf.mean():.4f}')

    return results
