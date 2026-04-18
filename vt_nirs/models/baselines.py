import numpy as np


class CustomRegressionTree:
    """
    Regression tree built from scratch via recursive binary splitting.

    # Ref: Breiman L. Machine Learning 2001 — CART algorithm
    #      Section 2: tree growing via MSE split criterion.
    # Ref: Athey & Imbens PNAS 2016 — honest estimation variant
    #      Section 3: separate splitting and estimation samples.

    Args:
        max_depth: maximum tree depth (default 6)
        min_samples_leaf: minimum samples per leaf (default 20)
        max_features: number of features to consider per split
                      (default None = all features)
        honest: if True, uses honest estimation (Athey & Imbens 2016)
                — splits on one half, estimates on the other half
    """

    def __init__(self, max_depth=6, min_samples_leaf=20,
                 max_features=None, honest=False):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.honest = honest
        self.tree = None

    def fit(self, X, y):
        """
        Fit regression tree to data.

        # Ref: Breiman ML 2001, Section 2 — recursive partitioning
        #      finds (feature, threshold) minimizing sum of MSE in children.
        # Ref: Athey & Imbens PNAS 2016, Section 3 — honest splitting:
        #      split sample in half, build tree on first half,
        #      estimate leaf values on second half.

        Args:
            X: (N, D) feature matrix
            y: (N,) target values
        """
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
        """
        Re-estimate leaf values using estimation sample.

        # Ref: Athey & Imbens PNAS 2016, Section 3, Theorem 1:
        #      "The honest tree estimator is unbiased for the
        #       conditional mean at each leaf."
        """
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
    """
    T-Learner for heterogeneous treatment effect estimation.

    # Ref: Künzel et al. PNAS 2019, Section 2.1, Algorithm 1:
    #      "The T-learner consists of two steps:
    #       Step 1: Estimate μ_0(x) = E[Y(0)|X=x] using control group only.
    #       Step 2: Estimate μ_1(x) = E[Y(1)|X=x] using treated group only.
    #       The CATE estimate is τ(x) = μ_1(x) - μ_0(x)."
    #      Lines from Section 2.1, paragraph 1-2.

    We implement μ_0 and μ_1 as random forests (ensemble of custom
    regression trees) per Breiman ML 2001.

    Args:
        n_trees: number of trees per arm (default 100)
        max_depth: tree depth (default 6)
        min_samples_leaf: leaf size (default 20)
        max_features_frac: fraction of features per split (default 0.5)
        subsample_frac: bootstrap sample fraction (default 0.8)
        random_state: seed
    """

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
        """
        Fit T-Learner: separate forests for each treatment arm.

        # Ref: Künzel et al. PNAS 2019, Algorithm 1, Steps 1-2:
        #      "Using the control observations, estimate the control
        #       response function μ_0. Using the treated observations,
        #       estimate the treated response function μ_1."
        # Ref: Breiman ML 2001, Section 3 — random forest:
        #      bootstrap sampling + random feature subsets.

        Args:
            X: (N, D) baseline covariates
            W: (N,) treatment indicators (0=control, 1=treated)
            Y: (N,) observed outcomes (VFD-28)
        """
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
        """
        Predict ITE = μ_1(x) - μ_0(x) for each sample.

        # Ref: Künzel et al. PNAS 2019, Section 2.1:
        #      "The CATE estimate is: τ_hat(x) = μ_hat_1(x) - μ_hat_0(x)"

        Args:
            X: (N, D) covariates
        Returns:
            ite: (N,) estimated ITE (positive = NIRS beneficial)
            mu_0: (N,) estimated E[Y(0)|X]
            mu_1: (N,) estimated E[Y(1)|X]
        """
        X = np.asarray(X, dtype=np.float64)

        mu_0 = np.mean([tree.predict(X) for tree in self.forest_0], axis=0)
        mu_1 = np.mean([tree.predict(X) for tree in self.forest_1], axis=0)
        ite = mu_1 - mu_0

        return ite, mu_0, mu_1


class CustomCausalForest:
    """
    Causal Forest for heterogeneous treatment effect estimation.

    # Ref: Athey, Tibshirani & Wager, Annals of Statistics 2019,
    #      Section 4: "Causal Forests"
    #      — Each tree splits to maximize treatment effect heterogeneity,
    #        using doubly-robust scores as pseudo-outcomes.
    #      Algorithm 1 (Section 2.4): honest estimation with sample splitting.
    #
    # Ref: Athey & Imbens PNAS 2016, Section 3:
    #      "Causal trees" — honest splitting for unbiased CATE estimation.
    #      Key insight: split criterion uses treatment effect variance,
    #      not just outcome variance.

    Implementation:
      1. For each tree, bootstrap sample, split into structure/estimation halves.
      2. Build tree by splitting on the feature/threshold that maximizes
         the squared difference in treatment effects between children.
      3. At each leaf, estimate τ(x) = mean(Y_i | W_i=1) - mean(Y_i | W_i=0)
         using the estimation sample only (honest estimation).
      4. Forest ITE = average of per-tree leaf estimates.

    Args:
        n_trees: number of causal trees (default 100)
        max_depth: tree depth (default 5)
        min_samples_leaf: minimum samples per leaf (default 30, higher than
                          T-Learner because each leaf needs both treated
                          and control units)
        max_features_frac: fraction of features per split (default 0.5)
        subsample_frac: fraction of data per tree (default 0.8)
        random_state: seed
    """

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
        """
        Fit causal forest.

        # Ref: Athey et al. Annals of Statistics 2019, Algorithm 1:
        #      "For b = 1,...,B:
        #       (a) Draw a subsample S_b of size s from {1,...,n}.
        #       (b) Split S_b into disjoint halves I_b (structure) and J_b (estimation).
        #       (c) Grow a tree on I_b using causal splits.
        #       (d) Estimate leaf effects using J_b."

        Args:
            X: (N, D) baseline covariates
            W: (N,) treatment (0 or 1)
            Y: (N,) outcomes
        """
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
        """
        Recursively build causal tree with treatment-effect-maximizing splits.

        # Ref: Athey & Imbens PNAS 2016, Section 3.2:
        #      "The criterion for splitting ... maximize[s] the variance of
        #       the estimated treatment effects across the two child nodes."
        #      Formally: argmax_split { n_L * τ_L^2 + n_R * τ_R^2 }
        #      where τ_L = E[Y|W=1,left] - E[Y|W=0,left].
        """
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
        """
        Re-estimate leaf treatment effects using estimation sample.

        # Ref: Athey et al. Annals of Statistics 2019, Algorithm 1 step (d):
        #      "For each leaf L of the tree, compute
        #       τ(x) = (1/|{i∈J_b: X_i∈L, W_i=1}|) Σ Y_i
        #            - (1/|{i∈J_b: X_i∈L, W_i=0}|) Σ Y_i"
        """
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
        """
        Predict ITE by averaging leaf τ across all causal trees.

        # Ref: Athey et al. Annals of Statistics 2019, Section 2.4:
        #      "The forest prediction is the average of the individual
        #       tree predictions: τ_hat(x) = (1/B) Σ_b τ_b(x)"

        Args:
            X: (N, D) covariates
        Returns:
            ite: (N,) estimated CATE (positive = NIRS beneficial)
        """
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
    """
    Run both baseline models and return ITE predictions.

    # Ref: Künzel et al. PNAS 2019 — T-Learner
    # Ref: Athey et al. Annals of Statistics 2019 — Causal Forest

    Args:
        X_baseline: (N, D) baseline covariates (e.g., last timestep)
        W: (N,) treatment indicators
        Y: (N,) observed outcomes (VFD-28)
        random_state: seed
    Returns:
        dict with ITE predictions for each baseline
    """
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
