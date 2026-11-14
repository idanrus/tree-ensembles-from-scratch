import numpy as np


# ---------- Разбиение по датам ----------

def train_test_split(X_, y_, test_size, valid_size, date_column):
    """Сортируем строки по дате: первые строки в train, средние в valid, последние в test."""
    sorted_idx = date_column.sort_values(kind='stable').index
    n = len(sorted_idx)

    train_frac = 1.0 - test_size - valid_size
    train_end = int(n * train_frac)
    valid_end = int(n * (train_frac + valid_size))

    train_idx = sorted_idx[:train_end]
    valid_idx = sorted_idx[train_end:valid_end]
    test_idx = sorted_idx[valid_end:]

    return (X_.loc[train_idx], X_.loc[valid_idx], X_.loc[test_idx],
            y_.loc[train_idx], y_.loc[valid_idx], y_.loc[test_idx])


# ---------- Узел дерева ----------

class Node:
    """Узел дерева. Пока дерево строится, хранит свои объекты X, y.
    Умеет считать неопределённость (Gini или дисперсию) и искать лучший сплит."""

    def __init__(self, X_, y_, depth=0):
        self.X = X_
        self.y = y_
        self.depth = depth
        self.feature_ind = None
        self.threshold = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.value = None  # в листе: вероятность класса 1 (классификатор) или среднее (регрессор)

    @staticmethod
    def gini(p):
        """Gini impurity для бинарного таргета, p это доля единиц."""
        return 1. - p ** 2 - (1 - p) ** 2

    def impurity(self, criterion):
        if criterion == 'gini':
            return self.gini(self.y.mean())
        return self.y.var()

    def best_split(self, criterion='gini', feature_indices=None):
        """Перебираем все пороги по каждому признаку из feature_indices.
        Признак сортируем один раз, а суммы слева и справа считаем накопленными суммами (cumsum),
        поэтому каждый порог проверяется за O(1).
        Качество сплита: Q = H(родитель) - nL/n * H(левый) - nR/n * H(правый)."""
        n = len(self.y)
        if feature_indices is None:
            feature_indices = range(self.X.shape[1])
        impurity_base = self.impurity(criterion)

        best_Q, best_feat, best_t = -np.inf, None, None
        nL = np.arange(1, n)  # сколько объектов слева, если резать после i-го
        nR = n - nL
        for j in feature_indices:
            xj = self.X[:, j]
            idx = np.argsort(xj, kind='stable')
            x_sorted = xj[idx]
            y_sorted = self.y[idx]

            # резать можно только между разными значениями признака
            can_split = x_sorted[:-1] != x_sorted[1:]
            if not can_split.any():
                continue

            Sl = np.cumsum(y_sorted)[:-1]
            Sr = y_sorted.sum() - Sl
            if criterion == 'gini':
                impurity_l = self.gini(Sl / nL)
                impurity_r = self.gini(Sr / nR)
            else:
                Sql = np.cumsum(y_sorted ** 2)[:-1]
                Sqr = (y_sorted ** 2).sum() - Sql
                impurity_l = Sql / nL - (Sl / nL) ** 2
                impurity_r = Sqr / nR - (Sr / nR) ** 2

            Q = impurity_base - nL / n * impurity_l - nR / n * impurity_r
            Q[~can_split] = -np.inf
            i = np.argmax(Q)
            if Q[i] > best_Q:
                best_Q = Q[i]
                best_feat = j
                best_t = (x_sorted[i] + x_sorted[i + 1]) / 2
        return best_feat, best_t

    def best_split_random(self, criterion='gini', feature_indices=None, n_random_splits=1, rng=None):
        """Сплит для Extra Tree: по каждому признаку берём n_random_splits случайных порогов
        между min и max и выбираем лучший из них, а не из всех возможных."""
        n = len(self.y)
        if feature_indices is None:
            feature_indices = range(self.X.shape[1])
        if rng is None:
            rng = np.random.RandomState(21)
        impurity_base = self.impurity(criterion)

        best_Q, best_feat, best_t = -np.inf, None, None
        for j in feature_indices:
            xj = self.X[:, j]
            min_val, max_val = xj.min(), xj.max()
            if min_val == max_val:
                continue
            for _ in range(n_random_splits):
                t = rng.uniform(min_val, max_val)
                mask = xj <= t
                y_left, y_right = self.y[mask], self.y[~mask]
                nL, nR = len(y_left), len(y_right)
                if nL == 0 or nR == 0:
                    continue
                if criterion == 'gini':
                    impurity_l = self.gini(y_left.mean())
                    impurity_r = self.gini(y_right.mean())
                else:
                    impurity_l = y_left.var()
                    impurity_r = y_right.var()
                Q = impurity_base - nL / n * impurity_l - nR / n * impurity_r
                if Q > best_Q:
                    best_Q = Q
                    best_feat = j
                    best_t = t
        return best_feat, best_t


# ---------- Деревья ----------

class DecisionTree_custom:
    """Общая часть всех деревьев: рост дерева, выбор признаков, проход объекта до листа.
    Наследники меняют только критерий, функцию поиска сплита и значение в листе."""
    criterion = 'gini'

    def __init__(self, max_depth, max_features=None, random_state=21):
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.tree = None

    def n_features_to_try(self, n_features):
        """Сколько признаков смотреть в каждом узле: None значит все, 'sqrt', 'log2', доля или число."""
        if self.max_features is None:
            return n_features
        if self.max_features == 'sqrt':
            k = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            k = int(np.log2(n_features))
        elif isinstance(self.max_features, float):
            k = int(self.max_features * n_features)
        else:
            k = self.max_features
        return max(1, min(k, n_features))

    def find_split(self, node, feature_indices):
        return node.best_split(self.criterion, feature_indices)

    def leaf_value(self, y):
        return y.mean()  # для классификатора это доля единиц, то есть вероятность класса 1

    def build_tree(self, X, y, depth=0):
        node = Node(X, y, depth=depth)
        if depth == self.max_depth or len(y) < 2 or np.all(y == y[0]):
            j = None
        else:
            # признаки выбираем заново в каждом узле, как в sklearn
            k = self.n_features_to_try(X.shape[1])
            feature_indices = self.rng.choice(X.shape[1], k, replace=False)
            j, t = self.find_split(node, feature_indices)

        if j is None:
            node.is_leaf = True
            node.value = self.leaf_value(y)
        else:
            mask = X[:, j] <= t
            node.feature_ind = j
            node.threshold = t
            node.left = self.build_tree(X[mask], y[mask], depth + 1)
            node.right = self.build_tree(X[~mask], y[~mask], depth + 1)
        # после построения данные в узле больше не нужны, освобождаем память
        node.X = None
        node.y = None
        return node

    def fit(self, X_, y_):
        X = np.asarray(X_, dtype=float)
        y = np.asarray(y_, dtype=float)
        self.rng = np.random.RandomState(self.random_state)
        self.tree = self.build_tree(X, y)
        return self

    def predict_values(self, X_):
        """Спускаем объекты по дереву сразу пачкой: в каждом узле делим индексы на левые и правые."""
        X = np.asarray(X_, dtype=float)
        out = np.zeros(len(X))
        stack = [(self.tree, np.arange(len(X)))]
        while stack:
            node, idx = stack.pop()
            if node.is_leaf:
                out[idx] = node.value
            else:
                go_left = X[idx, node.feature_ind] <= node.threshold
                stack.append((node.left, idx[go_left]))
                stack.append((node.right, idx[~go_left]))
        return out

    def get_depth(self, node=None):
        node = self.tree if node is None else node
        if node.is_leaf:
            return 0
        return 1 + max(self.get_depth(node.left), self.get_depth(node.right))


class DecisionTreeClassifier_custom(DecisionTree_custom):
    criterion = 'gini'

    def predict_proba(self, X):
        p = self.predict_values(X)
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        return (self.predict_values(X) >= 0.5).astype(int)


class DecisionTreeRegressor_custom(DecisionTree_custom):
    criterion = 'mse'

    def predict(self, X):
        return self.predict_values(X)


class ExtraTreeClassifier_custom(DecisionTreeClassifier_custom):
    def __init__(self, max_depth, max_features=None, n_random_splits=1, random_state=21):
        super().__init__(max_depth, max_features, random_state)
        self.n_random_splits = n_random_splits

    def find_split(self, node, feature_indices):
        return node.best_split_random(self.criterion, feature_indices, self.n_random_splits, self.rng)


class ExtraTreeRegressor_custom(DecisionTreeRegressor_custom):
    def __init__(self, max_depth, max_features=None, n_random_splits=1, random_state=21):
        super().__init__(max_depth, max_features, random_state)
        self.n_random_splits = n_random_splits

    def find_split(self, node, feature_indices):
        return node.best_split_random(self.criterion, feature_indices, self.n_random_splits, self.rng)


# ---------- Ансамбли ----------

class RandomForestClassifier_custom:
    """Случайный лес: каждое дерево учится на бутстрэп-выборке строк
    и в каждом узле смотрит только на часть признаков. Ответ: среднее вероятностей деревьев."""

    def __init__(self, n_estimators, max_depth, max_features='sqrt', bootstrap=True, random_state=21):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []

    def make_tree(self, seed):
        return DecisionTreeClassifier_custom(self.max_depth, self.max_features, random_state=seed)

    def fit(self, X_, y_):
        X = np.asarray(X_, dtype=float)
        y = np.asarray(y_, dtype=float)
        n = len(X)
        rng = np.random.RandomState(self.random_state)
        self.trees = []
        for i in range(self.n_estimators):
            if self.bootstrap:
                indices = rng.choice(n, size=n, replace=True)
            else:
                indices = np.arange(n)
            tree = self.make_tree(seed=rng.randint(10 ** 9))
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)
        return self

    def predict_proba(self, X):
        p = np.mean([tree.predict_values(X) for tree in self.trees], axis=0)
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class ExtraTreesClassifier_custom(RandomForestClassifier_custom):
    """Extra Trees: то же самое, но деревья Extra Tree и по умолчанию без бутстрэпа:
    случайности и так хватает за счёт случайных порогов."""

    def __init__(self, n_estimators, max_depth, max_features='sqrt', n_random_splits=1,
                 bootstrap=False, random_state=21):
        super().__init__(n_estimators, max_depth, max_features, bootstrap, random_state)
        self.n_random_splits = n_random_splits

    def make_tree(self, seed):
        return ExtraTreeClassifier_custom(self.max_depth, self.max_features, self.n_random_splits, random_state=seed)


class GBDT_custom:
    """Градиентный бустинг для бинарной классификации с логлоссом.
    Модель считает логит F(x) = F0 + lr * (h1(x) + h2(x) + ...).
    Антиградиент логлосса по логиту равен y - p, на него и учится каждое следующее дерево."""

    def __init__(self, n_estimators, max_depth, max_features=None, learning_rate=0.1,
                 subsample=0.8, random_state=21):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.random_state = random_state
        self.trees = []
        self.F0 = None
        self.train_loss = []

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def log_loss(y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def fit(self, X_, y_):
        X = np.asarray(X_, dtype=float)
        y = np.asarray(y_, dtype=float)
        n = len(y)
        rng = np.random.RandomState(self.random_state)

        # старт: логарифм шансов, то есть константа, лучшая по логлоссу
        y_mean = np.clip(y.mean(), 1e-15, 1 - 1e-15)
        self.F0 = np.log(y_mean / (1 - y_mean))
        current_logits = np.zeros(n) + self.F0
        self.trees = []
        self.train_loss = [self.log_loss(y, self.sigmoid(current_logits))]

        for i in range(self.n_estimators):
            p = self.sigmoid(current_logits)
            S = y - p  # антиградиент
            indices = rng.choice(n, size=int(self.subsample * n), replace=False)
            tree = DecisionTreeRegressor_custom(self.max_depth, self.max_features, random_state=rng.randint(10 ** 9))
            tree.fit(X[indices], S[indices])
            current_logits += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
            self.train_loss.append(self.log_loss(y, self.sigmoid(current_logits)))
        return self

    def decision_function(self, X):
        F = np.zeros(len(X)) + self.F0
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return F

    def predict_proba(self, X):
        p = self.sigmoid(self.decision_function(X))
        return np.vstack([1 - p, p]).T

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
