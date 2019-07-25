from sklearn.feature_extraction.text import TfidfVectorizer


class MyTfidfVectorizer(TfidfVectorizer):
    def fit_transform(self, X, y):
        result = super(MyTfidfVectorizer, self).fit_transform(X, y)
        result.sort_indices()
        return result
