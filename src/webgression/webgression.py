'''
    File name: webgression.py
    Author: Tyche Analytics Co.
'''
import shelve, time, requests
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from webgression_utils import text_from_html
from tqdm import *
import time
from nltk.stem.snowball import SnowballStemmer
from utils3 import binom_ci_analytic, mean
from itertools import repeat

class Webgression(object):
    def __init__(self, db_fname=None, max_features=None,
                 stop_words='english', stemmer='snowball',
                 click_through=True):
        """Initialize a Webgression object from optional db"""
        if db_fname is None:
            db_fname = "webgression_shelf_" + time.asctime().replace(" ", "_")
        self.shelf = shelve.open(db_fname)
        self.binger = Binger(click_through=click_through)
        if len(self.shelf) > 0:
            pass
        
    def search_name(self, name, outcome):
        print("starting on:", name)
        results = self.binger.bing_search(name)
        if results is None:
            print("Warning: search failed on", name)
        else:
            print("succeeded on:", name)
            self.shelf[name] = {"results":results,
                                "outcome":outcome,
                                "date":time.asctime()}

    def search_names(self, names, outcomes):
        for name, outcome in tqdm(zip(names, outcomes), total=len(names)):
            self.train_name(name, outcome)

    def fit(self, max_features = 1000):
        stemmer = SnowballStemmer('english')
        #names = list(self.shelf.keys())
        # we redefined names to only include seen non-renewals
        names = set([name for name, ren in zip(sample_df.insuredname, sample_df.renewal)
                     if type(name) is str and
                     name in seen_names and
                     name in self.shelf and
                     ren == 0])
        t0 = time.time()
        train_cutoff = int(len(names)*0.9)
        texts = [self.shelf[name]['results'] for name in names]
        train_texts = texts[:train_cutoff]
        test_texts = texts[train_cutoff:]
        ys = [int(self.shelf[name]['outcome']) for name in names]
        train_ys = ys[:train_cutoff]
        test_ys = ys[train_cutoff:]
        print("making train_tf")
        train_tf = self.vectorizer.fit_transform([text for text in train_texts])
        self.nb = MNB()
        self.nb.fit(train_tf.todense(), train_ys)
        train_yhats = self.nb.predict_proba(train_tf.todense())[:,1]
        print("making test_tf")
        test_tf = self.vectorizer.fit_transform([text for text in test_texts])
        test_yhats = self.nb.predict_proba(test_tf.todense())[:,1]
        print("train AUROC:", roc_auc_score(train_ys, train_yhats))
        print("test AUROC:", roc_auc_score(test_ys, test_yhats))

        text_clf = Pipeline([('vect', CountVectorizer(max_features=10000,
                                                      preprocessor=stemmer.stem)),
                             ('tfidf', TfidfTransformer()),
                             # ('clf', MNB()),
                             ('clf', SGDClassifier(loss='log', penalty='elasticnet',
                                                   alpha=0.00001))
        ])
        text_clf.fit(train_texts, train_ys)
        train_yhats = text_clf.predict_proba(train_texts)[:,1]
        test_yhats = text_clf.predict_proba(test_texts)[:,1]
        print("train AUROC:", roc_auc_score(train_ys, train_yhats))
        print("test AUROC:", roc_auc_score(test_ys, test_yhats))
        


    def summarize_model(self):
        words = sorted(self.vectorizer.vocabulary_.keys())
        ans = []
        for i, word in tqdm(enumerate(words)):
            n, p = map(int, self.nb.feature_count_[:,i])
            print(word, n, p)
            ci = binom_ci_analytic(n, p/(p + n))
            ans.append((word, n, p, p/(p+n), ci))
        for row in sorted(ans, key= lambda row:row[-2]):
            print(*row)
        return ans

    def predict(self, name):
        pass

class Binger(object):
    """wrap bing functionality.  The only job of the Binger is to accept
    names and return search results, up to the click_through depth"""
    def __init__(self, click_through=True):
        # set up api
        self.API_KEY = "<API KEY>" # free, new cc
        self.click_through=click_through

    def bing_search(self, query):
        url = 'https://api.cognitive.microsoft.com/bing/v7.0/search'
        # query string parameters
        payload = {'q': query}
        # custom headers
        headers = {'Ocp-Apim-Subscription-Key': self.API_KEY}
        # make GET request
        r = requests.get(url, params=payload, headers=headers)
        if not r.ok:
            print("Warning: Bing search failed on", query)
            return None
        # get JSON response
        json_obj = r.json()
        if not 'webPages' in json_obj:
            print("Warning: Bing search failed on", query, "(after request)")
            return None
        bing_items = json_obj['webPages']['value']
        bing_text = ("\n".join([x['name'] + " " + x['snippet']
                               for x in bing_items]))
        if self.click_through == False:
            return bing_text
        else: # if clicking_through
            link_text = ""
            urls = [x['url'] for x in bing_items]
            for url in urls:
                r = requests.get(url)
                if not r.ok:
                    print("Warning: request failed on", url)
                    continue
                html = r.text
                visible_text = text_from_html(html)
                link_text += "\n" + visible_text
            return bing_text + link_text
            
def transform_bing(train_df):
    with open(from_data_dir("bing_train_pos_texts.pkl"),'rb') as f:
        bing_train_pos_texts = dill.load(f)
    with open(from_data_dir("bing_train_neg_texts.pkl"),'rb') as f:
        bing_train_neg_texts = dill.load(f)
    with open(from_data_dir("bing_test_pos_texts.pkl"),'rb') as f:
        bing_test_pos_texts = dill.load(f)
    with open(from_data_dir("bing_test_neg_texts.pkl"),'rb') as f:
        bing_test_neg_texts = dill.load(f)
    bing_searches = dict(list(bing_train_pos_texts.items()) +
                         list(bing_train_neg_texts.items()) +
                         list(bing_test_pos_texts.items()) +
                         list(bing_test_neg_texts.items()))
    all_names = list(bing_searches.keys())
    safe_name_checklist = set(train_df.principal_name)
    safe_names = [name for name in all_names if name in safe_name_checklist]
    max_features = 1000
    name_vectorizer = CountVectorizer()
    vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=max_features,
                                stop_words='english',
                                 binary=True)
    t0 = time.time()
    name_tf = name_vectorizer.fit_transform(safe_names)
    tf = vectorizer.fit_transform([bing_searches[n] for n in safe_names])
    nb = GNB()
    bing_ys = [n in bing_train_pos_texts or n in bing_test_pos_texts
                for n in safe_names]
    nb.fit(tf.todense(), bing_ys)
    bing_yhats = [x[1] for x in nb.predict_proba(tf.todense())]
    train_yhats = np.zeros(len(train_df))
    print("analyzing train df")
    train_yhats = (name_vectorizer.transform(train_df.principal_name.fillna(""))
                   .dot(name_tf.T)
                   .dot(bing_yhats))
    print("len(train_yhats):", len(train_yhats))
    train_df['bing'] = train_yhats
    def transform_test_df(test_df):
        print("analyzing test df")
        test_yhats = (name_vectorizer.transform(test_df.principal_name.fillna(""))
                   .dot(name_tf.T)
                   .dot(bing_yhats))
        print("len(test_yhats):", len(test_yhats))
        test_df['bing'] = test_yhats
        return test_df
    return train_df, transform_test_df

def cross_validation_experiment():
    text_clf = Pipeline([('vect', CountVectorizer(preprocessor=stemmer.stem)),
                             ('tfidf', TfidfTransformer()),
                             # ('clf', MNB()),
                             ('clf', SGDClassifier(loss='log'))
        ])
    param_dist = {
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 100, 200, 500, 1000, 2000, 5000, 10000, 50000),
        #'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidf__use_idf': (True, False),
        #'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (0.00001, 0.000001),
        #'clf__alpha': (0.0000001,),
        'clf__penalty': ('l2', 'elasticnet'),
        #'clf__n_iter': (10, 50, 80),
    }
    n_iter_search = 10
    N = len(ys)
    train_cutoff = int(N*0.9)
    train_js = range(train_cutoff)
    test_js = range(train_cutoff, N)
    predef_split = PredefinedSplit([0 if i in test_js else -1 for i in range(N) ])
    random_search = RandomizedSearchCV(text_clf,
                                       param_distributions=param_dist,
                                       #cv=repeat((train_js, test_js)),
                                       cv=predef_split,
                                       scoring='roc_auc',
                                       n_iter=n_iter_search,
                                       verbose=3)
    cv_results = random_search.fit(texts, ys)
    

class Webgressor(BaseEstimator, TransformerMixin):
    def __init__(self):
        print("initializing webgressor")
        with open("webgression.pkl", 'rb') as f:
            raw_texts = pickle.load(f)
        self.text_dict = raw_texts
        for k in self.text_dict.keys():
            if self.text_dict[k] is None:
                self.text_dict[k] = ""
        self.fair_vocab = []
        self.analyzer = None
        print("finished initializing webgressor")

    def fit(self, names, ys, cutoff=3):
        print("self.text_dict :", len(self.text_dict))

        print("fitting webgressor")
        print("ys:", type(ys), len(ys))
        cv = CountVectorizer()
        texts = [self.text_dict[name] for name in names]
        print("fitting cv in webgressor.fit")
        X = cv.fit_transform(texts)
        Xp = X.tocsc()
        for word, j in (cv.vocabulary_.items()):
            col = Xp.getcol(j)
            js, _ = col.nonzero()
            rel_ys = ys.iloc[js]
            if sum(rel_ys == True) >= cutoff and sum(rel_ys == False) >= 3:
                self.fair_vocab.append(word)
        self.fair_vocab = set(self.fair_vocab)
        print("selected fair vocabulary of length:", len(self.fair_vocab))
        self.analyzer = cv.build_analyzer()
        print("finished fitting webgressor")
        return self
            

    def transform(self, names):
        print("transforming webgressor")
        print("self texts in transform:", len(self.text_dict))
        raw_texts = [self.text_dict[name] for name in names]
        fair_texts = [" ".join(filter(lambda x:x in self.fair_vocab, self.analyzer(raw_text)))
                      if raw_text else ""
                      for raw_text in raw_texts]
        print("webgression ans shape:", len(fair_texts))
        print("finished webgressor")
        return fair_texts

def interpret_pipeline(text_clf):
    vocab = sorted(text_clf.steps[0][1].vocabulary_.keys())
    coefs = text_clf.steps[2][1].coef_[0]
    sorted_vocab = sorted(zip(vocab, coefs), key=lambda vc:vc[1])
    print("bottom ten:")
    print(sorted_vocab[:10])
    print("top ten")
    print(sorted_vocab[-10:])
