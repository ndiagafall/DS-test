#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:25:10 2020

@author: fall
"""

#################################################################################################################################
############################ Systèmes de Recommandation #########################################################################
################################################################################################################################

# Ce cahier est une introduction pratique aux principales techniques du système de recommandation (RecSys).
# L'objectif d'un RecSys est de recommander des éléments pertinents aux utilisateurs, en fonction de leurs préférences.
# La préférence et la pertinence sont subjectives et sont généralement déduites des éléments que les utilisateurs ont consommés précédemment.
# Les principales familles de méthodes pour RecSys sont:

# Filtrage collaboratif: cette méthode fait des prédictions automatiques
# (filtrage) sur les intérêts d'un utilisateur en collectant les préférences ou les informations de goût de nombreux utilisateurs (collaborant). L'hypothèse sous-jacente de l'approche de filtrage collaboratif est que si une personne A a la même opinion qu'une personne B sur un ensemble d'éléments, A est plus susceptible d'avoir l'opinion de B pour un élément donné que celle d'une personne choisie au hasard.

# Filtrage basé sur le contenu: cette méthode utilise uniquement des informations
# sur la description et les attributs des éléments que les utilisateurs ont précédemment consommés pour modéliser les préférences de l'utilisateur. En d'autres termes, ces algorithmes essaient de recommander des éléments similaires à ceux qu'un utilisateur aimait dans le passé (ou examine dans le présent). En particulier, divers éléments candidats sont comparés avec des éléments précédemment évalués par l'utilisateur et les éléments les mieux adaptés sont recommandés.

# Méthodes hybrides: Des recherches récentes ont démontré qu'une approche hybride,
# combinant le filtrage collaboratif et le filtrage basé sur le contenu pourrait être plus efficace que les approches pures dans certains cas. Ces méthodes peuvent également être utilisées pour surmonter certains des problèmes courants dans les systèmes recommandés tels que le démarrage à froid et le problème de rareté.

# Dans ce cahier, nous utilisons un ensemble de données que nous avons partagé sur les ensembles de données Kaggle:
# Partage et lecture d'articles à partir de CI&T Deskdrop.
# Nous montrerons comment implémenter le filtrage collaboratif,
# le filtrage basé sur le contenu et les méthodes hybrides en Python, afin de fournir des recommandations personnalisées aux utilisateurs.

import numpy as np
import scipy
import pandas as pd
import math
import random

import sklearn
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from matplotlib.ticker import  FormatStrFormatter
from matplotlib.figure import Figure

articles_df = pd.read_csv('shared_articles.csv')
articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
articles_df.head(5)

# le dataset users_interactions.csv contient des journaux, des interactions des utilisateurs sur les articles partagés.
# Il peut être joint à articles_shared.csv par la colonne contentId.
# Les valeurs eventType sont:

# VIEW: l'utilisateur a ouvert l'article.
# LIKE: l'utilisateur a aimé l'article
# COMMENT CREATED: l'utilisateur a commenté l'article.
# FOLLOW: l'utilisateur souhaite recevoir des notificcations en cas de nouvel article.
# BOOKMARK: L'utilisateur a mis l'article en signet pour un retour facile à l'avenir.

interactions_df = pd.read_csv('users_interactions.csv')
interactions_df.head(10)
interactions_df.tail(5)

# Transfert de données (Data Munging)
# Comme il existe différents types d'interactions, nous les associons à un poids ou à une force,
# en supposant que, par exemple, un commentaire dans un article indique un intérêt 
# plus élevé de l'utilisateur pour l'article qu'un article similaire ou qu'une simple vue.

event_type_strength = {
   'VIEW': 1.0,
   'LIKE': 2.0, 
   'BOOKMARK': 2.5, 
   'FOLLOW': 3.0,
   'COMMENT CREATED': 4.0,  
}

interactions_df['eventStrength'] = interactions_df['eventType'].apply(lambda x: event_type_strength[x])

# Les systèmes de recommandation ont un problème connu sous le nom de démarrage à froid des utilisateurs,
# dans lequel il est difficile de fournir des recommandations personnalisées aux utilisateurs n'ayant pas ou très peu d'articles consommés,
# en raison du manque d'informations pour modéliser leurs préférences.
# Pour cette raison, nous ne conservons dans l'ensemble de données que les utilisateurs ayant au moins 5 interactions.

users_interactions_count_df = interactions_df.groupby(['personId', 'contentId']).size().groupby('personId').size()
print('# le nombre dutilisateurs: %d' % len(users_interactions_count_df))

users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['personId']]
print('# utilisateurs avec au moins 5 interactions: %d' % len(users_with_enough_interactions_df))

print('# le nombre dinteractions: %d' % 
      len(interactions_df))

interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df, 
               how = 'right',
               left_on = 'personId',
               right_on = 'personId')

print('# utilisateurs avec au moins 5 interactions: %d' %
      len(interactions_from_selected_users_df))


# Dans Deskdrop, les utilisateurs sont autorisés à consulter un article plusieurs fois et à interagir avec eux de différentes manières (par exemple, liker ou commenter).
# Ainsi, pour modéliser l'intérêt de l'utilisateur sur un article donné,
# nous agrégons toutes les interactions que l'utilisateur a effectuées dans un article par une somme pondérée de la force du type d'interaction et appliquons une transformation de journal pour lisser la distribution.

def smooth_user_preference(x):
    return math.log(1+x, 2)
    
interactions_full_df = interactions_from_selected_users_df \
                    .groupby(['personId', 'contentId'])['eventStrength'].sum() \
                    .apply(smooth_user_preference).reset_index()
                    
print('Nombre dutilisateurs unique ainsi que leur interactions: %d'
      % len(interactions_full_df))

interactions_full_df.head(10)


# Évaluation de notre modèle
# L'évaluation est importante pour les projets d'apprentissage automatique,
# car elle permet de comparer objectivement différents algorithmes et choix d'hyperparamètres pour les modèles.

# Un aspect clé de l'évaluation consiste à s'assurer que le modèle formé se généralise pour les données sur lesquelles il n'a pas été formé,
# en utilisant des techniques de validation croisée. Nous utilisons ici une approche de validation croisée simple appelée holdout,
# dans laquelle un échantillon de données aléatoires (20% dans ce cas) est mis de côté dans le processus de formation, et exclusivement utilisé pour l'évaluation.

# Toutes les mesures d'évaluation rapportées ici sont calculées à l'aide de l'ensemble de test. 

# Une approche d'évaluation plus robuste pourrait être de diviser les trains et les ensembles de tests par une date de référence,
# où l'ensemble de trains est composé de toutes les interactions avant cette date, et l'ensemble de tests sont des interactions après cette date.

# Par souci de simplicité, nous avons choisi la première approche aléatoire pour ce bloc-notes,
# mais vous souhaiterez peut-être essayer la deuxième approche pour mieux simuler les performances du recsys en production en prédisant les interactions "futures" des utilisateurs.

# Division du jeu de données en données train et données test
interactions_train_df, interactions_test_df = train_test_split(interactions_full_df, stratify=interactions_full_df['personId'], test_size=0.20, random_state=42)

print('# interactions dans le Train set: %d' % len(interactions_train_df))
print('# interactions dans le Test set: %d' % len(interactions_test_df))


# Dans les systèmes de recommandation, il existe un ensemble de mesures couramment utilisées pour l'évaluation.
# Nous avons choisi de travailler avec des métriques de précision Top-N,
# qui évaluent la précision des principales recommandations fournies à un utilisateur,
# en les comparant aux éléments auquel l'utilisateur a réellement interagis avec dans l'ensemble de tests.

# Cette méthode d'évaluation fonctionne comme suit:

# Pour chaque utilisateur
# Pour chaque élément, l'utilisateur a interagi dans l'ensemble de test
# Goûtez à 100 autres éléments que l'utilisateur n'a jamais interagis.

# Ici, nous supposons naïvement que ces éléments non interactifs ne sont pas pertinents pour l'utilisateur,
# ce qui pourrait ne pas être vrai, car l'utilisateur peut simplement ne pas être au courant de ces éléments non interactifs.
# Mais gardons cette hypothèse.

# Demandez au modèle recommandeur de produire une liste classée des éléments recommandés,
# à partir d'un ensemble composé d'un élément interactif et des 100 éléments non interactifs («non pertinents!)

# Calculez les mesures de précision Top-N pour cet utilisateur et l'élément interagi à partir de la liste classée des recommandations
# Agréger les métriques de précision Top-N globales

# La mesure de précision Top-N choisie était Recall @ N qui évalue si l'élément interagi fait partie des N meilleurs éléments (hit) dans la liste classée de 101 recommandations pour un utilisateur.
# D'autres mesures de classement populaires sont NDCG @ N et MAP @ N,
# dont le calcul du score prend en compte la position de l'élément pertinent dans la liste classée (valeur maximale
# si l'élément pertinent se trouve en première position).

# Indexation par personId pour accélérer les recherches lors de l'évaluation
interactions_full_indexed_df = interactions_full_df.set_index('personId')
# On l'applique dans le training set et le test set
interactions_train_indexed_df = interactions_train_df.set_index('personId')
interactions_test_indexed_df = interactions_test_df.set_index('personId')

# On crée une fonction "Obtenir un article auquel on interagi"
def Obtenir_des_elements_interagis(person_id, interactions_df):
    # Obtenir les données de l'utilisateur et fusionnez les informations sur le film.
    interacted_items = interactions_df.loc[person_id]['contentId']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

# Constantes de métriques de précision Top-N
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100

class EvaluateurModele:


    def obtenir_un_echantillon_elements_non_interagis(self, person_id, sample_size, seed=45):
        interacted_items = Obtenir_des_elements_interagis(person_id, interactions_full_indexed_df)
        all_items = set(articles_df['contentId'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    def verifier_hit_top_n(self, item_id, recommended_items, topn):        
            try:
                index = next(i for i, c in enumerate(recommended_items) if c == item_id)
            except:
                index = -1
            hit = int(index in range(0, topn))
            return hit, index

    def evaluer_modele_pour_utilisateur(self, model, person_id):
        
        # Obtenir des articles dans le test set
        interacted_values_testset = interactions_test_indexed_df.loc[person_id]
        if type(interacted_values_testset['contentId']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['contentId'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['contentId'])])  
        interacted_items_count_testset = len(person_interacted_items_testset) 

        # Obtenir une liste de recommandations classée à partir d'un modèle pour un utilisateur donné
        person_recs_df = model.recommendations_elements(person_id, items_to_ignore=Obtenir_des_elements_interagis(person_id, interactions_train_indexed_df), 
                                               topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0
        
        # Pour chaque élément, l'utilisateur a interagi dans l'ensemble de test
        for item_id in person_interacted_items_testset:
            # Obtenir un échantillon aléatoire (100) éléments que l'utilisateur n'a pas interagis
            # (pour représenter les éléments qui ne sont pas considérés comme pertinents pour l'utilisateur)
            non_interacted_items_sample = self.obtenir_un_echantillon_elements_non_interagis(person_id, 
                                                                          sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, 
                                                                          seed=item_id%(2**32))

            # Combiner l'élément interactif actuel avec les 100 éléments aléatoires
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            # Filtrer uniquement les recommandations qui sont soit l'élément interactif, soit un échantillon aléatoire de 100 éléments non interactifs
            valid_recs_df = person_recs_df[person_recs_df['contentId'].isin(items_to_filter_recs)]                    
            valid_recs = valid_recs_df['contentId'].values
           
            # Vérifier si l'élément interactif actuel fait partie des Top N éléments recommandés
            hit_at_5, index_at_5 = self.verifier_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self.verifier_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        # Le rappel (Recall) est le taux des éléments interagis qui sont classés parmi les Top N éléments recommandés,
        # lorsqu'il est mélangé avec un ensemble d'éléments non pertinents
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        person_metrics = {'hits@5_count':hits_at_5_count, 
                          'hits@10_count':hits_at_10_count, 
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}
        return person_metrics

    def evaluer_modele(self, model):
        # print('Exécution de l'évaluation pour les utilisateurs')
        people_metrics = []
        for idx, person_id in enumerate(list(interactions_test_indexed_df.index.unique().values)):
            # if idx % 100 == 0 and idx > 0:
            #    print('%d users processed' % idx)
            person_metrics = self.evaluer_modele_pour_utilisateur(model, person_id)
            person_metrics['_person_id'] = person_id
            people_metrics.append(person_metrics)
        print('%d users processed' % idx)
        
        # Voir un tableau des résultats
        detailed_results_df = pd.DataFrame(people_metrics) \
                            .sort_values('interacted_count', ascending=False)
        
        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        
        global_metrics = {'modelName': model.Obtenir_nom_du_modele(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10}    
        return global_metrics, detailed_results_df
    
Evaluateur_modele = EvaluateurModele()


# Modèle de popularité
# Une approche de référence courante (et généralement difficile à battre) est le modèle de popularité.
# Ce modèle n'est pas réellement personnalisé - il recommande simplement à un utilisateur les articles les plus populaires que l'utilisateur n'a pas consommés auparavant.

# Comme la popularité explique la «sagesse des foules», elle fournit généralement de bonnes recommandations, généralement intéressantes pour la plupart des gens.
# L'objectif principal d'un système de recommandation est de tirer parti des éléments à longue queue pour les utilisateurs ayant des intérêts très spécifiques,
# ce qui va bien au-delà de cette simple technique.

# Calcule les éléments les plus populaires
item_popularity_df = interactions_full_df.groupby('contentId')['eventStrength'].sum().sort_values(ascending=False).reset_index()
item_popularity_df.head(10)

# Créons une fonction pour recommander les articles les plus populaires
class Recommendation_populaire:
    
    MODEL_NAME = 'Popularite'
    
    def __init__(self, popularity_df, items_df=None):
        self.popularity_df = popularity_df
        self.items_df = items_df
        
    def Obtenir_nom_du_modele(self):
        return self.MODEL_NAME
        
    def recommendations_elements(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        
        # Recommandez les articles les plus populaires que l'utilisateur n'a pas encore vus.
        recommendations_df = self.popularity_df[~self.popularity_df['contentId'].isin(items_to_ignore)] \
                               .sort_values('eventStrength', ascending = False) \
                               .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['eventStrength', 'contentId', 'title', 'url', 'lang']]


        return recommendations_df
    
modele_de_popularite = Recommendation_populaire(item_popularity_df, articles_df)


# Ici, nous effectuons l'évaluation du modèle de popularité, selon la méthode décrite ci-dessus.
# Il a atteint le rappel @ 5 de 0,2417, ce qui signifie qu'environ 24% des éléments interagis dans l'ensemble de test
# ont été classés par modèle de popularité parmi les 5 premiers éléments (à partir de listes avec 100 éléments aléatoires).

# Et le rappel @ 10 était encore plus élevé (37%), comme prévu.
# Il pourrait vous surprendre que les modèles de popularité soient généralement aussi performants!

print('Évaluation du modèle de recommandation de popularité...')
pop_global_metrics, pop_detailed_results_df = Evaluateur_modele.evaluer_modele(modele_de_popularite)

print('\nGlobal metrics:\n%s' % pop_global_metrics)
pop_detailed_results_df.head(20)


##########  Modèle de filtrage basé sur le contenu 
# Les approches de filtrage basées sur le contenu exploitent la description ou les attributs des éléments que l'utilisateur a interagis pour recommander des éléments similaires.
# Cela ne dépend que des choix précédents de l'utilisateur, ce qui rend cette méthode robuste pour éviter le problème de démarrage à froid.
# Pour les éléments textuels, comme les articles, les actualités et les livres, il est simple d'utiliser le texte brut pour créer des profils d'élément et des profils utilisateur.
# Ici, nous utilisons une technique très populaire de recherche d'informations (moteurs de recherche) nommée TF-IDF.
# Cette technique convertit le texte non structuré en une structure vectorielle, où chaque mot est représenté par une position dans le vecteur,
# et la valeur mesure la pertinence d'un mot donné pour un article. Comme tous les éléments seront représentés dans le même modèle d'espace vectoriel,
# il s'agit de calculer la similitude entre les articles.

# Ignorer les mots vides ou stopwords (mots sans sémantique) de l'anglais et du portugais (car nous avons un corpus avec des langues mixtes)
stopwords_list = stopwords.words('english') + stopwords.words('portuguese')

# Entrainer un modèle dont la taille des vecteurs est de 5000
# composé des principaux unigrammes et bigrammes trouvés dans le corpus, en ignorant les mots vides
vectorizer = TfidfVectorizer(analyzer='word',
                     ngram_range=(1, 2),
                     min_df=0.003,
                     max_df=0.5,
                     max_features=5000,
                     stop_words=stopwords_list)

item_ids = articles_df['contentId'].tolist()

tfidf_matrix = vectorizer.fit_transform(articles_df['title'] + "" + articles_df['text'])
tfidf_feature_names = vectorizer.get_feature_names()
tfidf_matrix


# Pour modéliser le profil utilisateur, 
# nous prenons tous les types d'élément avec lesquels l'utilisateur a interagis et faisons la moyenne.
# La moyenne est pondérée par la force d'interaction, en d'autres termes,

# les articles que l'utilisateur a le plus interagis (par exemple, aimés ou commentés)
# auront une force plus élevée dans le profil utilisateur final.

interactions_indexed_df = interactions_train_df[interactions_train_df['contentId'] \
                                                   .isin(articles_df['contentId'])].set_index('personId')
class creation_profil_utilisateur:
    
    def obtenir_le_profil_darticle(item_id):
        idx = item_ids.index(item_id)
        item_profile = tfidf_matrix[idx:idx+1]
        return item_profile

    def obtenir_des_profils_darticles(ids):
        item_profiles_list = [obtenir_le_profil_darticle(x) for x in ids]
        item_profiles = scipy.sparse.vstack(item_profiles_list)
        return item_profiles

    def creer_un_profil_dutilisateur(person_id, interactions_indexed_df):
        interactions_person_df = interactions_indexed_df.loc[person_id]
        user_item_profiles = obtenir_des_profils_darticles(interactions_person_df['contentId'])
        user_item_strengths = np.array(interactions_person_df['eventStrength']).reshape(-1,1)
        
        # Moyenne pondérée des profils d'articles par la force des interactions
        user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(user_item_strengths)
        user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
        return user_profile_norm

    def creer_des_profils_dutilisateurs():
        interactions_indexed_df = interactions_train_df[interactions_train_df['contentId'] \
                                                   .isin(articles_df['contentId'])].set_index('personId')
        user_profiles = {}
        for person_id in interactions_indexed_df.index.unique():
            user_profiles[person_id] = creer_un_profil_dutilisateur(person_id, interactions_indexed_df)
        return user_profiles

profil_utilisateur = creer_des_profils_dutilisateurs()
len(profil_utilisateur)

# Jetons un coup d'oeil dans le profil. Il s'agit d'un vecteur unitaire de 5000 longueurs.
# La valeur de chaque position représente la pertinence d'un jeton (unigramme ou bigramme) pour moi.
# En regardant mon profil, il semble que les principaux jetons pertinents représentent vraiment mes intérêts professionnels dans l'apprentissage automatique,
# apprentissage en profondeur, intelligence artificielle et plateforme google cloud!
# Nous pouvons donc nous attendre à de bonnes recommandations ici!

mon_profile = profil_utilisateur[-1479311724257856983]
print(mon_profile.shape)

pd.DataFrame(sorted(zip(tfidf_feature_names,
                        profil_utilisateur[-1479311724257856983].flatten().tolist()),
                    key=lambda x: -x[1])[:20], columns=['token', 'relevance'])

# Fonction recommandeur de base de contenu
class Systeme_Recommandeur_base_de_contenu:
    
    MODEL_NAME = 'Base-de-contenu'
    
    def __init__(self, items_df=None):
        self.item_ids = item_ids
        self.items_df = items_df
        
    def Obtenir_nom_du_modele(self):
        return self.MODEL_NAME
        
    def Obtenir_un_element_similaire_au_profil_utilisateur(self, person_id, topn=1000):
        
        # Calcule la similitude cosinus entre le profil utilisateur et tous les profils d'élément
        cosine_similarities = cosine_similarity(profil_utilisateur[person_id], tfidf_matrix)
        
        # Obtenir les meilleurs articles similaires
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        
        # Trier les articles similaires par similitude
        similar_items = sorted([(item_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items
        
    def recommendations_elements(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        similar_items = self.Obtenir_un_element_similaire_au_profil_utilisateur(user_id)
        
        # Ignore les éléments auquel l'utilisateur a déjà interagis
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['contentId', 'recStrength']) \
                                    .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['recStrength', 'contentId', 'title', 'url', 'lang']]


        return recommendations_df

modele_de_recommandation_basé_sur_le_contenu = Systeme_Recommandeur_base_de_contenu(articles_df )

# Avec des recommandations personnalisées de modèle de filtrage basé sur le contenu,
# nous avons un rappel @ 5 à environ 0,162, ce qui signifie qu'environ 16% des éléments qui interagissent dans l'ensemble de test
# ont été classés par ce modèle parmi les 5 premiers éléments (à partir de listes avec 100 éléments aléatoires ).
# Et Recall @ 10 était de 0,261 (52%).
# Les performances plus faibles du modèle basé sur le contenu par rapport au modèle de popularité
# peuvent indiquer que les utilisateurs ne sont pas si fixes dans un contenu très similaire à leurs lectures précédentes.

print('Évaluation du modèle de filtrage basé sur le contenu...')
cb_global_metrics, cb_detailed_results_df = Evaluateur_modele.evaluer_modele(Systeme_Recommandeur_base_de_contenu)

print('\nMesures globales:\n%s' % cb_global_metrics)
cb_detailed_results_df.head(10)

###########################################################################################################################
# Modèle de filtrage collaboratif
# Le filtrage collaboratif (CF) a deux stratégies de mise en œuvre principales:

# Basé sur la mémoire: cette approche utilise la mémoire des interactions des utilisateurs précédents
# pour calculer les similitudes des utilisateurs en fonction des éléments qu'ils ont interagis (approche basée sur l'utilisateur)
# ou calculer les similitudes des éléments en fonction des utilisateurs qui ont interagi avec eux (approche basée sur les éléments).
# Un exemple typique de cette approche est le CF basé sur le voisinage utilisateur, dans lequel les N premiers utilisateurs similaires
# (généralement calculés en utilisant la corrélation de Pearson) pour un utilisateur sont sélectionnés et utilisés pour recommander des éléments
# que ces utilisateurs similaires aimaient, mais l'utilisateur actuel n'a pas interagi encore. Cette approche est très simple à mettre en œuvre,
# mais ne s'adapte généralement pas bien à de nombreux utilisateurs. Une belle implémentation Python de cette approche est disponible dans Crab.

# Basé sur des modèles: cette approche consiste à développer des modèles à l'aide de différents algorithmes d'apprentissage automatique pour recommander des éléments aux utilisateurs.
# Il existe de nombreux algorithmes CF basés sur des modèles, tels que les réseaux de neurones, les réseaux bayésiens, les modèles de clustering et les modèles de facteurs latents
# tels que la décomposition en valeurs singulières (SVD) et l'analyse sémantique latente probabiliste.


# Factorisation matricielle
# Les modèles de facteurs latents compressent la matrice utilisateur-élément en une représentation de faible dimension en termes de facteurs latents. 
# Un avantage d'utiliser cette approche est qu'au lieu d'avoir une matrice de haute dimension contenant un nombre abondant de valeurs manquantes,
# nous aurons affaire à une matrice beaucoup plus petite dans un espace de dimension inférieure.
# Une présentation réduite pourrait être utilisée pour les algorithmes de voisinage basés sur l'utilisateur ou sur les objets qui sont présentés dans la section précédente.
# Il y a plusieurs avantages avec ce paradigme. Il gère mieux la rareté de la matrice d'origine que celle basée sur la mémoire.
# La comparaison de la similitude sur la matrice résultante est également beaucoup plus évolutive, en particulier pour les grands ensembles de données clairsemés.
# Ici, nous utilisons un modèle de facteur latent populaire nommé Singular Value Decomposition (SVD).
# Il existe d'autres cadres de factorisation matricielle plus spécifiques à CF que vous pourriez essayer, comme surprise, mrec ou python-recsys.
# Nous avons choisi une implémentation SciPy de SVD car elle est disponible sur les noyaux Kaggle. P.s. Voir un exemple de SVD sur un jeu de données de films dans cet article de blog.
# Une décision importante est le nombre de facteurs pour factoriser la matrice utilisateur-article. Plus le nombre de facteurs est élevé,
# plus la factorisation est précise dans les reconstructions matricielles originales.
# Par conséquent, si le modèle est autorisé à mémoriser trop de détails de la matrice d'origine, il peut ne pas généraliser correctement pour les données sur lesquelles il n'a pas été formé.
# La réduction du nombre de facteurs augmente la généralisation du modèle.

# Création d'un tableau croisé dynamique avec des utilisateurs en lignes et des éléments en colonnes
users_items_pivot_matrix_df = interactions_train_df.pivot(index='personId', 
                                                          columns='contentId', 
                                                          values='eventStrength').fillna(0)

users_items_pivot_matrix_df.head(10)

users_items_pivot_matrix = users_items_pivot_matrix_df.values

users_items_pivot_matrix[:10]

users_ids = list(users_items_pivot_matrix_df.index)

users_ids[:10]

users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)

users_items_pivot_sparse_matrix

# Nombre de facteurs pour factoriser la matrice utilisateur-élément.
NUMBER_OF_FACTORS_MF = 15

# Effectue la factorisation matricielle de la matrice d'élément utilisateur d'origine
U, sigma, Vt = svds(users_items_pivot_sparse_matrix, k = NUMBER_OF_FACTORS_MF)

U.shape
Vt.shape

sigma = np.diag(sigma)
sigma.shape

# Après la factorisation, nous essayons de reconstruire la matrice d'origine en multipliant ses facteurs.
# La matrice résultante n'est plus rare. Il a été généré des prédictions pour les éléments que l'utilisateur n'a pas encor eu d'interaction,
# et que nous exploiterons pour des recommandations.

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
all_user_predicted_ratings

all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / (all_user_predicted_ratings.max() - all_user_predicted_ratings.min())
all_user_predicted_ratings_norm

# Conversion de la matrice reconstruite en une trame de données Pandas
cf_preds_df = pd.DataFrame(all_user_predicted_ratings_norm, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()
cf_preds_df.head(10)

len(cf_preds_df.columns)

class Systeme_recommandeur_Filtrage_collaboratif:
    
    MODEL_NAME = 'Filtrage collaboratif'
    
    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df
        
    def Obtenir_nom_du_modele(self):
        return self.MODEL_NAME
        
    def recommendations_elements(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        
        # Obtenez et triez les prédictions de l'utilisateur
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
                                    .reset_index().rename(columns={user_id: 'recStrength'})

        # Recommandez les films les mieux notés que l'utilisateur n'a pas encore vus.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['contentId'].isin(items_to_ignore)] \
                               .sort_values('recStrength', ascending = False) \
                               .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['recStrength', 'contentId', 'title', 'url', 'lang']]


        return recommendations_df
    
recommandeur_cf = Systeme_recommandeur_Filtrage_collaboratif(cf_preds_df, articles_df)


# En évaluant le modèle de filtrage collaboratif (factorisation matricielle SVD),
# nous observons que nous avons obtenu des valeurs Recall @ 5 (33%) et Recall @ 10 (46%),
# beaucoup plus élevées que le modèle de popularité et le modèle basé sur le contenu.

print('Evaluating Collaborative Filtering (SVD Matrix Factorization) model...')
cf_global_metrics, cf_detailed_results_df = Evaluateur_modele.evaluer_modele(recommandeur_cf)

print('\nGlobal metrics:\n%s' % cf_global_metrics)
cf_detailed_results_df.head(10)


###########################################################################################
#################### Recommandation hybride ###############################################
##########################################################################################
# Et si nous combinions les approches de filtrage collaboratif et de filtrage basé sur le contenu?
# Cela nous fournirait-il des recommandations plus précises?
# En fait, les méthodes hybrides ont donné de meilleurs résultats que les approches individuelles dans de nombreuses études et ont été largement utilisées par les chercheurs et les praticiens.
# Construisons une méthode d'hybridation simple, comme un ensemble qui prend la moyenne pondérée des scores CF normalisés avec les scores basés sur le contenu,
# et le classement par score résultant. Dans ce cas, comme le modèle CF est beaucoup plus précis que le modèle CB,
# les poids pour les modèles CF et CB sont respectivement de 100,0 et 1,0.

class Systeme_Recommandeur_hybride:
    
    MODEL_NAME = 'Recommandeur Hybride'
    
    def __init__(self, cb_rec_model, cf_rec_model, items_df, cb_ensemble_weight=1.0, cf_ensemble_weight=1.0):
        self.cb_rec_model = cb_rec_model
        self.cf_rec_model = cf_rec_model
        self.cb_ensemble_weight = cb_ensemble_weight
        self.cf_ensemble_weight = cf_ensemble_weight
        self.items_df = items_df
        
    def Obtenir_nom_du_modele(self):
        return self.MODEL_NAME
        
    def recommendations_elements(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        
        # Obtenir les 1000 premières recommandations de filtrage basé sur le contenu
        cb_recs_df = self.cb_rec_model.recommendations_elements(user_id, items_to_ignore=items_to_ignore, verbose=verbose,
                                                           topn=1000).rename(columns={'recStrength': 'recStrengthCB'})
        
        # Obtenir les 1000 premières recommandations de filtrage collaboratif
        cf_recs_df = self.cf_rec_model.recommendations_elements(user_id, items_to_ignore=items_to_ignore, verbose=verbose, 
                                                           topn=1000).rename(columns={'recStrength': 'recStrengthCF'})
        
        # Combiner les résultats par contentId
        recs_df = cb_recs_df.merge(cf_recs_df,
                                   how = 'outer', 
                                   left_on = 'contentId', 
                                   right_on = 'contentId').fillna(0.0)
        
        # Calcul d'un score de recommandation hybride basé sur les scores CF et CB
        # recs_df['recStrengthHybrid'] = recs_df['recStrengthCB'] * recs_df['recStrengthCF'] 
        recs_df['recStrengthHybrid'] = (recs_df['recStrengthCB'] * self.cb_ensemble_weight) \
                                     + (recs_df['recStrengthCF'] * self.cf_ensemble_weight)
        
        # Tri des recommandations par score hybride
        recommendations_df = recs_df.sort_values('recStrengthHybrid', ascending=False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['recStrengthHybrid', 'contentId', 'title', 'url', 'lang']]


        return recommendations_df
    
recommandeur_hybride = Systeme_Recommandeur_hybride(modele_de_recommandation_basé_sur_le_contenu,
                                                        recommandeur_cf,
                                                        articles_df,
                                                        cb_ensemble_weight=1.0,
                                                        cf_ensemble_weight=100.0)

# Nous avons un nouveau champion!
# Notre approche hybride simple surpasse le filtrage basé sur le contenu avec sa combinaison avec le filtrage collaboratif.
# Nous avons maintenant un rappel à 5 ​​de 34,2% et un rappel à 10 de 47,9%

print('Evaluating Hybrid model...')
hybrid_global_metrics, hybrid_detailed_results_df = Evaluateur_modele.evaluer_modele(recommandeur_hybride)

print('\nGlobal metrics:\n%s' % hybrid_global_metrics)
hybrid_detailed_results_df.head(10)


###########################################################################################
#### Comparaison des méthodes
global_metrics_df = pd.DataFrame ([cb_global_metrics, pop_global_metrics, cf_global_metrics, hybrid_global_metrics]) \
    .set_index ('modelName')
global_metrics_df


%matplotlib inline
ax = global_metrics_df.transpose().plot(kind='bar', figsize=(15,8))
for p in ax.patches: ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
majorFormatter=FormatStrFormatter('%8.10g')
ax.xaxis.set_major_formation(majorFormatter)
ax.yaxis.set_major_formation(majorFormatter)


### Essai
# Testons le meilleur modèle (hybride) pour mon utilisateur.

def inspect_interactions(person_id, test_set=True):
    if test_set:
        interactions_df = interactions_test_indexed_df
    else:
        interactions_df = interactions_train_indexed_df
    return interactions_df.loc[person_id].merge(articles_df, how = 'left', 
                                                      left_on = 'contentId', 
                                                      right_on = 'contentId') \
                          .sort_values('eventStrength', ascending = False)[['eventStrength', 
                                                                          'contentId',
                                                                          'title', 'url', 'lang']]

# Ici, nous voyons certains articles que j'ai interagis dans Deskdrop à partir d'un entrainement.
# On peut facilement constater que mes principaux intérêts sont l'apprentissage automatique, l'apprentissage profond,
# l'intelligence artificielle et la plate-forme Google Cloud.
                          
inspect_interactions(-1479311724257856983, test_set=False).head(20)

# Les recommandations correspondent vraiment à mes intérêts, car je les lirais toutes!

recommandeur_hybride.recommendations_elements(-1479311724257856983, topn=20, verbose=True)


# Dans ce cahier, nous avons exploré et comparé les principales techniques des systèmes de recommandation sur le jeu de données CI&T Deskdrop
# On a pu observer que pour la recommandation d'articles, le filtrage basé sur le contenu et une méthode hybride fonctionnaient mieux que le filtrage collaboratif seul.

# Il y a une grande marge d'amélioration des résultats. 

# Voici quelques conseils:
# Dans cet exemple, nous avons complètement ignoré l'heure, étant donné que tous les articles étaient disponibles pour être recommandés aux utilisateurs à tout moment.

# Une meilleure approche serait de filtrer uniquement les articles disponibles pour les utilisateurs à un moment donné.
# Vous pouvez tirer parti des informations contextuelles disponibles pour modéliser les préférences des utilisateurs à travers le temps (période de la journée,

# jour de la semaine, mois), l'emplacement (pays et état / district) et les appareils (navigateur, application native mobile).
# Ces informations contextuelles peuvent être facilement incorporées dans les modèles Learn-to-Rank (comme les arbres de décision XGBoost Gradient Boosting avec objectif de classement),
# les modèles logistiques (avec les caractéristiques catégorielles One-Hot encoded ou Feature Hashed) et les modèles Wide & Deep, qui sont implémentés dans TensorFlow.

# Jetez un œil dans le résumé de ma solution partagée pour la compétition Outbrain Click Prediction.

# Ces techniques de base ont été utilisées à des fins didactiques. Il existe des techniques plus avancées dans la communauté de recherche RecSys,
# spécialement des modèles avancés de factorisation matricielle et d'apprentissage profond.

# Vous pouvez en savoir plus sur les méthodes de pointe publiées dans Recommender Systems sur la conférence ACM RecSys.
# Si vous êtes plus un praticien qu'un chercheur, vous pouvez essayer certains cadres de filtrage collaboratif dans cet ensemble de données,
# comme surprise, mrec, python-recsys et Spark ALS Matrix Factorization (implémentation distribuée pour les grands ensembles de données).

# Jetez un oeil dans cette présentation où je décris un système de recommandation de production,
# axé sur les techniques de filtrage basé sur le contenu et de modélisation de sujet.



                   
