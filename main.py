###################################################################################
# Itay Rimmler
# itay.rimmler@gmail.com
# +972-50-331-4515
# Version 1.1
###################################################################################


# Research has shown me that I should use a certain model with a pipeline. So I'm gonna do that.

# We're importing spacy, to load an NLP model
import spacy

# Loading model...
nlp = spacy.load('en_core_web_sm')

# We're importing neuralcoref, to load a pipeline for our NLP model
import neuralcoref

# Creating pipeline...
coref = neuralcoref.NeuralCoref(nlp.vocab)

# Adding pipeline to the model...
nlp.add_pipe(coref, name='neuralcoref')

# Importing pandas, because I saved my claims in a csv format
import pandas as pd

# Reading csv...
claims = pd.read_csv("Claims/claims1.csv")

# Getting rid of irrelevant rows that I marked as such...
claims = claims.drop(claims[claims['Patent'] == 'ignore'].index)

# We're importing all the preprocessing functions
from preprocess import *

# Preprocessing the claims row by row...
for i in range(claims.shape[0]):
    claims.iloc[i, :] = split(clean(claims.iloc[i, :]))

# Turning the claims from the relevant rows to a list...
claims = list(claims['Claim'])

# Joining the claims and using the model on them...
claims = " ".join([" ".join(claim) for claim in claims])
docs = nlp(claims)

# Printing results...
print(f"Coreference clusters: {docs._.coref_clusters}")
