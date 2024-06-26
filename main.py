###################################################################################
# Itay Rimmler
# itay.rimmler@gmail.com
# +972-50-331-4515
# Version 1.0
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

# Turning the claims from the relevant rows to a list...
claims = list(claims.iloc[:, 1])

# Using the model on each claim...
docs = [nlp(claim) for claim in claims]

# Printing results...
for doc in docs:
    print(f"OG text: {doc.text}")
    print(f"Coreference clusters: {doc._.coref_clusters}\n")

# Impressive. Let's see what happens if all the claims are in one sentence...
all_claims = " ".join(claims)
doc = nlp(all_claims)

# Printing results...
print(f"Original Text: {doc.text}")
print(f"Coreference Clusters: {doc._.coref_clusters}")
