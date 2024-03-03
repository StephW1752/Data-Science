# Import spacy and download en_core_web_sm
import spacy

nlp = spacy.load('en_core_web_sm')

# Create list of garden path sentences, then add given garden path sentences to list.
gardenpathSentences = ['The florist sent the flowers was pleased.', 'The man who hunts ducks out on weekends.', 'Helen is expecting tomorrow to be a bad day.']
gardenpathSentences.extend(['Mary gave the child a Band-Aid.', 'That Jill is never here hurts.', 'The cotton clothing is made of grows in Mississipi.'])
print(gardenpathSentences)

# Loop through sentences in the list and tokenize each element of each list.
# Then use named entity recognition to classify named entities.
for i in range(0, len(gardenpathSentences)):
    doc = nlp(gardenpathSentences[i])
    print(doc)
    tokenized_doc = ([token.orth_ for token in doc if not token.is_punct | token.is_space])
    print(tokenized_doc)
    nlp_doc = nlp(doc)
    print([(i, i.label_, i.label) for i in nlp_doc.ents])

# Use spacy.explain to explain the entities found above.
print(spacy.explain("GPE"))
print(spacy.explain("DATE"))

# GPE entity explained as Countries, cities and states- yes this makes sense in terms of Mississipi, an American state.
# DATE entity explained as absolute or relative dates or periods- this makes sense in relation to 'tomorrow', but not
# in relation to 'a bad day' as this is not an absolute or relative date. 