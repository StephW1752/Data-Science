import spacy
import pandas as pd
nlp= spacy.load('en_core_web_md')

word1 = nlp("cat")
word2 = nlp("monley")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

tokens = nlp('cat apple monkey banana')

# First compare token1 to all the other tokens in string, then do same for token 2 and repeat cycle.
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

# Monkey and cat have relatively high similarity as are both animals.
# Banana and apple have relatively high similarity as both are fruit.
# Monkey and banana have much higher similarity than monkey and apple- 
# similarity assumably relating to monkeys eating bananas.
# Apple and monkey has similar similarity level to banana and cat- both animal and fruit.
# Banana and cat similarity much lower than banana and monkey.
        
tokens_2 = nlp('car plane cloud cupcake')
for token1 in tokens_2:
    for token2 in tokens_2:
        print(token1.text, token2.text, token1.similarity(token2))

# Car and plane relatively similar as both are vehicles.
# Plane and cloud higher similarity than car and cloud, could be related as seen together.
# Cupcake has low similarity to all other words as expected.
        
# When run with en_core_web_sm similarities of all pairs appear to be higher.
# Cloud and car has a relatively high similarity score as well as cloud and plane.
# Plane and cupcake have a much higher similarity score than with the medium model (0.12 vs 0.42)
        

sentence_to_compare = 'Why is my cat on the car'

sentences = ["Where did my dog go", 
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + "-" + str(similarity))


# PRACTICAL TASK 2

planet_hulk = ('''Will he save their world or destroy it? When the hulk becomes too 
dangerous for the earth, the Illuminati trick Hulk into a shuttle and launch him into
space to a planet where the hulk can live in peace. Unfortunately. Hulk lands on the
planet Sakaar where he is sold into slavery and trained as a gladiator.''')

# Read in text file, separator is :, use header=None so 1st row isn't mistaken for headings.
df = pd.read_csv('movies.txt', sep=":", header=None)

# Assign names to columns that we will need for comparison and results output.
movie_desc = df.iloc[:,1]
movie_no= df.iloc[:,0]

# Lemmatize data to preprocess for semantic similarity comparison.
def preprocess(text):
    doc = nlp(text)
    return ' '.join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])

# Convert preprocessed text to a spaCy doc.
def get_doc(text):
    return nlp(text)

# Create new columns in data set to hold the preprocessing description and spaCy doc objects.
df['preprocessed']= movie_desc.apply(preprocess) 
df['doc']=df['preprocessed'].apply(get_doc)

# Define column needed for semantic similarity comaprison.
movie_doc = (df.iloc[:,3])

# Create function that takes in a movie description and returns the most similar.
def recommend(planet_hulk, n_recommendations=1): 
    # Preprocess and convert to spaCy doc the inputed movie description.
    preprocessed_input = preprocess(planet_hulk)
    input_doc = get_doc(preprocessed_input)
    # Calculate similarity of inputed film's spaCy doc against spaCy doc for all movies in table.
    # Add another column to dataframe to store this similarity score.
    df['similarity'] = df['doc'].apply(lambda doc: input_doc.similarity(doc))
    # Select the row with the largest of the values in similarity column.
    recommendation = df.nlargest(1, 'similarity').head(1)
    # From this row select only the movie name from the first column in row. 
    # Strip symbols so it can be formatted into readable sentence.
    movie_recommendation = str(recommendation.iloc[:,0].values).strip('[,],\'')
    # From the same row select only the similarity score in the last column of the row.
    # Strip symbols so it can be formatted into a readable sentence.
    similarity_score = str(recommendation.iloc[:,4].values).strip('[,]')
    # Format the movie name and similarity score into outputed printed sentence.
    print(f"The most similar movie is {movie_recommendation} with a semantic similarity of {similarity_score}.")
# Apply recommend function to planet_hulk movie description.
recommend(planet_hulk)


#example
reader_preferences = {
    'Politics':0.8,
    'Technology':0.6,
    'Science':0.7
}

def byte_match(recommendation scores):
    sorted_recommendations = sorted(recommendation_scores.items(),
                                    key=lambda x : x[1], reverse=True)
#calling .items on a dictionary gets the keys and values, lambda makes sure sorts by scores
    top_recommendation = sorted_recommendations[0][0] #index zero
    return f"""Based on your preferences,
        we recommend checking out more articles on {top_recommendation}"""
    
#Call the function
recommendation = byte_match(reader_preferences)
print(recommendation)
    
# Extended example
import spacy
# import countVectorizer
from sklearn.metrics.pairwise import cosine_similarity

articles = {
    'Artcle One' : 'In the political arena, leaders discuss global issues.',
    'Article Two' : 'The latest technological advancements shape our future.',
    'Article Three' : 'Scientific breakthroughs revolutionize the way we live.'
}

user_preference = 'In the political arena, leaders discuss global issues.'articles

nlp = spacy.load('en_core_web_md')

def calculate_similarity(user_preference, articles): # user preference and articles we are comapring to
    #tokenise article and pull out vector for the values in articles (not the keys)
    vectors = [nlp(article).vector for article in articles.values()]
    
    user_vector = nlp(user_preference).vector
    similarity = cosine_similarity([user_vector], vectors)[0]
# cosine similarity demands that the 2 things we are comparing must have same shape. Need 2D here use [] around user_vector
    return dict(zip(articles.keys(), similarity)) 

scores = calculate_similarity(user_preference, articles)
print(scores)

recommend = byte_match(scores)
print(recommend)