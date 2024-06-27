# Movie-Recommender
Developed a content-based movie recommender system using Python and scikit-learn.
Extracted movie details and credits from two datasets using pandas and merged them based on movie titles.
Preprocessed the data by handling missing values, removing duplicates, and converting JSON strings to Python lists.
Engineered features by combining movie information such as overview, genres, keywords, cast, and crew into tags.
Applied text normalization techniques including lowercase conversion and stemming to improve data consistency.
Utilized CountVectorizer for feature extraction and vectorization of text data.
Calculated cosine similarity between movies based on their content features to generate a similarity matrix.
Implemented a recommendation function that suggests top 5 similar movies given a movie title.
The recommender system assists users in discovering movies based on their preferences and viewing history.
