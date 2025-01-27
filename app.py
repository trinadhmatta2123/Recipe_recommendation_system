from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
from googletrans import Translator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the dataset
file_path = "IndianFoodDataset.csv"
data = pd.read_csv(file_path)

# Clean and split individual ingredients
def clean_and_split_ingredients(ingredients):
    """Cleans and splits ingredients into a list of words."""
    if pd.isna(ingredients):
        return []
    ingredients_list = []
    for ingredient in ingredients.split(","):
        ingredient = re.sub(r"\b\d+[^a-zA-Z]*\b", "", ingredient)  # Remove numeric measurements
        ingredient = re.sub(r"\([^)]*\)", "", ingredient)  # Remove text in parentheses
        ingredient = re.sub(r"-.*", "", ingredient)  # Remove text after dashes
        ingredient = ingredient.strip().lower()
        ingredients_list.extend(ingredient.split())  # Split into individual words
    return list(set(ingredients_list))

# Preprocess the dataset
data["SplitIngredients"] = data["TranslatedIngredients"].apply(clean_and_split_ingredients)

# Recommendation function
def recommend_recipes(user_input, data, ingredient_column="SplitIngredients"):
    """Recommends recipes based on user input ingredients."""
    user_ingredients = [ing.strip().lower() for ing in user_input.split(",")]

    def count_matches(recipe_ingredients):
        return len(set(user_ingredients) & set(recipe_ingredients))

    data["MatchCount"] = data[ingredient_column].apply(count_matches)
    matched_recipes = data[data["MatchCount"] > 0].sort_values(by="MatchCount", ascending=False)
    return matched_recipes
# Preprocess data: Convert ingredients to a single string for each recipe
data["IngredientsString"] = data["TranslatedIngredients"].apply(
    lambda x: " ".join(clean_and_split_ingredients(x))
)

# Fit TF-IDF Vectorizer on the dataset
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data["IngredientsString"])

def recommend_recipes_with_cosine(user_input, data, tfidf_vectorizer, tfidf_matrix):
    """Recommends recipes using cosine similarity and TF-IDF vectors."""
    # Preprocess user input
    user_input_cleaned = " ".join([ing.strip().lower() for ing in user_input.split(",")])

    # Transform user input into TF-IDF vector
    user_vector = tfidf_vectorizer.transform([user_input_cleaned])

    # Compute cosine similarity with recipe TF-IDF matrix
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # Add similarity scores to the dataset and sort by score
    data["SimilarityScore"] = similarity_scores
    recommended = data.sort_values(by="SimilarityScore", ascending=False)

    return recommended[recommended["SimilarityScore"] > 0]


# Initialize the translator
translator = Translator()

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/results", methods=["POST"])
def results():
    try:
        user_input = request.form["ingredients"]
        language = request.form.get("language", "en")  # Default to English

        if not user_input.strip():
            return render_template("home.html", error="Please enter ingredients.")

        # Get recommendations using cosine similarity
        recommended = recommend_recipes_with_cosine(user_input, data, tfidf_vectorizer, tfidf_matrix)
        recipes = recommended.to_dict(orient="records")

        # Translate recipes if a language other than English is selected
        if language != "en":
            for recipe in recipes:
                recipe["RecipeName"] = translator.translate(recipe["RecipeName"], dest=language).text
                recipe["TranslatedIngredients"] = translator.translate(recipe["TranslatedIngredients"], dest=language).text
                recipe["TranslatedInstructions"] = translator.translate(recipe["TranslatedInstructions"], dest=language).text

        # Include additional fields in recipes
        for recipe in recipes:
            recipe["PrepTimeInMins"] = recipe.get("PrepTimeInMins", "N/A")
            recipe["Servings"] = recipe.get("Servings", "N/A")
            recipe["Cuisine"] = recipe.get("Cuisine", "N/A")
            recipe["Diet"] = recipe.get("Diet", "N/A")
            recipe["URL"] = recipe.get("URL", "#")

        return render_template("results.html", recipes=recipes, user_input=user_input, language=language)

    except Exception as e:
        print(f"Error: {e}")
        return render_template("home.html", error="An error occurred. Please try again.")



@app.route("/translate", methods=["POST"])
def translate():
    data = request.json
    recipe = data.get("recipe")
    language = data.get("language")
    
    translated = {
        "RecipeName": translator.translate(recipe["RecipeName"], dest=language).text,
        "TranslatedIngredients": translator.translate(recipe["TranslatedIngredients"], dest=language).text,
        "TranslatedInstructions": translator.translate(recipe["TranslatedInstructions"], dest=language).text,
    }
    return jsonify(translated)

if __name__ == "__main__":
    app.run(debug=True)
'''from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
from deep_translator import GoogleTranslator

app = Flask(__name__)

# Load the dataset
file_path = "IndianFoodDataset.csv"
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Dataset not found at {file_path}. Please check the file path.")

# Clean and split individual ingredients
def clean_and_split_ingredients(ingredients):
    """Cleans and splits ingredients into a list of words."""
    if pd.isna(ingredients):
        return []
    ingredients_list = []
    for ingredient in ingredients.split(","):
        ingredient = re.sub(r"\b\d+[^a-zA-Z]*\b", "", ingredient)  # Remove numeric measurements
        ingredient = re.sub(r"\([^)]*\)", "", ingredient)  # Remove text in parentheses
        ingredient = re.sub(r"-.*", "", ingredient)  # Remove text after dashes
        ingredient = ingredient.strip().lower()
        ingredients_list.extend(ingredient.split())  # Split into individual words
    return list(set(ingredients_list))

# Preprocess the dataset
data["SplitIngredients"] = data["TranslatedIngredients"].apply(clean_and_split_ingredients)

# Recommendation function
def recommend_recipes(user_input, data, ingredient_column="SplitIngredients"):
    """Recommends recipes based on user input ingredients."""
    user_ingredients = [ing.strip().lower() for ing in user_input.split(",")]

    # Precompute matches for efficiency
    def count_matches(recipe_ingredients):
        return len(set(user_ingredients) & set(recipe_ingredients))

    data["MatchCount"] = data[ingredient_column].apply(count_matches)
    matched_recipes = data[data["MatchCount"] > 0].sort_values(by="MatchCount", ascending=False)
    return matched_recipes

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/results", methods=["POST"])
def results():
    try:
        user_input = request.form["ingredients"]
        language = request.form.get("language", "en").lower()  # Default to English

        if not user_input.strip():
            return render_template("home.html", error="Please enter ingredients.")

        # Sanitize user input
        user_input = re.sub(r"[^a-zA-Z0-9, ]", "", user_input)

        recommended = recommend_recipes(user_input, data)
        recipes = recommended.to_dict(orient="records")

        # Translate recipes if a language other than English is selected
        if language != "en":
            translator = GoogleTranslator(source="auto", target=language)
            for recipe in recipes:
                recipe["RecipeName"] = translator.translate(recipe.get("RecipeName", ""))
                recipe["TranslatedIngredients"] = translator.translate(recipe.get("TranslatedIngredients", ""))
                recipe["TranslatedInstructions"] = translator.translate(recipe.get("TranslatedInstructions", ""))

        return render_template("results.html", recipes=recipes, user_input=user_input, language=language)

    except Exception as e:
        app.logger.error(f"Error in results route: {e}")
        return render_template("home.html", error="An error occurred. Please try again.")

@app.route("/translate", methods=["POST"])
def translate():
    try:
        data = request.json
        recipe = data.get("recipe")
        language = data.get("language")

        translator = GoogleTranslator(source="auto", target=language)
        translated = {
            "RecipeName": translator.translate(recipe.get("RecipeName", "")),
            "TranslatedIngredients": translator.translate(recipe.get("TranslatedIngredients", "")),
            "TranslatedInstructions": translator.translate(recipe.get("TranslatedInstructions", "")),
        }
        return jsonify(translated)
    except Exception as e:
        app.logger.error(f"Error in translate route: {e}")
        return jsonify({"error": "Translation failed."}), 500

if __name__ == "__main__":
    app.run(debug=True)
'''