from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# -------------------------------
# Text Cleaning
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# -------------------------------
# Complete Kabaddi Knowledge Base
# -------------------------------
kabaddi_data = {
 # BASIC INFO
"what is kabaddi": "Kabaddi is a high-contact sport where one brave soul raids seven defenders, tags them, and escapes without breathing — simple idea, brutal execution.",
"kabaddi meaning": "Kabaddi means running into danger voluntarily and trusting your lungs more than your luck.",
"origin of kabaddi": "Kabaddi was born in ancient India, back when fitness meant survival, not gym memberships.",
"how to play": "Kabaddi is played by sending a raider to annoy defenders, touch them, and sprint back before oxygen gives up.",
"history of kabaddi": "Kabaddi evolved from a survival sport into an organized game once people realized chaos needed rules.",

# COURT
"kabaddi court size": "A kabaddi court is just big enough for chaos: 13m by 10m of controlled violence.",
"kabaddi court lines": "Kabaddi court lines exist to make sure players break rules in a measurable way.",

# TEAM & PLAYERS
"how many players in kabaddi": "Each team fields 7 players, because more would be unfair and fewer would be suicidal.",
"kabaddi team size": "A kabaddi team has 12 players, including substitutes waiting for their turn to suffer.",
"raider role": "A raider’s job is to score points while being hunted like prey.",
"defender role": "Defenders exist to remind raiders that confidence has consequences.",

# MATCH RULES
"kabaddi match duration": "A kabaddi match lasts 40 minutes, which feels much longer if you’re the raider.",
"kabaddi raid rules": "A raider must cross the baulk line and return alive — touching is optional, survival is not.",
"kabaddi foul": "Kabaddi fouls happen when players forget this is a sport and not a street fight.",

# SCORING
"kabaddi scoring": "Kabaddi scoring rewards touching people and stopping them from escaping — legally.",
"bonus point in kabaddi": "A bonus point is earned by risking balance, speed, and dignity.",
"all out in kabaddi": "An all-out means your entire team failed together, earning the opponent extra points.",

# REVIVAL
"kabaddi revival rule": "Players revive one by one, because kabaddi believes in second chances earned the hard way.",

# TACKLES & MOVES
"types of kabaddi tackles": "Kabaddi tackles include ankle holds and chain tackles, all designed to ruin a raider’s plans.",
"dubki move": "Dubki is a move where the raider disappears under defenders and reappears as a highlight clip.",
"frog jump": "Frog jump is exactly what it sounds like — athletic, risky, and occasionally embarrassing.",

# OFFICIALS
"kabaddi officials": "Officials exist to keep kabaddi legal and stop it from turning into a wrestling match.",

# PRO KABADDI LEAGUE
"pro kabaddi league": "Pro Kabaddi League turned a traditional sport into prime-time entertainment.",
"pkl teams": "PKL teams are proof that kabaddi loyalty is as serious as cricket fandom.",

# PLAYERS
"pardeep narwal": "Pardeep Narwal is called the Dubki King because defenders still haven’t figured him out.",
"dubki king": "Dubki King refers to Pardeep Narwal, not a myth, just a recurring nightmare for defenders.",
"pkl king": "PKL King is a title given to Pardeep Narwal after years of defensive suffering.",
"rahul chaudhari": "Rahul Chaudhari is the Poster Boy of Kabaddi — talent plus timing.",
"manjeet chhillar": "Manjeet Chhillar is proof that defense can be just as terrifying as offense.",
"anup kumar": "Anup Kumar led teams with calm strategy while chaos unfolded around him.",
"ajay thakur": "Ajay Thakur captained India to a World Cup win and earned permanent respect.",

# BEST PLAYERS
"best raider in kabaddi": "Pardeep Narwal is widely considered the best raider because numbers don’t lie.",
"best defender in kabaddi": "Manjeet Chhillar and Anup Kumar made stopping raids look personal.",

# TYPES
"types of kabaddi": "Kabaddi comes in multiple styles, each with the same goal: outsmart and overpower.",

# FITNESS
"kabaddi fitness": "Kabaddi fitness demands speed, strength, lungs of steel, and quick decisions.",
"kabaddi training": "Kabaddi training is where excuses go to die.",

# INTERNATIONAL
"kabaddi countries": "Kabaddi is played worldwide by nations that enjoy intensity over comfort.",
"kabaddi tournaments": "Major kabaddi tournaments prove this sport is no longer a local secret.",

# AI / PROJECT
"kabaddi analytics": "Kabaddi analytics turns raw chaos into data, insights, and smart predictions.",

}

questions = [clean_text(q) for q in kabaddi_data.keys()]
answers = list(kabaddi_data.values())

# -------------------------------
# Vectorizer
# -------------------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    stop_words="english"
)

question_vectors = vectorizer.fit_transform(questions)

# -------------------------------
# Chatbot Function
# -------------------------------
def kabaddi_chatbot(user_input):
    user_input = clean_text(user_input)
    user_vector = vectorizer.transform([user_input])

    similarity = cosine_similarity(user_vector, question_vectors)
    best_match = similarity.argmax()
    score = similarity[0][best_match]

    if score >= 0.25:
        return answers[best_match]
    else:
        return "Ask something related to Kabaddi."

# -------------------------------
# Console Chat
# -------------------------------
print("Kabaddi Chatbot Activated")
print("Type 'exit' to quit\n")

while True:
    query = input("You: ")
    if query.lower() == "exit":
        print("Match finished")
        break
    print("Kabaddi Bot:", kabaddi_chatbot(query))
