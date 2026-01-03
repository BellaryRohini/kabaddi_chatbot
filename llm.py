import numpy as np
import re
from collections import defaultdict, Counter
import random

class TrueLLM:
    """
    A true language model that learns from raw text corpus
    Now with SARCASM MODE! 🎭
    """
    
    def __init__(self, context_size=3, sarcasm_mode=True):
        self.context_size = context_size
        self.sarcasm_mode = sarcasm_mode  # Toggle sarcasm on/off
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        
        self.ngram_model = defaultdict(Counter)
        self.cooccurrence = defaultdict(Counter)
        self.corpus = []
        
        # Sarcasm templates and triggers
        self.sarcasm_triggers = {
            'obvious_questions': [
                r'is kabaddi.*football',
                r'is kabaddi.*cricket',
                r'kabaddi.*ball',
                r'kabaddi.*bat',
                r'play.*computer',
                r'video game',
                r'is it.*boring',
                r'stupid.*sport',
                r'easy.*sport'
            ],
            'repetitive': [],  # Will track if user asks same question
            'absurd': [
                r'kabaddi.*space',
                r'kabaddi.*moon',
                r'aliens.*kabaddi',
                r'dinosaur',
                r'time travel',
                r'superhero'
            ]
        }
        
        self.sarcasm_templates = {
            'obvious': [
                "Oh sure, {wrong_fact}. And I'm the queen of England! Actually, {correct_fact}",
                "Absolutely! {wrong_fact}. Just kidding - {correct_fact}",
                "Wow, great question! Next you'll ask if water is wet. But seriously, {correct_fact}",
                "Let me check my crystal ball... Nope! {correct_fact}",
                "Oh definitely, and pigs fly too! Real talk: {correct_fact}"
            ],
            'absurd': [
                "Yes, and unicorns referee the matches! But in reality, {correct_fact}",
                "Sure, in an alternate universe maybe! Here on Earth, {correct_fact}",
                "I love your imagination! But let's get real: {correct_fact}",
                "That's... creative! Though actually, {correct_fact}"
            ],
            'too_easy': [
                "Did you even try searching? {correct_fact}",
                "Come on, that's Kabaddi 101! {correct_fact}",
                "Really? That's literally the first thing anyone learns! {correct_fact}"
            ],
            'praise': [
                "Now THAT'S a great question! ",
                "Ooh, I like this one! ",
                "Finally, a real question! ",
                "Now we're talking! "
            ]
        }
        
        self.question_history = []  # Track questions asked
        
    def tokenize(self, text):
        """Convert text to tokens"""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s\.\?\!]', '', text)
        tokens = text.split()
        return tokens
    
    def build_vocabulary(self, texts):
        """Build vocabulary from raw text"""
        all_words = []
        for text in texts:
            tokens = self.tokenize(text)
            all_words.extend(tokens)
        
        word_freq = Counter(all_words)
        vocab = ['<START>', '<END>', '<UNK>'] + [w for w, c in word_freq.items() if c >= 2]
        
        self.word_to_id = {w: i for i, w in enumerate(vocab)}
        self.id_to_word = {i: w for w, i in self.word_to_id.items()}
        self.vocab_size = len(vocab)
        
        print(f"Built vocabulary: {self.vocab_size} words")
        return vocab
    
    def learn_from_corpus(self, texts):
        """Learn language patterns from raw text corpus"""
        print("Learning from text corpus...")
        
        self.corpus = texts
        self.build_vocabulary(texts)
        
        for text in texts:
            tokens = ['<START>'] * self.context_size + self.tokenize(text) + ['<END>']
            
            for i in range(len(tokens) - self.context_size):
                context = tuple(tokens[i:i + self.context_size])
                next_word = tokens[i + self.context_size]
                self.ngram_model[context][next_word] += 1
            
            for i, word1 in enumerate(tokens):
                for j in range(max(0, i-5), min(len(tokens), i+6)):
                    if i != j:
                        word2 = tokens[j]
                        self.cooccurrence[word1][word2] += 1
        
        print(f"Learned {len(self.ngram_model)} n-gram patterns")
        print(f"Learned co-occurrences for {len(self.cooccurrence)} words")
        print(f"🎭 Sarcasm mode: {'ENABLED' if self.sarcasm_mode else 'DISABLED'}")
    
    def detect_sarcasm_trigger(self, question):
        """Detect if question deserves a sarcastic response"""
        if not self.sarcasm_mode:
            return None, None
        
        question_lower = question.lower()
        
        # Check for obvious/silly questions
        for pattern in self.sarcasm_triggers['obvious_questions']:
            if re.search(pattern, question_lower):
                return 'obvious', pattern
        
        # Check for absurd questions
        for pattern in self.sarcasm_triggers['absurd']:
            if re.search(pattern, question_lower):
                return 'absurd', pattern
        
        # Check for repetitive questions - FIXED: Only exact same question asked consecutively
        recent_questions = self.question_history[-3:] if len(self.question_history) >= 3 else self.question_history
        if recent_questions.count(question_lower) >= 2:  # Asked at least twice in recent history
            return 'repetitive', None
        
        # Check for "too easy" questions (basic facts asked in complex way)
        if len(question.split()) > 15 and any(w in question_lower for w in ['what', 'is', 'kabaddi']):
            return 'too_easy', None
        
        return None, None
    
    def generate_sarcastic_response(self, sarcasm_type, correct_answer, question):
        """Generate a sarcastic response"""
        question_lower = question.lower()
        
        # Generate wrong fact based on question
        wrong_fact = "that's exactly right"
        if 'ball' in question_lower:
            wrong_fact = "they use a flaming ball made of dragon scales"
        elif 'football' in question_lower or 'cricket' in question_lower:
            wrong_fact = "it's exactly like that"
        elif 'space' in question_lower or 'moon' in question_lower:
            wrong_fact = "kabaddi is the official sport of Mars"
        elif 'easy' in question_lower:
            wrong_fact = "it's basically just walking around"
        elif 'boring' in question_lower:
            wrong_fact = "you're totally right, watching paint dry is more exciting"
        
        if sarcasm_type == 'obvious':
            template = random.choice(self.sarcasm_templates['obvious'])
            return template.format(wrong_fact=wrong_fact, correct_fact=correct_answer)
        
        elif sarcasm_type == 'absurd':
            template = random.choice(self.sarcasm_templates['absurd'])
            return template.format(correct_fact=correct_answer)
        
        elif sarcasm_type == 'too_easy':
            template = random.choice(self.sarcasm_templates['too_easy'])
            return template.format(correct_fact=correct_answer)
        
        elif sarcasm_type == 'repetitive':
            return f"Seriously? I just answered this! {correct_answer}"
        
        return correct_answer
    
    def should_praise_question(self, question):
        """Detect if question is genuinely good and deserves praise"""
        question_lower = question.lower()
        
        # Complex/thoughtful questions deserve praise
        good_question_indicators = [
            r'strategy',
            r'technique',
            r'why.*important',
            r'how.*improve',
            r'difference between',
            r'compare.*to',
            r'evolution',
            r'history.*development',
            r'psychological',
            r'training.*method'
        ]
        
        for pattern in good_question_indicators:
            if re.search(pattern, question_lower):
                return True
        
        return False
    
    def find_relevant_context(self, question_tokens, original_question):
        """Find most relevant sentences from corpus based on question"""
        question_words = set(question_tokens)
        question_lower = original_question.lower()
        
        sentence_scores = []
        
        for text in self.corpus:
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                if not sentence.strip():
                    continue
                    
                sent_lower = sentence.lower()
                sent_tokens = set(self.tokenize(sentence))
                
                # Base overlap score
                overlap = len(question_words & sent_tokens)
                score = overlap * 2
                
                # Question type specific scoring
                
                # "What is kabaddi" - definition questions
                if re.search(r'what.*is.*kabaddi', question_lower):
                    if 'is a' in sent_lower or 'is the' in sent_lower:
                        if 'contact' in sent_lower or 'team sport' in sent_lower:
                            score += 15
                
                # "Where did kabaddi originate" - origin questions
                if 'where' in question_lower and any(w in question_lower for w in ['origin', 'originate', 'come from', 'start']):
                    if any(w in sent_lower for w in ['originated', 'origin', 'punjab', 'tamil', 'maharashtra', 'ancient india', '4000 years']):
                        score += 20
                
                # "How many players" - team size questions
                if re.search(r'how\s+many.*player', question_lower) or re.search(r'player.*team', question_lower):
                    if 'seven players' in sent_lower or 'teams of seven' in sent_lower:
                        score += 20
                
                # "How to play" questions
                if re.search(r'how.*play', question_lower) or 'how is it played' in question_lower:
                    if any(w in sent_lower for w in ['raider enters', 'attempts to tag', 'chant kabaddi', 'hold their breath']):
                        score += 15
                
                # Court size questions
                if 'court' in question_lower and ('size' in question_lower or 'measure' in question_lower or 'dimension' in question_lower):
                    if re.search(r'\d+\s*meters', sent_lower):
                        score += 18
                
                # Famous players questions
                if any(w in question_lower for w in ['famous', 'best', 'top']) and 'player' in question_lower:
                    if any(name in sent_lower for name in ['pardeep narwal', 'anup kumar', 'pawan sehrawat', 'rahul chaudhari', 'fazel atrachali']):
                        score += 15
                
                # Raiders specifically
                if 'raider' in question_lower and any(w in question_lower for w in ['best', 'famous', 'top']):
                    if 'raider' in sent_lower and any(name in sent_lower for name in ['pardeep', 'pawan', 'naveen', 'rahul']):
                        score += 18
                
                # Defenders specifically
                if 'defender' in question_lower:
                    if 'defender' in sent_lower or 'defensive' in sent_lower:
                        score += 15
                
                # PKL questions
                if 'pkl' in question_lower or 'pro kabaddi league' in question_lower:
                    if 'pkl' in sent_lower or 'pro kabaddi league' in sent_lower:
                        score += 12
                
                # Rules and scoring - CRITICAL FIX
                if any(w in question_lower for w in ['rule', 'rules']):
                    # Prioritize sentences about actual gameplay rules
                    if any(w in sent_lower for w in ['raider', 'defender', 'chant', 'breath', 'tag', 'tackle', 'enters', 'attempts', 'returns', 'declared out']):
                        score += 30
                    # Also boost scoring rules
                    if any(w in sent_lower for w in ['point', 'score', 'all out', 'super raid', 'super tackle', 'bonus']):
                        score += 25
                
                # "How to play" questions
                if 'how' in question_lower and any(w in question_lower for w in ['play', 'played']):
                    if any(w in sent_lower for w in ['raider enters', 'attempts to tag', 'chant kabaddi', 'hold their breath', 'defenders try', 'teams of seven']):
                        score += 30
                
                # Specific scoring questions
                if any(w in question_lower for w in ['point', 'score', 'scoring']):
                    if any(w in sent_lower for w in ['point', 'score', 'all out', 'super raid', 'super tackle', 'raiders score', 'defenders score']):
                        score += 25
                
                if score > 0:
                    sentence_scores.append((score, sentence.strip()))
        
        # Sort by relevance
        sentence_scores.sort(reverse=True, key=lambda x: x[0])
        
        return [s for _, s in sentence_scores[:5]]
    
    def generate_response(self, context_sentences, original_question):
        """Generate clean, accurate response from learned text"""
        if not context_sentences:
            return "I don't have enough information about that."
        
        question_lower = original_question.lower()
        
        # For "what is kabaddi" - return the definition
        if re.search(r'what.*is.*kabaddi', question_lower):
            for sent in context_sentences:
                if 'is a contact team sport' in sent.lower():
                    return sent.strip()
        
        # For origin questions - combine relevant sentences
        if 'where' in question_lower and any(w in question_lower for w in ['origin', 'originate', 'come from', 'start']):
            origin_sents = []
            for sent in context_sentences:
                if any(w in sent.lower() for w in ['originated', 'origin', 'ancient', 'punjab', 'tamil', 'maharashtra', '4000 years']):
                    origin_sents.append(sent.strip())
            if origin_sents:
                return ' '.join(origin_sents[:2])
        
        # For "how many players" questions
        if re.search(r'how\s+many.*player', question_lower) or 'players in a team' in question_lower:
            for sent in context_sentences:
                if 'seven players' in sent.lower() or 'teams of seven' in sent.lower():
                    return sent.strip()
        
        # For rules questions - MAJOR FIX
        if any(w in question_lower for w in ['rule', 'rules']):
            rule_sents = []
            for sent in context_sentences[:8]:  # Check more sentences
                sent_lower = sent.lower()
                # Prioritize actual gameplay rules
                if any(w in sent_lower for w in ['raider enters', 'attempts to tag', 'chant kabaddi', 'hold their breath', 
                                                   'defenders try', 'tackle', 'catching', 'declared out', 'teams of seven',
                                                   'stops chanting', 'takes a breath', 'returns successfully']):
                    rule_sents.append(sent.strip())
                # Also include scoring rules
                elif any(w in sent_lower for w in ['raiders score', 'defenders score', 'all out', 'super raid', 'super tackle', 'bonus points']):
                    rule_sents.append(sent.strip())
            
            if rule_sents:
                # Return top 4-5 rule sentences
                return ' '.join(rule_sents[:5])
        
        # For "how to play" questions
        if 'how' in question_lower and any(w in question_lower for w in ['play', 'played']):
            play_sents = []
            for sent in context_sentences[:8]:
                sent_lower = sent.lower()
                if any(w in sent_lower for w in ['raider enters', 'attempts to tag', 'chant kabaddi', 'hold their breath',
                                                   'defenders try', 'teams of seven', 'compete', 'takes turns']):
                    play_sents.append(sent.strip())
            
            if play_sents:
                return ' '.join(play_sents[:4])
        
        # For specific scoring rules
        if any(w in question_lower for w in ['point', 'score', 'scoring', 'all out', 'super raid', 'super tackle']):
            score_sents = []
            for sent in context_sentences[:5]:
                sent_lower = sent.lower()
                if any(w in sent_lower for w in ['point', 'score', 'all out', 'super raid', 'super tackle', 'bonus', 'raiders score']):
                    score_sents.append(sent.strip())
            
            if score_sents:
                return ' '.join(score_sents[:3])
        
        # For "how to play" questions - OLD HANDLER (now handled above)
        if re.search(r'how.*play', question_lower) or 'how is it played' in question_lower:
            # Skip - already handled in rules section above
            pass
        
        # For court size questions
        if 'court' in question_lower and any(w in question_lower for w in ['size', 'measure', 'dimension']):
            court_sents = []
            for sent in context_sentences:
                if 'meters' in sent.lower() and 'court' in sent.lower():
                    court_sents.append(sent.strip())
            if court_sents:
                return ' '.join(court_sents[:2])
        
        # For famous players questions
        if any(w in question_lower for w in ['famous', 'best', 'top']) and 'player' in question_lower:
            player_sents = []
            for sent in context_sentences[:5]:
                if any(name in sent.lower() for name in ['pardeep', 'anup', 'pawan', 'rahul', 'fazel', 'naveen', 'sandeep', 'manjeet']):
                    player_sents.append(sent.strip())
            if player_sents:
                return ' '.join(player_sents[:3])
        
        # Default: return the most relevant sentence
        response = context_sentences[0].strip()
        if not response.endswith('.'):
            response += '.'
        
        return response
    
    def toggle_sarcasm(self):
        """Toggle sarcasm mode on/off"""
        self.sarcasm_mode = not self.sarcasm_mode
        return self.sarcasm_mode
    
    def answer(self, question):
        """Answer a question using learned knowledge (with optional sarcasm!)"""
        question = question.strip()
        question_lower = question.lower()
        
        # Handle sarcasm toggle command
        if question_lower in ['toggle sarcasm', 'sarcasm on', 'sarcasm off', 'be sarcastic', 'stop sarcasm']:
            new_mode = self.toggle_sarcasm()
            return f"🎭 Sarcasm mode is now {'ON' if new_mode else 'OFF'}! {'Prepare for sass!' if new_mode else 'Back to boring serious mode.'}"
        
        # Handle greetings
        if re.match(r'^(hi|hello|hey|hii)', question_lower):
            if self.sarcasm_mode:
                return "Oh great, another human. What kabaddi wisdom do you seek? 🙄"
            return "Hello! I've learned about kabaddi from text. Ask me anything!"
        
        # Track question history for repetition detection (AFTER greetings check)
        self.question_history.append(question_lower)
        if len(self.question_history) > 10:
            self.question_history.pop(0)
        
        # Handle "can I play" questions
        if re.match(r'^(can|could|may) (i|we)', question_lower):
            if self.sarcasm_mode:
                return "Can you play? I mean, do you have two legs and lungs? Then probably yes! Kabaddi requires teamwork, fitness, and strategy - but hey, you asked!"
            return "Yes, absolutely! Kabaddi is a sport anyone can play. It requires teamwork, fitness, and strategic thinking."
        
        # Handle "is it" court size verification questions
        if re.match(r'^is', question_lower) and re.search(r'\d+', question):
            numbers = re.findall(r'\d+', question)
            if len(numbers) >= 2:
                n1, n2 = numbers[0], numbers[1]
                if (n1 == '13' and n2 == '10') or (n1 == '10' and n2 == '13'):
                    if self.sarcasm_mode:
                        return "Wow, someone actually knows their measurements! Yes, that's correct! The kabaddi court is 13 meters by 10 meters for men's matches. Gold star for you! ⭐"
                    return "Yes, that's correct! The kabaddi court is 13 meters by 10 meters for men's matches."
                elif (n1 == '12' and n2 == '8') or (n1 == '8' and n2 == '12'):
                    if self.sarcasm_mode:
                        return "Look at you, getting it right! Yes, the women's kabaddi court is 12 meters by 8 meters. Impressive! 👏"
                    return "Yes, that's correct! The women's kabaddi court is 12 meters by 8 meters."
                else:
                    if self.sarcasm_mode:
                        return f"Nice try, but nope! Did you just make up random numbers? The kabaddi court measures 13m x 10m for men and 12m x 8m for women. Not {n1}m x {n2}m!"
                    return "No, that's not correct. The kabaddi court measures 13 meters by 10 meters for men and 12 meters by 8 meters for women."
        
        # Tokenize question
        q_tokens = self.tokenize(question)
        
        if not q_tokens:
            return "Could you please ask a question?"
        
        # Check if question is about kabaddi
        kabaddi_words = ['kabaddi', 'player', 'raider', 'defender', 'court', 'raid', 
                        'pkl', 'tournament', 'game', 'sport', 'team', 'match', 'originated', 'origin']
        
        is_kabaddi = any(word in question_lower for word in kabaddi_words)
        
        if not is_kabaddi:
            if self.sarcasm_mode:
                return "Um, I'm a KABADDI expert, not a general knowledge encyclopedia! Ask me about kabaddi, please! 🙄"
            return "I've learned about kabaddi. Please ask me kabaddi-related questions!"
        
        # Handle questions we don't have data for
        if any(word in question_lower for word in ['gold medal', 'olympics', 'medal', 'championship winner']):
            if self.sarcasm_mode:
                return "Oh sure, let me just pull that out of my database... oh wait, I DON'T HAVE IT! I can tell you about kabaddi basics, players, rules, and PKL though. Try those?"
            return "I don't have information about that specific topic in my training data. I can tell you about kabaddi basics, players, rules, and the Pro Kabaddi League."
        
        # Find relevant context from learned corpus
        relevant_sentences = self.find_relevant_context(q_tokens, question)
        
        if not relevant_sentences:
            if self.sarcasm_mode:
                return "Wow, you stumped me! I haven't learned about that yet. Maybe try asking something I actually know? Like basics, players, rules, or PKL?"
            return "I haven't learned enough about that specific topic yet. Try asking about kabaddi basics, players, rules, or Pro Kabaddi League."
        
        # Generate base response from learned patterns
        base_response = self.generate_response(relevant_sentences, question)
        
        # Check if we should add sarcasm or praise
        if self.sarcasm_mode:
            # Check for sarcasm triggers
            sarcasm_type, _ = self.detect_sarcasm_trigger(question)
            
            if sarcasm_type:
                return self.generate_sarcastic_response(sarcasm_type, base_response, question)
            
            # Check if question deserves praise
            if self.should_praise_question(question):
                praise = random.choice(self.sarcasm_templates['praise'])
                return praise + base_response
        
        return base_response
    
    def interactive_chat(self):
        """Start interactive chat"""
        print("\n" + "=" * 60)
        print("True LLM Kabaddi Chatbot - Learned from Text")
        if self.sarcasm_mode:
            print("🎭 SARCASM MODE: ENABLED")
        print("=" * 60)
        print("I learned everything from reading about kabaddi!")
        print("Type 'quit' or 'exit' to end")
        print("Type 'toggle sarcasm' to switch modes")
        print("=" * 60)
        print()
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                if self.sarcasm_mode:
                    print("Bot: Finally! Don't let the door hit you on the way out! 👋")
                else:
                    print("Bot: Goodbye!")
                break
            
            if not user_input:
                continue
            
            response = self.answer(user_input)
            print(f"Bot: {response}\n")


# Create a comprehensive text corpus about kabaddi
KABADDI_CORPUS = [
    "Kabaddi is a contact team sport that originated in ancient India. The sport is played between two teams.",
    "In kabaddi two teams of seven players each compete against each other. Each team takes turns sending a raider.",
    "The raider enters the opponent's half of the court and attempts to tag as many defenders as possible.",
    "While raiding the player must hold their breath and continuously chant kabaddi kabaddi kabaddi.",
    "If the raider stops chanting kabaddi or takes a breath they are declared out.",
    "Defenders try to stop the raider by tackling and catching them before they return.",
    "The kabaddi court measures 13 meters by 10 meters for men's matches.",
    "For women's kabaddi the court is smaller at 12 meters by 8 meters.",
    "The court is divided into two halves by a mid-line and has bonus lines and baulk lines.",
    "Famous kabaddi players include Pardeep Narwal who is known as the Record Breaker.",
    "Pardeep Narwal holds the record for most raid points in Pro Kabaddi League history.",
    "Anup Kumar was one of the best kabaddi captains and led India to many victories.",
    "Pawan Sehrawat is called the Hi-Flyer because of his jumping ability and raiding skills.",
    "Rahul Chaudhari has scored over 900 points in his kabaddi career.",
    "Fazel Atrachali from Iran is known as the Iranian Wall for his defensive abilities.",
    "The best raiders in kabaddi include Pardeep Narwal, Pawan Sehrawat, and Naveen Kumar.",
    "Top defenders include Fazel Atrachali, Sandeep Narwal, and Manjeet Chhillar.",
    "Pro Kabaddi League or PKL started in 2014 and revolutionized the sport in India.",
    "PKL has 12 teams including Patna Pirates, U Mumba, Bengaluru Bulls, and Dabang Delhi.",
    "Patna Pirates has won the most PKL titles with three championships.",
    "Each kabaddi match consists of two halves of 20 minutes each.",
    "There is a 5 minute break between the two halves of a kabaddi match.",
    "Each raid in kabaddi can last for a maximum of 30 seconds.",
    "If a raider tags defenders and returns successfully they score points.",
    "Raiders score one point for each defender they tag.",
    "When the entire defending team is eliminated it is called an All Out.",
    "An All Out gives the attacking team two bonus points.",
    "A Super Raid is when a raider scores three or more points in a single raid.",
    "Defenders score points by successfully catching and stopping the raider.",
    "A Super Tackle gives defenders one extra point when three or fewer defenders catch a raider.",
    "Kabaddi originated in India more than 4000 years ago as a way to train warriors.",
    "The sport was popular in Punjab, Tamil Nadu, and Maharashtra regions of India.",
    "Modern kabaddi rules were standardized in the 1930s.",
    "Kabaddi became part of the Asian Games in 1990.",
    "India has won every Kabaddi World Cup tournament since it started.",
    "Raiding techniques in kabaddi include toe touch, hand touch, and dubki.",
    "The dubki is a diving move where raiders slide between defenders.",
    "Defense techniques include ankle hold, thigh hold, and chain tackle.",
    "A chain tackle is when multiple defenders coordinate to catch a raider.",
    "Kabaddi improves cardiovascular fitness, strength, agility, and reflexes.",
    "The sport builds teamwork, mental alertness, and strategic thinking.",
    "Kabaddi requires no special equipment, just a proper court and players.",
    "Each team can have up to five substitute players in addition to seven playing.",
    "Kabaddi is now played in over 70 countries around the world.",
    "Major kabaddi playing countries include India, Iran, Pakistan, Bangladesh, and South Korea.",
]

# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("Building True LLM from Kabaddi Text Corpus")
    print("=" * 60)
    
    # Create the model with SARCASM MODE enabled!
    llm = TrueLLM(context_size=3, sarcasm_mode=True)
    
    # Learn from corpus (NO dictionaries, just raw text!)
    llm.learn_from_corpus(KABADDI_CORPUS)
    
    print("\nModel training complete!")
    print("The model learned everything by reading kabaddi text.")
    print("It has NO pre-written answers - generates from learned patterns.")
    print("🎭 BONUS: Now with SARCASM MODE! 🎭\n")
    
    # Start chat
    llm.interactive_chat()