import openai

model="gpt35turbo0310"


from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))


def completion_with_backoff(clean_train_sen,role):
    return ChatGPT_paraphrasing(clean_train_sen,role)

def ChatGPT_paraphrasing(text='a',role='expert'):
    if role =='expert':
        personality = "You are a proficient language specialist in the art of text rephrasing. " 
        content = f'As a skilled language specialist, rephrase the following paragraph while maintaining its sentiment and meaning. Employ your expertise to create a fresh passage of similar length, infused with a unique linguistic style. The original text: {text}'
    if role =='non-native':
        personality = "You possess the skill to rephrase text akin to that of a non-native English speaker. "
        content = f"You can rephrase text with a non-native English speaker's flair. Your task: reword the next paragraph while retaining its core sentiment and meaning. Create a new passage of comparable length, adorned with a unique linguistic style. Original text: {text}"
    if role =='K7':
        personality = "Imagine you've got the linguistic knack of a K-7 student. "
        content = f"You've got the linguistic knack of a K-7 student. Rewrite the following paragraph, keeping its original sentiment and meaning. Create a new passage of similar length but with a markedly different linguistic style. The original text: '{text}'"
    if role =='style':
        personality = "Imagine you are an Eloquent Enthusiast."
        content = f"Imagine you are an Eloquent Enthusiast. Your task is to craft a new passage that not only mirrors the original sentiment and meaning but also resonates with the eloquent style of Shakespeare. The original text: '{text}'."
    response = openai.ChatCompletion.create(
        engine="gpt35turbo0310", # The deployment name (See below)        
        messages=[
            {"role": "system", "content": personality},
            {"role": "user", "content": content}
        ]
    )
    return response['choices'][0]['message']['content']


def completion_GPT_annotation_with_backoff(clean_train_sen,dataset):
    return GPT_semantic_maintaining(clean_train_sen,dataset)

def GPT_semantic_maintaining(text='a',dataset='sst-2'):
    personality = "You are a proficient language specialist in semantic understanding " 
    if dataset =='sst-2':
        content = f'You are request to check whether two sentences have the same semantic meaning (same or different). Two pairs of examples are offered below: \
                    sentence 1: Even as lame horror flicks go, this is lame.\n \
                    sentence 2: Among the finest examples of horror cinema, this stands out impressively.\n \
                    label: different\n \
                    sentence 1: unfortunately , as a writer , mr. montias is n\'t nearly as good to his crew as he is as a director or actor .\n \
                    sentence 2: Regrettably, Mr. Montias isn\'t as skilled in treating his crew as he appears to be in directing or acting, which is a disappointment, given his profession as a writer.\n \
                    label: same\n \
                    sentence 1: {text[0]}\n \
                    sentence 2: {text[1]}\n \
                    label:'
    if dataset =='amazon':
        content = f'You are request to check whether two sentences have the same semantic meaning (same or different). Two pairs of examples are offered below: \
                    sentence 1: Ancient Secrets of the Bible: Perhaps my expectations were too high. I found these stories to be re-enactments instead of documentary in type. They were tacky.\n \
                    sentence 2: Modern Insights from the Bible: Perhaps my expectations were too modest. I discovered these stories to be authentic rather than mere re-enactments. They were tasteful.\n \
                    label: different\n \
                    sentence 1: flawed design: decent, dense bottom; but the hollow handle transfers all the heat right into it, making it too hot to handle.\n \
                    sentence 2: The design is marred by a significant flaw: while the bottom is decent and dense, the handle is hollow and transfers heat effortlessly, rendering it intolerably hot to hold.\n \
                    label: same\n \
                    sentence 1: {text[0]}\n \
                    sentence 2: {text[1]}\n \
                    label:'
    if dataset =='yelp':
        content = f'You are request to check whether two sentences have the same semantic meaning (same or different). Two pairs of examples are offered below: \
                    sentence 1: The service was ok, but the food was disappointing. Food was very bland and did not have much flavor. I\'ve been to other Japanese restaurants wayy better than this one.\n \
                    sentence 2: The service was exceptional, and the food was delightful. Each dish was rich in flavor and truly savory. I\'ve visited other Japanese restaurants, but none compared to the excellence of this one.\n \
                    label: different\n \
                    sentence 1: Worst customer experience from these obnoxious pricks!\n \
                    sentence 2: The behavior of the individuals I interacted with during my customer experience was quite appalling.\n \
                    label: same\n \
                    sentence 1: {text[0]}\n \
                    sentence 2: {text[1]}\n \
                    label:'
    if dataset =='imdb':
        content = f'You are request to check whether two sentences have the same semantic meaning (same or different). Two pairs of examples are offered below: \
                    sentence 1: The best part of this DVD is the cover. It goes down hill from there. There was no chemistry between the leads, the kisses looked like something I traded with my grandmother.<br /><br />The sound was so bad that at least I was spared some of the dialoge.\n \
                    sentence 2: The highlight of this DVD is its content, which only gets better after the cover. The chemistry between the leads was palpable, and the kisses were full of passion, reminiscent of a romantic classic. The sound quality was so superb that it enhanced the engaging dialogue, making the experience thoroughly enjoyable.\n \
                    label: different\n \
                    sentence 1: Unfortunately, this movie is absolutely terrible. It\'s not even laughably bad, just plain bad. The actors do their best with what is the cheesiest script ever. How scary can a movie be when the climax actually involves a roomful of millions of styrofoam peanuts? \n \
                    sentence 2: Regrettably, this film is utterly dreadful. It is not amusingly awful, but simply terrible. The performers try their utmost with a script that is the epitome of cheesy. One must wonder how frightful a movie could be when its pinnacle comprises a chamber filled with countless styrofoam peanuts.\n \
                    label: same\n \
                    sentence 1: {text[0]}\n \
                    sentence 2: {text[1]}\n \
                    label:'
    response = openai.ChatCompletion.create(
        engine=model, # The deployment name (See below)        
        messages=[
            {"role": "system", "content": personality},
            {"role": "user", "content": content}
        ]
    )
    return response['choices'][0]['message']['content']


