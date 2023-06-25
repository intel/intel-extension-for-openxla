from transformers import AutoTokenizer, FlaxGPTJForCausalLM
import jax
import time
from functools import partial
from itertools import chain
import numpy as np

model_id = "EleutherAI/gpt-j-6B"
print("---- Model loading", flush=True)
model = FlaxGPTJForCausalLM.from_pretrained(model_id, dtype=jax.numpy.float16)
model.params = model.to_fp16(model.params)
print("---- Model loading done", flush=True)

# 32->32
max_new_tokens = 32
prompt = "Once upon a time, there existed a little girl, who liked to have adventures." + \
         " She wanted to go to places and meet new people, and have fun."
# output:
# "Once upon a time, there existed a little girl, who liked to have adventures." + \
# " She wanted to go to places and meet new people, and have fun." + \
# " One day, she decided to go on an adventure. She packed her bags, and set off on her journey." + \
# "\n\nThe little girl walked and walked,"

# #1024->128
# max_new_tokens = 128
# prompt = "It is done, and submitted. You can play 'Survival of the Tastiest' on Android, and on the web. Playing on the web works, but you have to simulate multiple touch for table moving and that can be a bit confusing. There is a lot I'd like to talk about. I will go through every topic, insted of making the typical what went right/wrong list. Concept Working over the theme was probably one of the hardest tasks which I had to face. Originally, I had an idea of what kind of game I wanted to develop, gameplay wise - something with a lot of enemies/actors, simple graphics, maybe set in space, controlled from a top-down view. I was confident that I could fit any theme around it. In the end, the problem with a theme like 'Evolution' in a game is that evolution is unassisted. It happens through several seemingly random mutations over time, with the most apt permutation surviving. This genetic car simulator is, in my opinion, a great example of actual evolution of a species facing a challenge. But is it a game? In a game, you need to control something to reach an objective. That control goes against what evolution is supposed to be like. If you allow the user to pick how to evolve something, it's not evolution anymore - it's the equivalent of intelligent design, the fable invented by creationists to combat the idea of evolution. Being agnostic and a Pastafarian, that's not something that rubbed me the right way. Hence, my biggest dillema when deciding what to create was not with what I wanted to create, but with what I did not. I didn't want to create an 'intelligent design' simulator and wrongly call it evolution. This is a problem, of course, every other contestant also had to face. And judging by the entries submitted, not many managed to work around it. I'd say the only real solution was through the use of artificial selection, somehow. So far, I haven't seen any entry using this at its core gameplay. Alas, this is just a fun competition and after a while I decided not to be as strict with the game idea, and allowed myself to pick whatever I thought would work out. My initial idea was to create something where humanity tried to evolve to a next level, but had some kind of foe trying to stop them from doing so. I kind of had this image of human souls flying in space towards a monolith or a space baby (all based in 2001: A Space Odyssey of course) but I couldn't think of compelling (read: serious) mechanics for that. Borgs were my next inspiration, as their whole hypothesis fit pretty well into the evolution theme. But how to make it work? Are you the borg, or fighting the Borg? The third and final idea came to me through my girlfriend, who somehow gave me the idea of making something about the evolution of Pasta. The more I thought about it the more it sounded like it would work, so I decided to go with it. Conversations with my inspiring co-worker Roushey (who also created the 'Mechanical Underdogs' signature logo for my intros) further matured the concept, as it involved into the idea of having individual pieces of pasta flying around and trying to evolve until they became all-powerful. A secondary idea here was that the game would work to explain how the Flying Spaghetti Monster came to exist - by evolving from a normal dinner table. So the idea evolved more or less into this: you are sitting a table. You have your own plate, with is your 'base'. There are 5 other guests at the table, each with their own plate. Your plate can spawn little pieces of pasta. You do so by 'ordering' them through a menu. Some pastas are better than others; some are faster, some are stronger. They have varying 'costs', which are debited from your credits (you start with a number of credits). Once spawned, your pastas start flying around. Their instinct is to fly to other plates, in order to conquer them (the objective of the game is having your pasta conquer all the plates on the table). But they are really autonomous, so after being spawned, you have no control over your pasta (think DotA or LoL creeps). Your pasta doesn't like other people's pasta, so if they meet, they shoot sauce at each other until one dies. You get credits for other pastas your own pasta kill. Once a pasta is in the vicinity of a plate, it starts conquering it for its team. It takes around 10 seconds for a plate to be conquered; less if more pasta from the same team are around. If pasta from other team are around, though, they get locked down in their attempt, unable to conquer the plate, until one of them die (think Battlefield's standard 'Conquest' mode). You get points every second for every plate you own. Over time, the concept"

tokenizer = AutoTokenizer.from_pretrained(model_id)
input_ids = tokenizer([prompt], return_tensors="np").input_ids
print("---- Prompt size:", input_ids.shape, flush=True)

# generate args
prng_key = jax.random.PRNGKey(0)
generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4, prng_key=prng_key)

@jax.jit
def run_model(input_ids):
    gen_tokens = model.generate(input_ids, max_new_tokens=max_new_tokens, **generate_kwargs)
    return gen_tokens

total_time = 0.0
num_iter = 10
num_warmup = 3
total_list = []
for i in range(num_iter):
    tic = time.time()
    gen_tokens = run_model(input_ids)
    gen_text = tokenizer.batch_decode(gen_tokens[0], skip_special_tokens=False)
    toc = time.time()
    print(gen_text, flush=True)
    print("Inference latency: %.3f sec." % (toc - tic), flush=True)
    dur = toc - tic
    if i >= num_warmup:
        total_time += dur
        total_list.append(gen_tokens[1])

print("\n", "-" * 10, "Summary:", "-" * 10, flush=True)
latency = total_time / (num_iter - num_warmup)
print("Inference latency: %.3f sec." % (latency), flush=True)
print(total_list, flush=True)

# first_latency = np.mean([x[0] for x in total_list])
# average_2n = list(chain(*[x[1:] for x in total_list]))
# average_2n.sort()
# average_2n_latency = np.mean(average_2n)
# p90_latency = average_2n[int(len(average_2n) * 0.9)]
# print("First token average latency: %.3f sec." % first_latency, flush=True)
# print("Average 2... latency: %.3f sec." % average_2n_latency, flush=True)
# print("P90 2... latency: %.3f sec." % p90_latency, flush=True)
