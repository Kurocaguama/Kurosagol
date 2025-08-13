"""
A preview of how to use different aspects of the Kurosagol package.

Two main examples are showcased: 
    1) Obtaining preference datasets (pd).
    2) Aligning an LLM based on the pd.
"""
import Kurosagol as k
import torch
from datasets import load_dataset

# Kurosagol is developed with the FOLIO dataset in mind.
dataset = load_dataset('yale-nlp/FOLIO', split = 'train')


#----------------------------------------TUTORIAL 1----------------------------------------

# You can always modify the prompt to include different strategies like CoT, 0,1,Multi-Shot learning, etc.
prompt = """
    Your task is to parse the problem into first-order logic.
    -----------------
    Problem:
    {}
    Predicates:
"""
translation = k.DatasetManager(dataset, 'meta-llama/Llama-3.1-8B', 85, prompt)
translation.clean_dataset()
# translation.clean_list = translation.clean_list[0] // IN CASE YOU WANT TO USE JUST ONE STRATEGY. -> translation.clean_list = translation.clean_list[:5] if you want to use various.
translation.gen_strats_list()
translation_ds = translation.good_dataset(translation.greedy) # Just mind that the generating strategy used here is included in line 28.
translation_ds.push_to_hub('Kurosawama/translation_DPO_greedy')


#----------------------------------------TUTORIAL 2----------------------------------------
align_ds = load_dataset('Kurosawama/translation_DPO_greedy')
alignment = k.DPO('meta-llama/Llama-3.1-8B', 'C:\Users\FLopezP\Desktop') # Modify directory accordingly
alignment.train_and_push(align_ds, 'Kurosawama/Llama-3.1-8B-Translation')