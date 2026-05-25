# Evaluates all aligned models using Logic_Sim.

import Kurosagol as k

def trans_logic_sim(ds_name):
    aux = k.Evaluation(ds_name, 'trans')
    column = list(aux.ds.features.keys())[1]
    aux.logic_sim(column)
    #----
    aux1 = k.Evaluation(ds_name, 'inference')
    column = list(aux.ds.features.keys())[1]
    aux1.logic_sim(column)

dataset_name = [
    'Kurosawama/EVAL_gemma-3-1b-it',
    'Kurosawama/EVAL_Llama-3.2-3B',
    'Kurosawama/EVAL_Llama-3.1-8B',
    'Kurosawama/EVAL_Llama-3.2-3B-Instruct',
    'Kurosawama/EVAL_Llama-3.1-8B-Instruct'
]

for _ in dataset_name:
    print("=====================")
    print("Modelo: {}".format(_))
    trans_logic_sim(_)