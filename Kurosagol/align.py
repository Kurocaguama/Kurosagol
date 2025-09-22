# This file implements DPO to a given LLM and pushes the resulting model to the HF Hub.

import Kurosagol as k
import torch, re
from datasets import load_dataset

def t_and_p(model_id):
    """
        Alinea y sube a huggingface. OBS: SE GENERAR TRES (3) MODELOS, UNO POR CADA OBJETIVO DE ALINEACIÓN.
        model_id = str ; El modelo a alinear
    """
    filtered_name = re.split('\/', model_id)[1]
    steps = ['trans', 'infer', 'retrans']
    aux = k.DPO(model_id, '/media/discoexterno/francisco/modelos')
    for _ in steps:
        if _ == 'trans':
            dataset = load_dataset('Kurosawama/Translation_DPO_{}'.format(filtered_name), split='train')
            pushed_model_name = filtered_name + '-Translation-align'
        if _ == 'infer':
            dataset = load_dataset('Kurosawama/Inference_DPO_{}'.format(filtered_name), split='train')
            pushed_model_name = filtered_name + '-Inference-align'
        if _ == 'retrans':
            dataset = load_dataset('Kurosawama/Retranslation_DPO_{}'.format(filtered_name), split='train')
            pushed_model_name = filtered_name + '-Retranslation-align'
        torch.cuda.empty_cache()

        aux.train_and_push(dataset, pushed_model_name)
        print('Modelo {} subido a HuggingFace'.format(pushed_model_name))
    print('Finalizado. Como siempre, Viva Messi.')


checkpoint_list = [ 
    'meta-llama/Llama-3.1-8B',  
    'meta-llama/Llama-3.1-8B-Instruct', 
    'meta-llama/Llama-3.2-3B', 
    'meta-llama/Llama-3.2-3B-Instruct', 
    'google/gemma-3-1b-it'
]

# Iteración previa. NO SE USÓ PARA EL PAPER.
#for _ in checkpoint_list:
#    t_and_p(_)

def align(ds_name, output_dir, model_name, model_hub_name):
    """
    Alinea (Mixture-of-Steps, Single Objective) y sube el modelo en cuestión a HuggingFace. OBS: SE GENERA UN ÚNICO MODELO. ESTOS SON LOS REPORTADOS EN EL PAPER.

    ds_name = str ; Nombre en HuggingFace del dataset.
    output_dir = str ; Dirección de guardado del modelo (en el cluster de Helena)
    model_name = str ;  Nombre del checkpoint a alinear.
    model_hub_name = str ; Nombre para guardar el modelo el HF.
    """
    dataset = load_dataset(ds_name, split = 'train')
    aligner = k.DPO(model_name, output_dir)
    aligner.train_and_push(dataset, model_hub_name)
    print("--------------------------------------------------")

align('Kurosawama/Full_DPO_Llama-3.1-8B', '/media/discoexterno/francisco/modelos', 'meta-llama/Llama-3.1-8B', 'Kurosawama/Llama-3.1-8B-Full-align')
align('Kurosawama/Full_DPO_Llama-3.1-8B-Instruct', '/media/discoexterno/francisco/modelos', 'meta-llama/Llama-3.1-8B-Instruct', 'Kurosawama/Llama-3.1-8B-Instruct-Full-align')
align('Kurosawama/Full_DPO_Llama-3.2-3B', '/media/discoexterno/francisco/modelos', 'meta-llama/Llama-3.2-3B', 'Kurosawama/Llama-3.2-3B-Full-align')
align('Kurosawama/Full_DPO_Llama-3.2-3B-Instruct', '/media/discoexterno/francisco/modelos', 'meta-llama/Llama-3.2-3B-Instruct', 'Kurosawama/Llama-3.2-3B-Instruct-Full-align')
align('Kurosawama/Full_DPO_gemma-3-1b-it', '/media/discoexterno/francisco/modelos', 'google/gemma-3-1b-it', 'Kurosawama/gemma-3-1b-it-Full-align')


print('Final total. Grande Sabrina Carpenter.')