"""
Ejemplo de uso de la paquetería para generar los datasets de DPO.

1.- El prompt se debe de declarar antes de pasar la instancia del DatasetManager. Aquí es cuando se determina si es una estrategia con zero-shot, one-shot, o few-shot.
"""

import Kurosagol as k
from datasets import load_dataset

aux = load_dataset('yale-nlp/FOLIO', split = 'train')

# CORRECCIONES: Modificar el prompt.
prompt = 'You are Terry Tao, translate the following logic premises: {}'

# Declaramos la instancia del DatasetManager con los parámetros necesarios (dataset['columna'], model_id, max_new_tokens)
test = k.DatasetManager(aux['premises'], 'meta-llama/Llama-3.2-1B', 100, prompt)

# Con esto creamos un dataset basado en Greedy Generation. Hay que revisar si se puede optimizar la generación y creación del dataset
#greedy_search_dataset = test.good_dataset(test.greedy)
#greedy_search_dataset.push_to_hub('Kurosawama/nombre_bastardo', private = True)