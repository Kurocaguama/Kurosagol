import torch, os
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from peft import LoraConfig, TaskType
from trl import DPOConfig, DPOTrainer
import utils

class DatasetManager:
	"""
	Actualmente replica el código de MSc 5 Getting DPO Dataset. Pero hay que optimizarlo.

	Hay que ejecutar el código presente en test.py para ver qué pingas.
	Dudas:
		1.- ¿Funcionan los cambios que le estamos haciendo al prompt? (prompt = prompt.format(_)) # SÍ XD pero se tiene que hacer bien flaco nmms
		2.- ¿Podemos tirarle un tqdm a la generación? # Chance pero zzzzzzzzzzzzz
	"""
	def __init__(self, dataset, model_id, max_new_tokens, prompt, stage):
		"""
		dataset = load_dataset('a/b', split = 'x') ; 
		model_id = str ; Model_id directo de HuggingFace.
		max_new_tokens = int ; Cantidad máxima de tokens para la generación.
		prompt = str ; Prompt definido de manera externa. Debe de permitir un .format(premise) (Detalles en test.py)
		stage = str ; Opciones: 'trans', 'infer', 'retrans'. Cada una determina el proceso en cuestión.
		"""
		# General details
		self.dataset = dataset
		self.max_new_tokens = max_new_tokens
		self.prompt = prompt
		self.clean_list = None # Is updated for ds creating purposes
		self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# Model details
		self.quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype = torch.bfloat16)
		self.gen_config = GenerationConfig.from_pretrained(model_id)
		self.tokenizer = AutoTokenizer.from_pretrained(model_id)
		self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config = self.quant_config).to(self.dev)
		self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

		# Generated lists per strategy
		self.greedy = []
		self.contrastive = []
		self.beam = []
		self.diverse_beam = []
		self.multinomial = []
		self.beam_multinomial = []

		# Stage of the pipeline
		self.stage = stage

	# Funciones auxiliares
	def unite_str(self, aux):
		string = ''
		for _ in aux:
			string += _ + ' '
		return string

	# Construcción del dataset de FOLIO		
	def clean_dataset(self):
		"""
		Limpia el dataset en cuestión. Regresa una lista ordenada sin repeticiones de premisas.
		"""
		if self.stage == 'trans':
			column = 'premises'
		if self.stage == 'infer':
			column = 'premises-FOL'
		if self.stage == 'retrans':
			column = 'conclusion-FOL'
		else:
			print('Etapa desconocida. Favor de redefinir')
 
		self.clean_list = [self.dataset[column][i].split('\n') for i in range(len(self.dataset[column]))]

		# Lo que hace esto es eliminar casos en donde se repiten premisas, no obstante puede ser problemático ya que se genera una cantidad dispareja de elementos.
		# Evitaremos eliminar los casos repetidos para evitar desfases entre conjuntos de datos.
		#ordered_list = []
		#ordered_list.append(premise_full[0])
		#for i in range(1, len(premise_full)):
		#	if premise_full[i] != ordered_list[-1]:
		#		ordered_list.append(premise_full[i])

		#self.clean_list = ordered_list
	
	def generation_with_strat(self, strategy):
		"""
		Se consideran las siguientes posibles estrategias de generación:
			1. Greedy Search (gs)
			2. Contrastive Search (cs)
			3. Beam Search (bs)
			4. Diverse Beam Search (dbs)
			5. Multinomial Sampling (ms)
			6. Beam Search + Multinomial Sampling (bsms)
		"""

		inputs = self.tokenizer(self.prompt, return_tensors = 'pt').to(self.dev)

		if strategy == 'gs':
			outputs = self.model.generate(**inputs, max_new_tokens = self.max_new_tokens)#.to(self.dev)
		elif strategy == 'cs':
			outputs = self.model.generate(**inputs, max_new_tokens = self.max_new_tokens, penalty_alpha = 0.6, top_k = 5)
		elif strategy == 'bs':
			outputs = self.model.generate(**inputs, max_new_tokens = self.max_new_tokens, num_beams = 3)
		elif strategy == 'dbs':
			outputs = self.model.generate(**inputs, max_new_tokens = self.max_new_tokens, num_beams = 3, num_beam_groups = 3, diversity_penalty = 1.0, do_sample = False)
		elif strategy == 'ms':
			outputs = self.model.generate(**inputs, max_new_tokens = self.max_new_tokens, num_beams = 1, do_sample = True)
		elif strategy == 'bsms':
			outputs = self.model.generate(**inputs, max_new_tokens = self.max_new_tokens, num_beams = 4, do_sample = True)

		answer = self.tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]
		answer = answer[len(self.prompt):]
		
		return answer
	
	def vector_generation(self):
		"""
		Genera un vector que contiene todos las posibles generaciones de un modelo 
		"""
		#gen_strats = ['gs', 'cs', 'bs', 'dbs', 'ms', 'bsms'] # You can use this line in case you want to add multiple generation strategies.
		gen_strats = ['gs']
		return [self.generation_with_strat(_) for _ in gen_strats]

	def gen_strats_list(self):
		"""
		Itera sobre el dataset limpio y agrega a los valores de la instancia de la clase.		
		"""
		og_prompt = self.prompt
		it = 0
		for _ in self.clean_list:
			self.prompt = og_prompt.format(_)
			llm_ans = self.vector_generation()
			self.greedy.append(llm_ans[0])

			# Uncomment the following lines if multiple generation strategies are used.
			#self.contrastive.append(llm_ans[1])
			#self.beam.append(llm_ans[2])
			#self.diverse_beam.append(llm_ans[3])
			#self.multinomial.append(llm_ans[4])
			#self.beam_multinomial.append(llm_ans[5])
			it += 1
			if it % 50 == 0:
				print('Iteración: {}'.format(it))
		self.prompt = og_prompt	


	def list_dict(self, nl_value, ds_value):
		"""
		Genera una lista compuesta por dos diccionarios. Este valor será una entrada para el DPO Dataset
		"""
		d1 = {'content': self.prompt.format(nl_value), 'role' : 'user'}
		d2 = {'content': ds_value, 'role': 'assistant'}
		return [d1, d2]

	#def good_dataset(self, ds_list, column):
	def good_dataset(self, ds_list):
		"""
		Genera un Dataset en formato de HuggingFace a partir de las listas provenientes del dataset y de las generaciones del modelo.
		"""

		if self.stage == 'trans':
			column = 'premises-FOL'
		if self.stage == 'infer':
			column = 'conclusion-FOL'
		if self.stage == 'retrans':
			column = 'conclusion'
		else:
			print('Etapa desconocida.')

		chosen = [self.list_dict(self.clean_list[i], self.dataset[column][i]) for i in range(len(self.clean_list))]
		#chosen = [self.list_dict(self.clean_list[i], self.dataset[column][i]) for i in range(len(self.clean_list))]
		rejected = [self.list_dict(self.clean_list[i], ds_list[i]) for i in range(len(self.clean_list))]

		# Queremos listas cuyo valor sea más alto entre más similares sean los pares de oraciones.
		pref_scores = [round(np.random.rand(1)[0], 3) + 8 if utils.get_sim(chosen[i], rejected[i]) > 0.5 else round(np.random.rand(1)[0], 3) + 7 for i in range(len(chosen))]
		bad_scores = [round(np.random.rand(1)[0], 3) + 4 if utils.get_sim(chosen[i], rejected[i]) > 0.5 else round(np.random.rand(1)[0], 3) + 3 for i in range(len(chosen))]

		preference_dictionary = {'chosen': chosen, 'rejected': rejected, 'score_chosen': pref_scores, 'score_rejected': bad_scores}
		return Dataset.from_pandas(pd.DataFrame(preference_dictionary))
	


class DPO:
	"""
	Clase para realizar el alineamiento de los modelos a partir de los conjuntos de preferencia.
	"""
	def __init__(self, model_id, output_dir, device_id=None):
		# Hiperparámetros
		#self.dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		if device_id is not None:
			self.dev = torch.device(f'cuda:{device_id}')
			os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
		else:
			self.dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
			if torch.cuda.is_available():
				os.environ["CUDA_VISIBLE_DEVICES"] = "0"

		self.quant_config = BitsAndBytesConfig(load_in_4bit = True, bnb_4bit_compute_dtype = torch.bfloat16)
		self.gen_config = GenerationConfig.from_pretrained(model_id)
		self.tokenizer = AutoTokenizer.from_pretrained(model_id)
		self.gen_config.pad_token_id = self.tokenizer.pad_token_id
		self.output_dir = output_dir
		self.lora_config = LoraConfig(
			task_type = TaskType.CAUSAL_LM,
			inference_mode = False,
			r = 8,
			lora_alpha = 32,
			lora_dropout = 0.1
		)
		self.tokenizer.chat_template = """
			<|im_start|>system
			{SYSTEM}<|im_end|>
			<|im_start|>user
			{INPUT}<|im_ed|>
			<|im_start|>assistant
			{OUTPUT}<|im_end|>
		"""

		# Instanciamos el modelo
		self.model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir = output_dir, quantization_config = self.quant_config, generation_config = self.gen_config).to(self.dev)
		self.model.add_adapter(self.lora_config, adapter_name = 'LoRa_Adapter')
		self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
		self.tokenizer.pad_token = self.tokenizer.eos_token

	def train_and_push(self, dataset, aligned_id):
		"""
			dataset = load_dataset(id) ; El conjunto de datos de preferencia
			aligned_id = str ; El nombre del modelo.
		"""
		training_args = DPOConfig(output_dir = self.output_dir, logging_steps = 30)
		trainer = DPOTrainer(model = self.model, args = training_args, processing_class = self.tokenizer, train_dataset = dataset)
		trainer.train()
		print('Fully trained. VM.')
		self.model.push_to_hub(aligned_id)
		print('Model in the hub. VM.')