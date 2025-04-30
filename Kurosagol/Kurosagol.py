import transformers, torch 
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig

class DatasetManager:
	"""
	Actualmente replica el código de MSc 5 Getting DPO Dataset. Pero hay que optimizarlo.

	Hay que ejecutar el código presente en test.py para ver qué pingas.
	Dudas:
		1.- ¿Funcionan los cambios que le estamos haciendo al prompt? (prompt = prompt.format(_))
		2.- ¿Podemos tirarle un tqdm a la generación?
	"""
	def __init__(self, dataset, model_id, max_new_tokens, prompt):
		"""
		dataset = load_dataset('a/b', split = 'x') ; 
		model_id = str ; Model_id directo de HuggingFace.
		max_new_tokens = int ; Cantidad máxima de tokens para la generación.
		prompt = str ; Prompt definido de manera externa. Debe de permitir un .format(premise) (Detalles en test.py)
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
		self.model = AutoModelForCausalLM.from_pretrained(model_id)
		self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

		# Generated lists per strategy
		self.greedy = None
		self.constrastive = None
		self.beam = None
		self.diverse_beam = None
		self.multinomial = None
		self.beam_multinomial = None

	# Shit test functions
	def get_chosen(self):
		return self.dataset['chosen']

	# Shit test functions 2
	def get_rejected(self):
		return self.dataset['rejected']

	# Shit test functions 3
	def get_length(self):
		print(len(self.dataset)) 

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
		premise_full = [self.dataset['premise'][i].split('\n') for i in range(len(self.dataset['premise']))]
		premise_list = [self.unite_str(premise_full[i]) for i in range(len(premise_full))]

		ordered_list = []
		ordered_list.append(premise_full[0])
		for i in range(1, len(premise_full)):
			if premise_full[i] != ordered_list[-1]:
				ordered_list.append(premise_full[i])

		self.clean_list = ordered_list
#		return ordered_list
	
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
			outputs = self.model.generate(**inputs, max_new_tokens = self.max_new_tokens)
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
		Genera un vector de 
		"""
		gen_strats = ['gs', 'cs', 'bs', 'dbs', 'ms', 'bsms']
		return [self.generation_with_strat(_) for _ in gen_strats]

	def gen_strats_list(self):
		"""
		Itera sobre el dataset limpio y agrega a los valores de la instancia de la clase.		
		"""
		for _ in self.clean_list:
			prompt = prompt.format(_)
			llm_ans = self.vector_generation()
			self.greedy.append(llm_ans[0])
			self.contrastive.append(llm_ans[1])
			self.beam.append(llm_ans[2])
			self.diverse_beam.append(llm_ans[3])
			self.multinomial.append(llm_ans[4])
			self.beam_multinomial.append(llm_ans[5])	


	def list_dict(self, nl_value, ds_value):
		d1 = {'content': self.prompt.format(nl_value), 'role' : 'user'}
		d2 = {'content': ds_value, 'role': 'assistant'}
		return [d1, d2]

	def good_dataset(self, ds_list):
		chosen = [self.list_dict(self.clean_list[i], self.dataset['premise'][i]) for i in range(len(self.clean_list))]
		rejected = [self.list_dict(self.clean_list[i], ds_list[i]) for i in range(len(self.clean_list))]
		pref_scores = [np.random.rand(1)[0] + 8 for i in range(len(self.clean_list))]
		bad_scores = [np.random.rand(1)[0] + 3 for i in range(len(self.clean_list))]

		preference_dictionary = {'chosen': chosen, 'rejected': rejected, 'score_chosen': pref_scores, 'score_rejected': bad_scores}
		return Dataset.from_pandas(pd.DataFrame(preference_dictionary))