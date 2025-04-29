class DatasetLoader:
	"""
	La idea va a ser que con las funciones definidas en esta clase
	podamos crear un nuevo dataset adecuado para DPO (eventualmente para el algoritmo que sea).

	Habrá que dejarlo bonito y consistente con la metodología de MSc-Thesis
	"""
	def __init__(self, dataset):
		self.dataset = dataset

	def get_chosen(self):
		return self.dataset['chosen']

	def get_rejected(self):
		return self.dataset['rejected']

	def get_length(self):
		return len(self.dataset)