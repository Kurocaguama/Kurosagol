#class Ejemplo:
#	def __init__(
#		self,
#		model_id
#	):
#		self.model_id = model_id
#		self.params = ['hola', 'prueba', 'dinosaurios']
#
#	def print_params(self):
#		for _ in self.params:
#			print(_)


class DatasetLoader:
	def __init__(self, dataset):
		self.dataset = dataset

	def test_function(self, dataset):
		return dataset['chosen']

	def test_length(self, dataset):
		return len(dataset)

#def test_function(dataset):
#	return dataset['chosen']

#def length(dataset):
#	return len(dataset)
