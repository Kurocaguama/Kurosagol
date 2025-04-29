class Ejemplo:
	def __init__(
		self,
		model_id
	):
		self.model_id = model_id
		self.params = ['hola', 'prueba', 'dinosaurios']

	def print_params(self):
		for _ in self.params:
			print(_)


def test_function(dataset):
	return dataset['chosen']

def length(dataset):
	return len(dataset)