import re
import pandas as pd
import numpy as np

# LogicSim+
def get_nice_dict(regex, instance, const):
	"""
		A partir de una regex, extrae todos los valores encontrados y los formatea para tener un diccionario con apariciones.

		regex = str ; Expresión regular a buscar.
		const = bool ; Determina si estamos analizando constantes. Esto permite que se realice un filtro extra en el código.
	"""
	lista = []
	for _ in instance:
		aux = re.finditer(regex, _)
		for expression in aux:
			if const:
				regex_list = expression.group()[1:-1]
				regex_list = regex_list.split(',')
				for j in regex_list:
					if len(j) > 1:
						constant = re.sub(' ', '', j)
						lista.append(constant)
			else:
				lista.append(expression.group())

	set_set = list(set(lista))
	dict_dict = {}
	for _ in set_set:
		dict_dict["{}".format(_)] = lista.count(_)

	return dict_dict


def extract_info(ds_answer, llm):
	"""
		Limpia la respuesta de un dataset y extrae las premisas necesarias para calcular LogicSim.

		llm = Bool ; Señala si se va a evaluar la respuesta generada por un LLM.
	"""
	instance = ds_answer.split('\n')
	
    # Este bloque permite limpiar las respuestas que siguen en exceso el prompt inicial.
	if llm: 
		for i in range(len(instance)):
			instance[i] = re.sub('(::)+([ A-z.]+)', '', instance[i])
			instance[i] = re.sub('(:::)+([ A-z.]+)', '', instance[i])
			instance[i] = re.sub('(  )+', '', instance[i])

	pred_dict = get_nice_dict(r'[A-z]+\(([A-z]+(,? [A-z]+)*)\)', instance, False)
	const_dict = get_nice_dict(r'(\([A-z]+(\, [A-z0-9]+){0,}\))', instance, True)
	logop_dict = get_nice_dict(r'[∀∧→⊕¬∨∃]+', instance, False)

	# len(instance) = Cantidad de premisas
	# pred_dict = Diccionario con cantidad de predicados. DE AQUÍ SE OBTIENE APARICIONES TOTALES Y CANTIDAD DE PREDICADOS
	# constant_dict =  Diccionario con cantidad de constantes. DE AQUÍ SE OBTIENE APARICIONES TOTALES Y CANTIDAD DE CONSTANTES
	# logop_dict =  Diccionario con cantidad de operadores y cuantificadores. DE AQUÍ SE OBTIENE APARICIONES TOTALES Y CANTIDAD DE CUANTIFICADORES
	return pred_dict, const_dict, logop_dict, len(instance)


def total_apps(dict1, preds):
    """
        Obtiene:
            1) La cantidad total de apariciones de un valor (Const/Pred/LogOps)
            2) La cantidad de constantes/predicados/logops distintos.

        dict1 = dict ; El diccionario obtenido previamente.
        preds = bool ; Valor booleano que permite realizar un filtro sobre los predicados.
    """
    if preds:
        lista_chida_aux = [re.search(r'([A-z]+\()', _).group() for _ in dict1.keys()]
        lista_chida = list(set(lista_chida_aux))

        # Guardamos la cantidad de apariciones en el diccionario previo.
        aux_dict = {"{}".format(_):0 for _ in lista_chida}
        for key in dict1.keys():
            for _ in lista_chida:
                if _ in key:
                    aux_dict["{}".format(_)] += dict1[key]
        
        if preds:
            aux_dict = {"{}".format(_[:-1]): aux_dict[_] for _ in aux_dict}
    else:
        aux_dict = dict1
    
    total_value = 0
    for _ in aux_dict:
        total_value += aux_dict[_]

    #total_value = Apariciones totales
    #len(aux_dict) = Cantidad de [VALUE] distinto.
    return total_value, len(aux_dict)


def logic_sim_plus_individual(llm_ans, folio_ans):
    """
        Extrae los valores distintos y valores totales de cada instancia. 
    """
    if llm_ans == 'nan':
        return llm_ans

    pred_llm, const_llm, logops_llm, len_llm = extract_info(llm_ans, True)
    pred_folio, const_folio, logops_folio, len_folio = extract_info(folio_ans, False)

    #llm
    dif_preds_llm, pred_count_llm = total_apps(pred_llm, True)
    dif_const_llm, const_count_llm = total_apps(const_llm, False)
    dif_logops_llm, logops_count_llm = total_apps(logops_llm, False)

    #folio
    dif_preds_folio, pred_count_folio = total_apps(pred_folio, True)
    dif_const_folio, const_count_folio = total_apps(const_folio, False)
    dif_logops_folio, logops_count_folio = total_apps(logops_folio, False)

    #Absolute Values
    dif_preds = abs(dif_preds_llm - dif_preds_folio)
    tot_aps_preds = abs(pred_count_llm - pred_count_folio)
    
    dif_const = abs(dif_const_llm - dif_const_folio)
    tot_aps_const = abs(const_count_llm - const_count_folio)

    dif_logops = abs(dif_logops_llm - dif_logops_folio)
    tot_aps_logops = abs(logops_count_llm - logops_count_folio)

    dif_premises = abs(len_llm - len_folio)

    logicsim_plus = dif_preds + tot_aps_preds + dif_const + tot_aps_const + dif_logops + tot_aps_logops + dif_premises
    return logicsim_plus


def clean_llm(dataset, value):
    """
        Filtra y deja bonito cada dataset para pasarlo por LogicSim+
    """
    text_value = str(dataset["Translation"][value])
    llm_ans = re.sub('\', \'', r'\n', text_value)
    llm_ans = re.sub(r'[\'\"\[\].]', '', llm_ans)
    llm_ans = re.sub(' , ', r'\n', llm_ans)
    llm_ans = re.sub('  ', '', llm_ans)
    return llm_ans


class Evaluator:
    def __init__(self, path, val_or_test):
        """
        path = str ; Directorio del dataset a trabajar
        val_or_test = bool ; Determina si se trabaja con el conjunto de validación o de prueba
        """
        if val_or_test:
            self.path = path.format('validation')
            self.folio = pd.read_json('/home/flopezp/Kurosagol/FOLIO/FOLIO/folio_validation.jsonl', lines = True)
            self.split = 'Validation'
        else:
            self.path = path.format('test')
            self.folio = pd.read_json('/home/flopezp/Kurosagol/FOLIO/FOLIO/folio_test.jsonl', lines = True)
            self.split = 'Test'

        self.dataset = pd.read_csv(self.path)
        if "Unnamed: 0" in self.dataset.columns:
            self.dataset = self.dataset.drop(columns = ["Unnamed: 0"])
        self.dataset = self.dataset.rename(columns = {"Retranslation": "Translation"})

        # Temp correction.
        if 'TRANS_Qwen3-14B-FP8_NEW' in path:
            self.dataset['Translation'][21] = self.dataset['Translation'][21][5:61]

        self.val_or_test = val_or_test
        self.checkpoint_name = path.split('/')[-1][6:]
        self.clean_premises = []
        self.logsim_list = []
        self.avg_logsim = 0

        print('\t --------------------------------------')
        print('\t Checkpoint: {}'.format(self.checkpoint_name))
        print('\t Split: {}'.format(self.split))
        print('\t --------------------------------------')

    def logicsim_list(self):
        for i in range(len(self.dataset)):
            llm_instance = clean_llm(self.dataset, i)
            self.clean_premises.append(llm_instance)
            folio_instance = self.folio['premises-FOL'][i]
            log_sim = logic_sim_plus_individual(llm_instance, folio_instance)
            self.logsim_list.append(log_sim)

    def logicsim_values(self):
        self.logicsim_list()
        aux = [int(x) for x in self.logsim_list if (x != None and x != 'nan')]
        print("\t =================================")
        print('\t \t 🤯 LOGICSIM+ 🤯')
        print("\t =================================")
        print('\t Cantidad total de NaN: {}'.format(len(self.logsim_list) - len(aux)))
        print("\t Avg:", round(np.mean(aux), 2), "Std:", round(np.std(aux), 2), "Var:", round(np.var(aux), 2))
        self.avg_logsim = round(np.mean(aux), 2)

    def show_outliers(self):
        """
        Muestra las premisas cuya traducción difiere de su contraparte de FOLIO por más del doble del promedio del ds.
        Si x es un conjunto de premisas del checkpoint, y es el equivalente de FOLIO, las presmisas (x, y) salen syss
        LogicSim(x, y) > avgLogicSimDS*2
        """
        assert len(self.logsim_list) != 0, 'You have to run .logicsim_values() first!'

        print('-------------------------------------------------------')
        print("El valor promedio de este checkpoint es {}".format(self.avg_logsim))
        print("Los siguientes pares de premisas son aquellos que se desvían más del doble que el promedio.")
        print('-------------------------------------------------------')
        
        #high_count = [self.logsim_list.index(value) for value in self.logsim_list if (value!='nan' and value > self.avg_logsim)]
        
        high_count = []
        for i in range(len(self.logsim_list)):
            aux = self.logsim_list[i]
            if aux != 'nan' and aux > self.avg_logsim:
                high_count.append(i)
        
        for value in high_count:
            print('LogicSim+: {}'.format(self.logsim_list[value]))
            print('-'*10, 'FOLIO', '-'*10)
            print(self.folio['premises-FOL'][value])
            print('--------')
            print('-'*10, 'LLM', '-'*10)
            print(clean_llm(self.dataset, value))
            print('=========')