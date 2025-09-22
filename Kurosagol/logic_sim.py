import Kurosagol as k

eval = k.Evaluation('Kurosawama/EVALUATION_trans_Llama-3.1-8B', 'train')

print(eval.ds)

# HAY QUE CAMBIAR EL NOMBRE DE LOS DATASETS ME LLEVA LA CHINGADA
eval.logic_sim()