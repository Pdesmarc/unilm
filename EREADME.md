Pour faire de l'inférence sur le test set:

- J'ai commenté l'import optimization_fp16.py dans le fichier __init__.py du dossier pytorch_pretrained_bert

- J'ai télécharger les 3 zip celui du modèle (qg_model.zip), des données (qg_data.zip) et de l'output attendu du test set (qg_output.zip)
J'ai mis les dossiers DONNEES et MODELE contenant les zip dézippés à la racine du projet unilm. Le zip output se trouve dans unilm/src/qg/output.

- nlg-eval est un package issu d'un repo github (cf README unilm) qui doit se trouver dans src. Je l'ai renommé NLGEval à cause du dash qui était pas pratique. Dans nlgeval on trouve pycocoevalcap puis les dossiers de chaque métrique bleu, rouge et meteor. J'ai du respécifier les chemins absolus de ces fichiers dans les scripts eval.py et eval_on_unilm_tokenized_ref.py
