CoLES TrxEncoder modification  
   
Ноутбуки   
   
coles-test-pipeline.ipynb - ноутбук с тестовым пайплайном   
emb_size_analysis.ipynb - ноутбук с исследованием влияния размера эмбеддинга на качество модели   
coles_discretizer_analysis.ipynb - ноутбук с исследованием дискретизаторов   
   
Результаты   
   
age_bins_metrics.xlsx - результаты экспериментов на датасете age_bins, подробнее что есть что описано в отчете   
gender_metrics.xlsx - результаты экспериментов на датасете gender   
nir_3sem_report.docx - отчет   

Исходный код   
   
ptls.nn.trx_encoder.trx_encoders_custom.py - реализации trx_encoder которые я использовал в экспериментах   
    - TrxEncoderGlove - glove emb + discretization   
    - TrxEncoderCat - noisy emb (basic) + discretization (PLE optionally)    
    - TrxEncoderTran - альтернативный подход к кодированию, описан в конце отчета   
   
ptls.nn.trx_encoder.custom_embeddings.py - содержит дополнительные классы эмбеддингов для кастомных trx_encoder   
   
ptls.preprocessing.baseline_discretizer.k_discretizer.py - обертка над sklearn.KBinsDiscretizer, из которого    использовались реализации Quant, KMeans и Uniform   
   
ptls.preprocessing.baseline_discretizer.single_tree_discretizer.py - дискретизатор SingleTree   
   
ptls.preprocessing.deeptlf - дискретизатор DeepTLF и связанные классы   
    

Пайплайн обучения и тестирования
    
exp_pipeline.py - скрипт    
exp_config.yaml - конфиг из тестового ноутбука из репозитория pytorch lifestream    
exp_config_my_params.yaml - конфиг на котором я проводил эксперименты    
Makefile - мейкфайл с экспериментами    

Пример эксперимента:    
    exp_pipeline.py $(CONFIG_PATH) --exp-name=<experiment-name> --ds-name=<age_bins / gender> --mode=<train-test / train / test>