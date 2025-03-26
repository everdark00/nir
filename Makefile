CONFIG_PATH = C:\Users/toppc/Documents/diploma/ptls-glove/exp_config.yaml
CONFIG_PATH_OLD = C:\Users/toppc/Documents/diploma/ptls-glove/exp_config_my_params.yaml

### mingw32-make test

test:
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=baseline
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_cat
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_num_emb_dist_common_emb_cat
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_sum
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=quant_100_dist_common_emb_mean
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=st_num_emb_dist_common_emb_cat
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=st_num_emb_dist_common_emb_mean
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=st_num_emb_dist_common_emb_sum
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=deeptlf9_disc_common_emb_cat
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=deeptlf9_disc_common_emb_sum
	python exp_pipeline.py $(CONFIG_PATH) --exp-name=deeptlf9_disc_common_emb_mean
	python exp_pipeline.py $(CONFIG_PATH_OLD) --exp-name=baseline_myparams
	python exp_pipeline.py $(CONFIG_PATH_OLD) --exp-name=quant_100_dist_common_emb_myparams_cat
	python exp_pipeline.py $(CONFIG_PATH_OLD) --exp-name=quant_num_emb_dist_common_emb_myparams_cat
	python exp_pipeline.py $(CONFIG_PATH_OLD) --exp-name=quant_100_dist_common_emb_myparams_sum
	python exp_pipeline.py $(CONFIG_PATH_OLD) --exp-name=quant_100_dist_common_emb_myparams_mean
	python exp_pipeline.py $(CONFIG_PATH_OLD) --exp-name=st_num_emb_dist_common_emb_myparams_cat
	python exp_pipeline.py $(CONFIG_PATH_OLD) --exp-name=st_num_emb_dist_common_emb_myparams_mean
	python exp_pipeline.py $(CONFIG_PATH_OLD) --exp-name=st_num_emb_dist_common_emb_myparams_sum
	python exp_pipeline.py $(CONFIG_PATH_OLD) --exp-name=deeptlf9_disc_common_emb_myparams_cat
	python exp_pipeline.py $(CONFIG_PATH_OLD) --exp-name=deeptlf9_disc_common_emb_myparams_sum
	python exp_pipeline.py $(CONFIG_PATH_OLD) --exp-name=deeptlf9_disc_common_emb_myparams_mean