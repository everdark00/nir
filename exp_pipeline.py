import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
import copy
import pickle
import sys
import os
from pdb import set_trace

import pytorch_lightning as pl

import logging
import click
import yaml
from dotsi import Dict

#sys.path.insert(0, os.path.dirname(os.getcwd()) +  '\\ptls')
#set_trace() 
from functools import partial
from ptls.nn import RnnSeqEncoder, TrxEncoder
from ptls.nn.trx_encoder.trx_encoders_custom import TrxEncoderGlove, TrxEncoderCat
from ptls.nn.trx_encoder import GloveEmbedding
from ptls.preprocessing.baseline_discretizer import KDiscretizer, SingleTreeDiscretizer
from ptls.preprocessing.deeptlf.src import DeepTLF
from ptls.frames.coles import CoLESModule
from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.frames.coles import ColesDataset
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.frames import PtlsDataModule, TestModule
from ptls.frames.coles.metric import BatchRecallTopK
from ptls.preprocessing.deeptlf import DeepTLFDisc

import ptls
import torch
from torch import nn
from ptls.preprocessing import PandasDataPreprocessor
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from ptls.data_load.datasets import inference_data_loader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def cuda_memory_clear():
    import gc
    gc.collect()
    torch.cuda.empty_cache()

def prepare_data_age_bins_scenario():
    data_path = '../data/age_bins'

    source_data = pd.read_csv(os.path.join(data_path, 'transactions_train.csv'))

    df_params = {
        "numeric_cols" : ["amount_rur"],
        "cat_cols" : ["small_group"],
        "date_col" : "trans_date",
        "cat_unique" : [],
        "id_col" : "client_id",
        "target" : "bins"
    }

    for f in df_params["cat_cols"] + [df_params["date_col"]]:
        df_params["cat_unique"].append(source_data[f].unique().shape[0])

    targets = pd.read_csv(os.path.join('../data/age_bins', 'train_target.csv'))    

    return source_data, targets, df_params

def prepare_data_gender_scenario():
    data_path = '../data/gender'

    source_data = pd.read_csv(os.path.join(data_path, 'transactions.csv'))
    source_data = source_data.drop(columns=["term_id"]).rename(columns={'customer_id' : 'client_id'})
    if 'Unnamed: 0' in source_data.columns:
        source_data = source_data.drop(columns=['Unnamed: 0'])

    source_data.tr_datetime = [int(i.split()[0]) for i in source_data.tr_datetime.values]

    df_params = {
        "numeric_cols" : ["amount"],
        "cat_cols" : ["mcc_code", "tr_type"],
        "cat_unique" : [],
        "date_col" : "tr_datetime",
        "id_col" : "client_id",
        "target" : "gender"
    }

    for f in df_params["cat_cols"] + [df_params["date_col"]]:
        df_params["cat_unique"].append(source_data[f].unique().shape[0])

    targets = pd.read_csv(os.path.join('../data/gender', 'gender_train.csv')).rename(columns={'customer_id' : 'client_id'})
    targets = source_data[['client_id']].drop_duplicates().merge(targets, on='client_id', how='left').dropna() 
    
    return source_data, targets, df_params

def init_disc(params, df_params, config):
    emb_size = None
    if params.fixed_emb:
        emb_size = config.model.embed_size
        
    if params.type in {'quantile', 'uniform', 'kmeans'}:
        disc = KDiscretizer(
            f_names = df_params['numeric_cols'],
            k_bins = params.k_bins,
            d_type = params.type,
            emb_sz = emb_size
        )
    elif params.type == 'st':
        disc = SingleTreeDiscretizer(
            f_names = df_params['numeric_cols'], 
            target_name = df_params['target'], 
            target_type = params.task_type, 
            k_bins = [params.k_bins],
            emb_sz = emb_size
        )
    elif params.type == 'deeptlf':
        disc = DeepTLFDisc({
          "n_est" : params.n_est,
          "min_freq" : params.min_freq,
          "features" : df_params['numeric_cols'] + df_params['cat_cols'],
          "features_to_split" : df_params['numeric_cols'],
        })
    else:
        raise Exception(f'No discretizer with name {params.type} availible')
    return disc

def get_basic_model_encoder(df_params, config):
    embeddings=dict()
    for i, f in enumerate(df_params["cat_cols"] + [df_params["date_col"]]):
        embeddings[f] = {'in' : df_params["cat_unique"][i], 'out' : config.model.embed_size}

    trx_encoder_params = dict(
        embeddings_noise=0.003,
        numeric_values=dict([(fe, 'identity') for fe in df_params['numeric_cols']]),
        embeddings=embeddings
    )
    
    seq_encoder = RnnSeqEncoder(
        trx_encoder=TrxEncoder(**trx_encoder_params),
        hidden_size=config.model.hidden_size,
        type=config.model.seq_encoder_type,
        bidir=False,
        trainable_starter='static'
    )
    
    return seq_encoder

def get_cat_encoder(df_params, agg_type, config, num_emb_flag=False):
    embeddings=dict()
    for i, f in enumerate(df_params["cat_cols"] + [df_params["date_col"]]):
        embeddings[f] = {'in' : df_params["cat_unique"][i], 'out' : config.model.embed_size}

    trx_encoder_params = dict(
        embeddings=embeddings,
        embeddings_noise=0.003,
        agg_type=agg_type,
        numeric_separate=num_emb_flag,
        numeric_features=df_params['numeric_cols']
    )
    
    seq_encoder = RnnSeqEncoder(
        trx_encoder=TrxEncoderCat(**trx_encoder_params),
        hidden_size=config.model.hidden_size,
        type=config.model.seq_encoder_type,
        bidir=False,
        trainable_starter='static'
    )
    return seq_encoder

def get_trans_encoder(df_params, agg_type, algo, config, numeric_separate=False):
    embeddings=dict()

    trx_encoder_params = dict(
        feature_names=df_params['cat_cols'] + [df_params["date_col"]], 
        in_emb_sizes=df_params["cat_unique"],
        out_emb_size=config.model.embed_size,
        agg_type=agg_type,
        numeric_separate=numeric_separate,
        numeric_features=df_params['numeric_cols']
    )
    
    seq_encoder = RnnSeqEncoder(
        trx_encoder=TrxEncoderTrans(**trx_encoder_params),
        hidden_size=config.model.hidden_size,
        type=config.model.seq_encoder_type,
        bidir=False,
        trainable_starter='static'
    )
    return seq_encoder

def get_glove_encoder(df_params, exp, glove_embedding, config):
    seq_encoder = RnnSeqEncoder(
        trx_encoder=TrxEncoderGlove(glove_embedding, agg_type=exp['agg_type'], numeric_separate=exp['nsep']),
        hidden_size=config.model.hidden_size,
        type=config.model.seq_encoder_type,
        bidir=False,
        trainable_starter='static'
    )
    return seq_encoder

def get_train_test_age_bins_scenario(df_params, train_embeds, test_embeds, train, test):
    data_path = "../data/age_bins"
    
    df_target = pd.read_csv(os.path.join(data_path, 'train_target.csv'))
    df_target = df_target.set_index(df_params["id_col"])
    df_target.rename(columns={"bins": "target"}, inplace=True)
    
    train_df = pd.DataFrame(data=train_embeds, columns=[f'embed_{i}' for i in range(train_embeds.shape[1])])
    train_df[df_params["id_col"]] = [x[df_params["id_col"]] for x in train]
    train_df = train_df.merge(df_target, how='left', on=df_params["id_col"])
    
    test_df = pd.DataFrame(data=test_embeds, columns=[f'embed_{i}' for i in range(test_embeds.shape[1])])
    test_df[df_params["id_col"]] = [x[df_params["id_col"]] for x in test]
    test_df = test_df.merge(df_target, how='left', on=df_params["id_col"])
    return train_df, test_df

def get_train_test_gender_scenario(df_params, train_embeds, test_embeds, train, test):
    data_path = "../data/gender"

    df_target = pd.read_csv(os.path.join('../data/gender', 'gender_train.csv')).drop(columns=['Unnamed: 0']).rename(columns={'customer_id' : 'client_id'})
    df_target = df_target.set_index(df_params["id_col"])
    df_target.rename(columns={"gender": "target"}, inplace=True)
    
    train_df = pd.DataFrame(data=train_embeds, columns=[f'embed_{i}' for i in range(train_embeds.shape[1])])
    train_df[df_params["id_col"]] = [x[df_params["id_col"]] for x in train]
    train_df = train_df.merge(df_target, how='left', on=df_params["id_col"])
    
    test_df = pd.DataFrame(data=test_embeds, columns=[f'embed_{i}' for i in range(test_embeds.shape[1])])
    test_df[df_params["id_col"]] = [x[df_params["id_col"]] for x in test]
    test_df = test_df.merge(df_target, how='left', on=df_params["id_col"])
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    return train_df, test_df


@click.command()
@click.argument('exp-config-path', type=click.Path(exists=True))
@click.option('--exp-name', default=None)
@click.option('--ds-name', default='gender')
@click.option('--mode', default='train-test')
def main(exp_config_path, exp_name, ds_name, mode):
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger().setLevel(logging.INFO)

    torch.set_float32_matmul_precision('high')
    
    logging.info(f'experiment {exp_name} started')
    with open(exp_config_path) as yf:
        config = Dict(yaml.full_load(yf))

    if ds_name == 'age_bins': 
        data, targets, df_params = prepare_data_age_bins_scenario()
    elif ds_name == 'gender': 
        data, targets, df_params = prepare_data_gender_scenario()
    else:
        raise Exception('Incorrect dataset name provided!')
    logging.info(f"{exp_name}: data loaded")

    exp = config.experiments[exp_name]

    disc = None
    if 'disc' in exp:
        disc = init_disc(exp.disc, df_params, config)
        if exp_name[:2] != "st" and ('ST' not in exp_name) and (disc is not None):
            disc.fit(data)
            data = disc.transform(data, to_embeds=exp['nemb'] if 'nemb' in exp else False)
        elif (disc is not None):
            disc.fit(data.sample(int(2e+5), random_state=42).merge(targets, on=df_params['id_col'], how='inner'))
            data = disc.transform(data, to_embeds=exp['nemb'] if 'nemb' in exp else False)
        logging.info(f"{exp_name}: data discretized")

    if 'nemb' in exp and not exp['nemb']:
        df_params['cat_cols'] =  df_params['numeric_cols'] + df_params['cat_cols']
        df_params["cat_unique"] = (disc.k_bins if (type(disc.k_bins) is list) else [disc.k_bins] * len(df_params['numeric_cols'])) + df_params["cat_unique"]
        df_params['numeric_cols'] = []
    else:
        nn = []
        for fn in df_params['numeric_cols']:
            nn += [fn + '_val', fn + '_pos']

    if not os.path.isfile(f"{config.prep_datasets_path}/{exp_name}_dataset_{ds_name}.pkl"):
        if 'glove_config' in exp:
            if not exp['nsep']:
                embedded_feats = df_params['numeric_cols'] + df_params['cat_cols'] + [df_params["date_col"]]
            else:
                embedded_feats = df_params['cat_cols'] + [df_params["date_col"]]
            folder_nm = f'{config.emb_path}/{exp_name}'[:-4] if exp['agg_type'] != 'mean' else f'{config.emb_path}/{exp_name}'[:-5]
            glove_embedding = GloveEmbedding(
                feature_names=embedded_feats,
                calculate_cooccur=False,
                embedding_folder=folder_nm,
                glove_params=exp['glove_config']
            )
            glove_embedding.load()
            data = glove_embedding.tokenize_data(data)
    
        preprocessor = PandasDataPreprocessor(
            col_id=df_params['id_col'],
            col_event_time=df_params['date_col'],
            event_time_transformation='none',
            category_transformation = 'none' if ('glove_config' in exp) else 'frequency',
            cols_category=df_params['cat_cols'],
            cols_numerical= nn if ('nemb' in exp and exp['nemb']) else df_params['numeric_cols'] ,
            return_records=True,
        )
    
        dataset = preprocessor.fit_transform(data)
    
        dataset = sorted(dataset, key=lambda x: x[df_params['id_col']])
    
        with open(f"{config.prep_datasets_path}/{exp_name}_dataset_{ds_name}.pkl", "wb") as fl:
            pickle.dump(dataset , fl)
        logging.info(f"{exp_name}: data preprocessed and saved")
    else:
        with open(f"{config.prep_datasets_path}/{exp_name}_dataset_{ds_name}.pkl", "rb") as fl:
            dataset = pickle.load(fl)
        logging.info(f"{exp_name}: data has been already preprocessed, load data")
    
    train, test = train_test_split(dataset, test_size=config.datasets[ds_name].test_split_coef, random_state=config.random_state)

    train, val = train_test_split(train, test_size=config.datasets[ds_name].val_split_coef, random_state=config.random_state)

    del dataset, data

    if mode == 'train' or mode == 'train-test':
        train_dl = PtlsDataModule(
            train_data = ColesDataset(
                    MemoryMapDataset(
                        data=train,
                        i_filters=[
                            SeqLenFilter(min_seq_len=config.train.data_loader.train.min_seq_len),
                        ],
                    ),
                    splitter=SampleSlices(
                        split_count=config.train.data_loader.train.split_count,
                        cnt_min=config.train.data_loader.train.cnt_min,
                        cnt_max=config.train.data_loader.train.cnt_max,
                    ),
                ),
            train_num_workers=config.train.data_loader.num_workers,
            train_batch_size=config.train.data_loader.train.batch_size,
            valid_data = ColesDataset(
                    MemoryMapDataset(
                        data=val,
                        i_filters=[
                            SeqLenFilter(min_seq_len=config.train.data_loader.val.min_seq_len),
                        ],
                    ),
                    splitter=SampleSlices(
                        split_count=config.train.data_loader.val.split_count,
                        cnt_min=config.train.data_loader.val.cnt_min,
                        cnt_max=config.train.data_loader.val.cnt_max,
                    ),
                ),
            valid_num_workers=config.train.data_loader.num_workers,
            valid_batch_size=config.train.data_loader.val.batch_size,
        )
    
        if exp.trx_encoder_type == 'cat':
            seq_encoder = get_cat_encoder(df_params, agg_type=exp.agg_type, config=config, num_emb_flag=exp.nemb)
        elif exp.trx_encoder_type == 'trans':
            seq_encoder = get_trans_encoder(df_params, agg_type=exp.agg_type, algo=exp.algo, config=config, numeric_separate=exp.nsep)
        elif exp.trx_encoder_type == 'glove':
            seq_encoder = get_glove_encoder(df_params, exp, glove_embedding, config=config)
        elif exp.trx_encoder_type == 'basic':
            seq_encoder = get_basic_model_encoder(df_params, config=config)
        else:
            raise Exception(f"No trx encoder with name {exp.trx_encoder_type}!")
    
        lr_scheduler = None
        if config.train.lr_scheduler.enabled:
            lr_scheduler = partial(torch.optim.lr_scheduler.StepLR, step_size=config.train.lr_scheduler.step_size, gamma=config.train.lr_scheduler.gamma)
        
        model = CoLESModule(
            seq_encoder=seq_encoder,
            optimizer_partial=partial(torch.optim.Adam, lr=config.train.lr, weight_decay=config.train.weight_decay),
            lr_scheduler_partial=lr_scheduler,
        )
    
        callbacks = []
        if config.train.early_stopping.enabled:
            callbacks.append(EarlyStopping(f'valid/{model.metric_name}', mode='max', patience=config.train.early_stopping.patience, min_delta=config.train.early_stopping.min_delta))
        
        trainer = pl.Trainer(
            max_epochs=config.train.max_epochs,
            accelerator=config.train.device,
            callbacks = callbacks,
            enable_progress_bar=True,
            enable_model_summary=False,
            logger=False
        )
    
        logging.info(f"{exp_name}: train starts")
    
        trainer.fit(model, train_dl)
        logging.info(trainer.logged_metrics)
    
        torch.save(seq_encoder.state_dict(), f"{config.models_path}/{exp_name}_{ds_name}.pt")
    
        logging.info(f"{exp_name}: train ended, model saved")

    if mode == 'test' or mode == 'train-test':
        if 'glove_config' in exp:
            if not exp['nsep']:
                embedded_feats = df_params['numeric_cols'] + df_params['cat_cols'] + [df_params["date_col"]]
            else:
                embedded_feats = df_params['cat_cols'] + [df_params["date_col"]]
            folder_nm = f'../glove_embeddings/{exp_name}'[:-4] if exp['agg_type'] != 'mean' else f'../glove_embeddings/{exp_name}'[:-5]
            glove_embedding = GloveEmbedding(
                feature_names=embedded_feats,
                calculate_cooccur=False,
                embedding_folder=folder_nm,
                glove_params=exp['glove_config']
            )
            glove_embedding.load()
            
        if exp.trx_encoder_type == 'cat':
            seq_encoder = get_cat_encoder(df_params, agg_type=exp.agg_type, config=config, num_emb_flag=exp.nemb)
        elif exp.trx_encoder_type == 'trans':
            seq_encoder = get_trans_encoder(df_params, agg_type=exp.agg_type, algo=exp.algo, config=config, numeric_separate=exp.nsep)
        elif exp.trx_encoder_type == 'glove':
            seq_encoder = get_glove_encoder(df_params, exp, glove_embedding, config=config)
        elif exp.trx_encoder_type == 'basic':
            seq_encoder = get_basic_model_encoder(df_params, config=config)
        else:
            raise Exception(f"No trx encoder with name {exp.trx_encoder_type}!")

        seq_encoder.load_state_dict(torch.load(f"{config.models_path}/{exp_name}_{ds_name}.pt", weights_only=True))

        res_metrics = []

        if config.test.recall_top_k.enable:
            metric = BatchRecallTopK(config.test.recall_top_k.data_loader.split_count - 1)

            if config.test.recall_top_k.calc_on_train : 
                datasets = [('train', train), ('test', test)]
            else:
                datasets = [('test', test)]
                
            for ds_nm, ds in datasets:
                dl = PtlsDataModule(
                    test_data = ColesDataset(
                            MemoryMapDataset(
                                data=ds,
                                i_filters=[
                                    SeqLenFilter(min_seq_len=config.test.recall_top_k.data_loader.min_seq_len),
                                ],
                            ),
                            splitter=SampleSlices(
                                split_count=config.test.recall_top_k.data_loader.split_count,
                                cnt_min=config.test.recall_top_k.data_loader.cnt_min,
                                cnt_max=config.test.recall_top_k.data_loader.cnt_max,
                            ),
                        ),
                    train_num_workers=config.test.num_workers,
                    train_batch_size=config.test.recall_top_k.data_loader.batch_size,
                )
        
                module = TestModule(
                    model = seq_encoder,
                    metrics = {"recall_top_k" : metric}
                )
            
                predictor = pl.Trainer(
                        accelerator=config.test.device,
                        enable_progress_bar=True,
                        enable_model_summary=False,
                        logger=False
                )
        
                predictor.predict(module, dl)
        
                ds_metrics = module.get_metrics()
        
                for m in ds_metrics:
                    res_metrics.append([exp_name, ds_nm, m, ds_metrics[m]])

        if config.test.proxy_metrics.enable:
            coles_model = CoLESModule(
                seq_encoder=seq_encoder,
            )
        
            inference_runner = pl.Trainer(
                accelerator=config.test.device,
                enable_progress_bar=True,
                enable_model_summary=False,
                logger=False
            )
        
            with torch.no_grad():
                cuda_memory_clear()
                train_dl = inference_data_loader(train, num_workers=config.test.num_workers, batch_size=config.test.proxy_metrics.batch_size)
                train_embeds = torch.vstack(inference_runner.predict(coles_model, train_dl))
                cuda_memory_clear()
                test_dl = inference_data_loader(test, num_workers=config.test.num_workers, batch_size=config.test.proxy_metrics.batch_size)
                test_embeds = torch.vstack(inference_runner.predict(coles_model, test_dl))

            if ds_name == 'age_bins': 
                train_df, test_df = get_train_test_age_bins_scenario(df_params, train_embeds, test_embeds, train, test)
            elif ds_name == 'gender': 
                train_df, test_df = get_train_test_gender_scenario(df_params, train_embeds, test_embeds, train, test)
            else:
                raise Exception(f"No raw dataset with name {ds_name} exists!")

            metrics = {}
            for m in config.test.proxy_metrics.metrics:
                if m == 'accuracy':
                    metrics[m] = accuracy_score
                elif m == 'roc_auc':
                    metrics[m] = roc_auc_score
                else:
                    raise Exception(f"No proxy metric with name {m} exists!")
                    
            logging.info(f"{exp_name}: proxy models eval started")
        
            for model_name, model_config in config.test.proxy_metrics.models.items():
                if 'basic' in model_config:
                    model_config = {}
                if model_name == 'lgbm_boosting':
                    clf = LGBMClassifier(**model_config)
                elif model_name == 'random_forest':
                    clf = RandomForestClassifier(**model_config)
                else:
                    raise Exception(f"No proxy model with name {model_name} exists!")

                avg_metrics_train = {i : 0 for i in metrics.keys()}
                avg_metrics_test = {i : 0 for i in metrics.keys()}

                for i in tqdm(range(config.test.proxy_metrics.n_trials)):
                    embed_columns = [x for x in train_df.columns if x.startswith('embed')]
                    x_train, y_train = train_df[embed_columns], train_df['target']
                    x_test, y_test = test_df[embed_columns], test_df['target']
                    
                    clf.fit(x_train, y_train)

                    for m_name, metric in metrics.items():
                        avg_metrics_test[m_name] += metric(y_test, clf.predict(x_test)) / config.test.proxy_metrics.n_trials
                    if config.test.proxy_metrics.calc_on_train:
                        for m_name, metric in metrics.items():
                            avg_metrics_train[m_name] += metric(y_train, clf.predict(x_train)) / config.test.proxy_metrics.n_trials

                for m_name, m_value in avg_metrics_test.items():
                    res_metrics.append([exp_name, 'test', f"{m_name}_{model_name}", m_value])
                
                if config.test.proxy_metrics.calc_on_train:
                    for m_name, m_value in avg_metrics_train.items():
                        res_metrics.append([exp_name, 'train', f"{m_name}_{model_name}", m_value])

                logging.info(f"{exp_name}: metrics via {model_name} calculated")
        
        report = pd.DataFrame(res_metrics, columns = ['exp_name', 'dataset', 'metric', 'value'])
        if os.path.isfile(f"{config.report_path}/{config.test.report_name}_{ds_name}.csv"):
            prev_report = pd.read_csv(f"{config.report_path}/{config.test.report_name}_{ds_name}.csv").drop(columns=['Unnamed: 0'])
            pd.concat([prev_report, report]).to_csv(f'{config.report_path}/{config.test.report_name}_{ds_name}.csv')
        else:
            report.to_csv(f'{config.report_path}/{config.test.report_name}_{ds_name}.csv')         
            

if __name__=="__main__":
    main()