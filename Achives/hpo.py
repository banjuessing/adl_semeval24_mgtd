import argparse
import json
import optuna
import sys
import multiprocessing
import pandas as pd
from functools import partial
from optuna.pruners import MedianPruner
from class_util import TextClassificationTrainer

def load_data(train_data_path, val_data_path):
    """
    Load training and validation data from given paths.
    """
    try:
        train_data = pd.read_json(train_data_path, lines=True)
        val_data = pd.read_json(val_data_path, lines=True)
        return train_data, val_data
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def save_config(config, path):
    """
    Save configuration to a file.
    """
    try:
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"Error saving configuration: {e}")
        sys.exit(1)


def objective(trial, config, train_data, val_data):
    fixed_params = config["fixed_params"]
    tunable_params = config["tunable_params"]

    for param_name, param_config in tunable_params.items():
        if param_config["type"] == "float":
            fixed_params[param_name] = trial.suggest_float(param_name, *param_config["args"])
        elif param_config["type"] == "int":
            fixed_params[param_name] = trial.suggest_int(param_name, *param_config["args"])
        elif param_config["type"] == "categorical":
            fixed_params[param_name] = trial.suggest_categorical(param_name, *param_config["args"])

    
    fixed_params["output_dir"] = fixed_params["output_dir"] + f"_trial_{trial.number}"
    fixed_params["logging_dir"] = fixed_params["logging_dir"] + f"_trial_{trial.number}"

    def metrics_processor(metrics):
        epoch = trainer.get_current_epoch()
        trial.report(metrics[fixed_params['eval_metric']], step=epoch)

    try:
        trainer = TextClassificationTrainer(fixed_params, train_data, val_data, metrics_processor)
        results = trainer.train()
        print(f"Trial {trial.number} RESULTS: {results}")
        metric = results[fixed_params['eval_metric']]

        return metric
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"Trial {trial.number} exceeded GPU VRAM. Skipping to next trial.")
            trial.report(float('inf'), step=0)
            raise optuna.TrialPruned()
        else:
            raise e   

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    train_data, val_data = load_data(config['fixed_params']['train_data_path'], config['fixed_params']['val_data_path'])

    try:
        pruner = MedianPruner(n_startup_trials=config['fixed_params']['n_startup_trials'], 
                              n_warmup_steps=config['fixed_params']['n_warmup_steps'], 
                              interval_steps=config['fixed_params']['interval_steps']
                              )
        study = optuna.create_study(direction="maximize", pruner=pruner)
        study.optimize(partial(objective, config=config, train_data=train_data, val_data=val_data), n_trials=config['fixed_params']['n_trials'])

        print("Best parameters:", study.best_params)

        study_results = []
        for trial in study.trials:
            study_results.append({
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': str(trial.state),
                'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                'duration': trial.duration.total_seconds() if trial.duration else None,
                'user_attrs': trial.user_attrs,
                'system_attrs': trial.system_attrs,
                'intermediate_values': trial.intermediate_values
            })

        save_config(study_results, config['fixed_params']["study_results"])

    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
