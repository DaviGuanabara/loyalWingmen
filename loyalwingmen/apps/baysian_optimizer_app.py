import os
import sys
sys.path.append("..")
import logging
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from modules.environments.demo_env import DemoEnvironment
from modules.models.policy import CustomActorCriticPolicy, CustomCNN
from modules.factories.callback_factory import callbacklist, CallbackType
from typing import List, Tuple
from datetime import datetime
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from openpyxl import load_workbook, Workbook
from openpyxl.worksheet.worksheet import Worksheet
import warnings
from gymnasium import Env
from gymnasium import spaces, Env
from typing import Optional, Union

# Ignorar o aviso específico
#warnings.filterwarnings("ignore", message="WARN: Box bound precision lowered by casting to float32")
warnings.filterwarnings("ignore", category=UserWarning)

# Seu código aqui


# Configurando o logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_output_folder(experiment_name: str):
    # Obter a data atual
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Criar o nome da pasta com base na data atual
    #folder_name = f"output/{experiment_name}"
    folder_name = os.path.join("output", experiment_name)

    # Criar a pasta se ainda não existir
    os.makedirs(folder_name, exist_ok=True)

    # Retornar o caminho completo para a pasta
    return folder_name


def create_vectorized_environment(n_envs: int, frequency: int) -> VecMonitor:
    
    #TODO: type hint problem. but thats not a problem at al.

    env_fns = [lambda: DemoEnvironment(rl_frequency=frequency) for _ in range(n_envs)]
    vectorized_environment = SubprocVecEnv(env_fns) # type: ignore
    vectorized_environment = VecMonitor(vectorized_environment)
    return vectorized_environment

def create_callback_list(n_envs: int, vectorized_environment: VecMonitor, model_dir: str, log_dir: str, callbacks_to_include: List[CallbackType] = [CallbackType.EVAL, CallbackType.CHECKPOINT, CallbackType.PROGRESSBAR]):
    
    #log_path = "./logs/"
    #model_path = "./models/"
    save_freq = 100_000
    callback_list = callbacklist(
        vectorized_environment,
        log_path=log_dir,
        model_path=model_dir,
        n_envs=n_envs,
        save_freq=save_freq,
        callbacks_to_include=callbacks_to_include
    )
    return callback_list

def create_policy_kwargs(hiddens: list, learning_rate: float) -> dict:
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
        normalize_images=False,
        net_arch=dict(pi=hiddens, vf=hiddens)
    )
    return policy_kwargs

def create_model(vectorized_environment: VecMonitor, policy_kwargs: dict, learning_rate: float) -> PPO:
    tensorboard_log = "./logs/my_first_env/"
    model = PPO(
        CustomActorCriticPolicy,
        vectorized_environment,
        verbose=0,
        device="auto",
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
    )
    return model

def train_model(model: PPO, callback_list, n_timesteps: int = 1_000_000) -> PPO:
    model.learn(total_timesteps=n_timesteps, callback=callback_list)
    return model

def save_model(model: PPO, hiddens: list, frequency: int, learning_rate: float, output_folder: str):
    folder_name = f"model_{hiddens}_{frequency}_{learning_rate}"
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(output_folder, f"models_{folder_name}_{current_time}")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "my_model")
    model.save(model_path)
    logging.info(f"Model saved at: {model_path}")

def evaluate_with_dynamic_episodes(model, env, max_episodes=500, target_std=0.1, tolerance=0.01, deterministic=True):
    """
    Evaluate the performance of a reinforcement learning model on a given environment with dynamically adjusted
    evaluation episodes.

    Args:
        model (BaseAlgorithm): The trained reinforcement learning agent model to be evaluated.
        env (gym.Env): The evaluation environment compatible with Stable Baselines3.
        max_episodes (int, optional): The maximum number of evaluation episodes to run. Defaults to 500.
        target_std (float, optional): The target standard deviation of rewards for performance convergence.
                                      Defaults to 0.1.
        tolerance (float, optional): The tolerance level for considering performance convergence. Defaults to 0.01.
        deterministic (bool, optional): If True, the agent will use deterministic actions during evaluation.
                                        Defaults to True.

    Returns:
        tuple: A tuple containing the average reward, standard deviation of rewards, and the final number of episodes
               used for evaluation.

    """
    num_episodes = 10  # Start with a small number of episodes
    avg_rewards, std_devs = evaluate_policy(model, env, n_eval_episodes=num_episodes, deterministic=deterministic)
    avg_reward = sum(avg_rewards) / len(avg_rewards) if isinstance(avg_rewards, list) else avg_rewards
    std_dev: float = sum(std_devs) / len(std_devs) if isinstance(std_devs, list) else std_devs
    
    # Continue evaluating until the standard deviation is below the target threshold or the maximum episodes is reached
    while std_dev > target_std and num_episodes < max_episodes:
        num_episodes *= 2  # Double the number of episodes
        avg_rewards, std_devs = evaluate_policy(model, env, n_eval_episodes=num_episodes, deterministic=deterministic)
        avg_reward = sum(avg_rewards) / len(avg_rewards) if isinstance(avg_rewards, list) else avg_rewards
        std_dev: float = sum(std_devs) / len(std_devs) if isinstance(std_devs, list) else std_devs

        # Check for convergence within the specified tolerance
        if abs(std_dev - target_std) < tolerance:
            break

    return avg_reward, std_dev, num_episodes

def cross_validation_simulation(hiddens: list, frequency: int, learning_rate: float, num_evaluations: int, output_folder: str, n_timesteps: int) -> float:
    number_of_logical_cores = os.cpu_count()
    n_envs: int = number_of_logical_cores if number_of_logical_cores is not None else 1

    vectorized_environment: VecMonitor = create_vectorized_environment(n_envs, frequency)

    model_dir = os.path.join(output_folder, 'models')
    log_dir = os.path.join(output_folder, 'logs')
    callback_list= create_callback_list(n_envs, vectorized_environment, model_dir=model_dir, log_dir=log_dir, callbacks_to_include=[CallbackType.PROGRESSBAR])
    policy_kwargs = create_policy_kwargs(hiddens, learning_rate)
    model = create_model(vectorized_environment, policy_kwargs, learning_rate)

    logging.info(model.policy)
    model = train_model(model, callback_list, n_timesteps)
    save_model(model, hiddens, frequency, learning_rate, output_folder)
    
    avg_reward, std_dev, num_episodes = evaluate_with_dynamic_episodes(model, vectorized_environment)
    logging.info(f"Avg score: {avg_reward}")
    return avg_reward

def generate_random_parameters() -> Tuple[list, int, float]:
    hidden_dist = randint(10, 1000)
    num_hiddens = randint(3, 8).rvs()
    hiddens = [hidden_dist.rvs() for _ in range(num_hiddens)]
    frequency = (randint(1, 8).rvs() * 15)
    learning_rate = uniform(0.00000000001, 0.1).rvs() 
    return hiddens, frequency, learning_rate





def save_results_to_excel(results, output_folder, file_name):
    # Construir o caminho completo do arquivo
    file_path = os.path.join(output_folder, file_name)

    # Verificar se o diretório de saída existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Verificar se o arquivo existe
    if not os.path.isfile(file_path):
        # Se o arquivo não existe, criar uma nova planilha
        workbook = Workbook()
        #TODO: possible cast error (_WorkBookChild | None to Worksheet)
        sheet: Worksheet = workbook.active # type: ignore
        
        # Definir os cabeçalhos das colunas
        sheet.append(['hiddens', 'frequency', 'learning_rate', 'score'])    
    else:
        # Carregar a planilha existente
        workbook = load_workbook(file_path)
        # Obter a planilha ativa
        #TODO: possible cast error (_WorkBookChild | None to Worksheet)
        sheet: Worksheet = workbook.active  # type: ignore
        
    for result in results:
        hiddens_str = ', '.join(str(x) for x in result[0])
        sheet.append([hiddens_str, result[1], result[2], result[3]])

    try:
        # Salvar as alterações no arquivo
        workbook.save(file_path)
    except PermissionError:
        print(f"Não foi possível salvar o arquivo '{file_name}'. Permissão negada.")
    except Exception as e:
        print(f"Erro ocorrido ao salvar os resultados: {e}")    

    # Adicionar os novos resultados
    #for result in results:
    #    sheet.append(result)






def objective(trial: Trial, output_folder: str, n_timesteps: int) -> float:
    
    num_hiddens = trial.suggest_int('num_hiddens', 3, 8)
    hiddens = [trial.suggest_int(f'hiddens_{i}', 100, 1000) for i in range(num_hiddens)]
    frequency = trial.suggest_int('frequency', 1, 8) * 15
    exponent = trial.suggest_uniform('exponent', -10, -1)
    learning_rate = 10 ** exponent

    
    print("Parameters:")
    print("Hiddens: ", hiddens)
    print("Frequency: ", frequency)
    print("Learning Rate: ", learning_rate)

    avg_score = cross_validation_simulation(hiddens, frequency, learning_rate, num_evaluations=100, output_folder=output_folder, n_timesteps=n_timesteps)

    # Salvar os resultados na planilha
    result = (hiddens, frequency, learning_rate, avg_score)
    save_results_to_excel([result], output_folder, 'simulation_results.xlsx')

    return avg_score

def print_best_parameters(results: List[Tuple[List[int], int, float, float]]):
    results = sorted(results, key=lambda x: x[-1], reverse=True)
    print("Best score: %.4f" % results[0][-1])
    print("Best parameters:")
    print("Hiddens: %s" % ', '.join(map(str, results[0][0])))
    print("Frequency: %.4f" % results[0][1])
    print("Learning rate: %.10f" % results[0][2])

def main():
    
    
    n_timesteps = 1_000_000
    experiment_name = "Optimizer_baysian_app"
    # Criar a pasta de saída
    output_folder = create_output_folder(experiment_name)

    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    study.optimize(lambda trial: objective(trial, output_folder, n_timesteps), n_trials=100)

    results = []
    for trial in study.trials:
        hiddens = [trial.params[f'hiddens_{i}'] for i in range(trial.params['num_hiddens'])]
        results.append((hiddens, trial.params['frequency'], trial.params['learning_rate'], trial.value))

    #save_results_to_excel(results)
    print_best_parameters(results)

if __name__ == "__main__":
    main()

#375, 103, 702, 296	105	0,000424858	8506,027344
