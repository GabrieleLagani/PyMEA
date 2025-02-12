from json_data_convert import run_experiment
import params as P


if __name__ == '__main__':
    run_experiment(name=P.EXP_NAME, dataset=P.EXP_DATASET, mode=P.EXP_MODE, seeds=P.EXP_SEEDS, dataseeds=P.EXP_DATASEEDS)

