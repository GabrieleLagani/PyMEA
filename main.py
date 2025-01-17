from exp import run_experiment
import params as P


if __name__ == '__main__':
    run_experiment(name=P.EXP_NAME, dish_id=P.DISH_ID, dataset=P.EXP_DATASET, mode=P.EXP_MODE, device=P.EXP_DEVICE, seeds=P.EXP_SEEDS, dataseeds=P.EXP_DATASEEDS, restart=P.EXP_RESTART)

