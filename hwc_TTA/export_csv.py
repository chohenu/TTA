import pandas as pd 
import wandb
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("stickypanda03/VISDA-C", filters = {"group": 'VISDAC_online_ab_COMPOENT'})
# breakpoint()
summary_list, config_list, name_list = [], [], []
test_post_list = []
test_list = []
job_type_list = []
for run in runs: 
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    run_dict = run.summary._json_dict
    if 'Test Post Acc' in run_dict.keys(): 

        test_post_acc = run_dict['Test Post Acc']
        acc = run_dict['Test Acc']
        test_post_list.append(test_post_acc)
        test_list.append(acc)
        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
            if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)
        job_type_list.append(run.job_type)
    else: 
        pass

runs_df = pd.DataFrame({
    "job_type": job_type_list,
    "name": name_list,
    "test_post_acc": test_post_list,
    "test_acc": test_list
    })

runs_df.to_csv("project.csv")