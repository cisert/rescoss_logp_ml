.ONESHELL:
SHELL=/bin/bash
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate


# Commands
conda=conda
python=python

all: env saved_models prepare_data test

env: 
	${conda} env create -f env.yml 

saved_models: 
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.data_handling.download
	tar -xvzf saved_models.tar.gz

test: 
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m pytest --basetemp="./.pytest_scr" -rs tests/ -p no:warnings

prepare_data: 
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.data_handling.dataset_splitting --csv data/az_final.csv --output_basepath proc_data/az_set/dataset_splits --split_type random
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.data_handling.dataset_splitting --csv data/az_final.csv --output_basepath proc_data/az_set/dataset_splits --split_type scaffold

	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.data_handling.learning_curve_dataset_splitting --dataset_name az --split_type random
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.data_handling.learning_curve_dataset_splitting --dataset_name az --split_type scaffold

	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.descriptors az 2d

train_models: # takes a while
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model rf --dataset az --split_type random --target logp_rescoss --no_overwrite --auto_submit_if_all_experiments_done
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model lasso --dataset az --split_type random --target logp_rescoss --no_overwrite --auto_submit_if_all_experiments_done
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model xgb --dataset az --split_type random --target logp_rescoss --no_overwrite --auto_submit_if_all_experiments_done
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model chemprop --dataset az --split_type random --target logp_rescoss --no_overwrite --auto_submit_if_all_experiments_done

	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model rf --dataset az --split_type scaffold --target logp_rescoss --no_overwrite --auto_submit_if_all_experiments_done
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model lasso --dataset az --split_type scaffold --target logp_rescoss --no_overwrite --auto_submit_if_all_experiments_done
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model xgb --dataset az --split_type scaffold --target logp_rescoss --no_overwrite --auto_submit_if_all_experiments_done
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model chemprop --dataset az --split_type scaffold --target logp_rescoss --no_overwrite --auto_submit_if_all_experiments_done

	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model rf --dataset az --split_type random --target logp_exp --no_overwrite --auto_submit_if_all_experiments_done
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model lasso --dataset az --split_type random --target logp_exp --no_overwrite --auto_submit_if_all_experiments_done
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model xgb --dataset az --split_type random --target logp_exp --no_overwrite --auto_submit_if_all_experiments_done
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model chemprop --dataset az --split_type random --target logp_exp --no_overwrite --auto_submit_if_all_experiments_done

	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model rf --dataset az --split_type scaffold --target logp_exp --no_overwrite --auto_submit_if_all_experiments_done
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model lasso --dataset az --split_type scaffold --target logp_exp --no_overwrite --auto_submit_if_all_experiments_done
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model xgb --dataset az --split_type scaffold --target logp_exp --no_overwrite --auto_submit_if_all_experiments_done
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model chemprop --dataset az --split_type scaffold --target logp_exp --no_overwrite --auto_submit_if_all_experiments_done

learning_curves: # takes a while
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model rf --dataset az --split_type random --target logp_rescoss --no_overwrite --auto_submit_if_all_experiments_done --learning_curve
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model lasso --dataset az --split_type random --target logp_rescoss --no_overwrite --auto_submit_if_all_experiments_done --learning_curve
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model xgb --dataset az --split_type random --target logp_rescoss --no_overwrite --auto_submit_if_all_experiments_done --learning_curve
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model chemprop --dataset az --split_type random --target logp_rescoss --no_overwrite --auto_submit_if_all_experiments_done --learning_curve

	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model rf --dataset az --split_type scaffold --target logp_rescoss --no_overwrite --auto_submit_if_all_experiments_done --learning_curve
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model lasso --dataset az --split_type scaffold --target logp_rescoss --no_overwrite --auto_submit_if_all_experiments_done --learning_curve
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model xgb --dataset az --split_type scaffold --target logp_rescoss --no_overwrite --auto_submit_if_all_experiments_done --learning_curve
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model chemprop --dataset az --split_type scaffold --target logp_rescoss --no_overwrite --auto_submit_if_all_experiments_done --learning_curve

	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model rf --dataset az --split_type random --target logp_exp --no_overwrite --auto_submit_if_all_experiments_done --learning_curve
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model lasso --dataset az --split_type random --target logp_exp --no_overwrite --auto_submit_if_all_experiments_done --learning_curve
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model xgb --dataset az --split_type random --target logp_exp --no_overwrite --auto_submit_if_all_experiments_done --learning_curve
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model chemprop --dataset az --split_type random --target logp_exp --no_overwrite --auto_submit_if_all_experiments_done --learning_curve

	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model rf --dataset az --split_type scaffold --target logp_exp --no_overwrite --auto_submit_if_all_experiments_done --learning_curve
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model lasso --dataset az --split_type scaffold --target logp_exp --no_overwrite --auto_submit_if_all_experiments_done --learning_curve
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model xgb --dataset az --split_type scaffold --target logp_exp --no_overwrite --auto_submit_if_all_experiments_done --learning_curve
	$(CONDA_ACTIVATE) rescoss_logp_ml; ${python} -m rescoss_logp_ml.run_model_parallel --model chemprop --dataset az --split_type scaffold --target logp_exp --no_overwrite --auto_submit_if_all_experiments_done --learning_curve

clean: # remove data from pytest runs etc.
	rm -r proc_data/az_set/dataset_splits/random/inner_fold_*/results_rf_test_only
	rm -r proc_data/az_set/dataset_splits/random/results_rf_test_only
	rm -r *.pyc __pycache__

todo:
	grep "# TODO" */*.py | sed -e 's/    //g' | sed -e 's/# TODO//'