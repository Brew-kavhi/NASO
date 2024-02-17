## 1.0.0 (2024-02-17)

### Breaking Changes
- **Metrics**: Now using physically correct naming power and energy measurement
- **Pruning**: rebuild the model and apply pruning based on layers
- **Model File**: Storing the models in modern .keras format now

### Feat

- **comparison**: add prediction metrics to graph
- **energy_consumption**: compute energy consumption as well
- **prediction**: include memory usage in prediction metrics
- **inference**: run inference on any stored tensorflow model
- **trial**: set name and description
- **graphs**: show labels on all plots
- **comparison**: modify comparisons
- **autokeras**: plot energy consumption
- **runs_list**: sort by name
- **graphs**: enable zoomingon important graphs
- **comparison**: add plot for point metric
- **model_size**: save size of model file
- **comparison**: reset session with comparisons
- **runs_list**: display model size in details
- **comparison**: introduce comparison model and view
- **comparison**: show everything in one plot
- **details**: show size on disk for runs
- **device_name**: saving human readable device name
- **safe_delete**: enablre restoriation and harddelete
- **comparison**: showing opatimizer and pruning properties
- **tensorflow**: save disk size and choose from list of models
- **pruning**: save zippped model to the disk
- **pruning**: measuring sparsity and fixing layers prunable
- **comparison**: implemented comparison view for runs
- **comparison**: select the runs
- **comparison**: select the runs
- **runs**: edit run details
- **pruning**: show details in run page
- **memory**: measure memory usage on gpu

### Fix

- **timing**: measure time in first separate callback
- **reruninig**: wrong parameter name
- **autokeras**: extra field for storing path to model files
- **deletion**: delete model files when deleting the run
- **TRIALS**: load trial model in celery worker because  of memory_leak
- **autokeras**: trial overview chart fix height
- **runs_list**: load the charts only when cards expand
- **trial**: when loading a trial also try load ing weights
- **dashboard**: link for queued tasks
- **celery**: restart the workers always
- **pruning**: catch some errors in pruning initialization
- **filesize**: use correct function to get filesize of network
- **celery**: initial_load of celery workers was broken
- **autokeras_tuner**: add argument
- **autokeras**: max_model_size was ignored
- **sparsity**: return 0 if no pruning
- **memory_usage**: memory cannotbe measured on CPU
- **rerun**: load data from the selected model
- **get_or_create**: for some objects dont use it
- **get_or_create**: found another get or create
- **database_lock**: catch lock error and try again
- **importing**: fixed further importing errors
- **count_params**: size is only determined by trainable_weights
- **timeouterror**: error occured often on reloading
- **trial_run**: fixed some bugs for running trials
- **celery**: bugfix when run not found
- **comparison**: key in session was missing
- **comparison**: session key error
- **celery_api**: no active tasks return Response
- **keras_trial**: build pruning model in each step
- **kerastrial**: fix pruning for keras model runs
- **autokeras**: cannot prune last autokeras layer
- **selection**: delete autokeras wrong url
- **autokeras**: start timer for autokeras run
- **design**: sticky side and head bar
- **comparison**: keyerror in session
- **prediction**: use export model for prediction
- **all**: proper number formatting
- **comparison**: ids for runs were not unique
- **reset_graph**: use correct function to reset
- **runs**: reset tensorflow graph after run
- **runs_list**: tags showed the evaluation metrics
- **runs**: delete modal for run shows name
- **auto-import**: read dtypes from annotations
- **runs**: update state

### Refactor

- **energy**: correct power and energy naming
- **build_model**: use one function for all models to build
- **save_model**: use modern keras format
- **pruning**: pruning is now on layer base
- **keras_model_runs**: use tensorflow runs to run autokeras trials

## 0.4.6 (2023-12-20)

### Fix

- **pruning**: parameters

## 0.4.5 (2023-12-20)

### Feat

- **runs**: measure energy constantly
- **plugins**: kaggle plugin offers deepsat dataset
- **energy**: show total energy consumption of runs
- **runs**: added Soft delete
- **runs**: filter runs by rating
- **gpu**: display name of the device in seletion
- **plugin**: added Kaggledataset loader
- **worker**: task states in run details page and worker control

### Fix

- **autokeras**: best trial in details page
- **celery**: prefetching for multiple workers
- **runs**: strong of autokeras trials
- **style**: runs list adaptive for smaller screens
- **ci**: set upstream branch
- **CI**: push tag to main
- **CI**: commitizen arguments
- **CI**: ignore NoIncrementError on Commitizen bump
- **CI**: install commitizen adn bump version
- **importing**: null objects
- fixed rating api, new tensorflow run and task-details api

## 0.3.0 (2023-12-04)

### Feat

- **worker**: task states in run details page and worker control

### Refactor

- **header**: mov4ed logs in header and state bar of workers

## 0.2.0 (2023-12-03)

### Feat

- **runs**: ratings and description
- **formatting**: html formatting with djlint
- **templates**: administration of templates
- **dataset**: sizes are dynamically retrieved from the dataset
- **details**: details page for tensorflow run
- **pruning**: autokeras pruning implemented
- **logs**: implemented api for logs
- **gpu**: trainings can be run on selected gpu now
- **prediction**: included callbacks in prediction to store metrics
- **callbacks**: implemented callbacks for tensorflow run"
- **pruning**: enable for tensorflow runs
- **fine tuning trial**: loading weights and metrics and callbacks
- **tine tuning trial**: trial can be rerun now with the same config
- **fine tune trial**: implemented function to get the model for a trial
- **metrics weights**: stacked bar chart for metrics
- **graphs**: added tooltips and proper xAxis handling
- **csv**: added download button
- **details page**: last fixes
- **logging**: loggins the trial id now a well
- **metric weights**: display a proper settings for mertic weights and objectives
- **detail page**: bug fixes in model delete, in model dircetory, and in trial_id datatype and extend page setup object
- **details page**: added autokeras trial graphs
- **details page**: fixes
- **logging**: loggins the trial id now a well
- **metric weights**: display a proper settings for mertic weights and objectives
- **details page**: implementing api
- **page**: implementing details page
- **datasets**: added sklearn dataset loader
- **energy**: added energy measurement as callback
- **dataset loaders**: added dataset loaders so plugisn can register for custo data loading
- **network templates**: added possibility to save network config as reloadable template
- **kill task**: can kill task and show pending tasks on dashboard
- **model size**: added model size metric
- **callbacks integration**: added types and function
- **autokeras metrics**: integrated metrics and a summed objective
- **plugin-system**: integrated basic plugin functioniality;
- **autokeras integration**: implemented autokerasrun model
- **autokeras-logs**: logging callback for autokeras runs
- **autokeras integration**: added list view of autokeras runs
- **autokeras integration**: implemented view for autokeras runs
- started working on integration of autokeras

### Fix

- dashboard had a nug wih autokeras_trial, sidebar indent and objective selection
- multiple loads of javasript
- **new run**: build model from graph was wrong because of herad-check
- **performance**: implemtned api for runs list
- **metrics**: fixed metric weights not showing when reruning
- pip requirements, dataset call and loss html in the new tensorflow run form
- used metrics in tensorflow runs and other beautifications
- **callbacks**: fixed the metrics calculation in the callbacks
- code style
- **reruns**: naode ids and instances additional_arguments
- rerun
- importing arguments
- **migrations**: removed verbose fields htat mde problems when migrting
- AutoKeras models
