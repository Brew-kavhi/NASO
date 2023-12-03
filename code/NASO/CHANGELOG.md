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
