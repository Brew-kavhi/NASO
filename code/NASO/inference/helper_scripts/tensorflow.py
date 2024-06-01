from helper_scripts.database import lock_safe_db_operation
from inference.celery.run_inference import run_inference
from inference.models.inference import Inference


def run_inference_from_run(run, gpu, queue):
    inference = Inference.objects.create(
        name=run.model_name,
        description=run.description,
        model_file=run.model_file,
        gpu=gpu,
        worker=queue,
        batch_size=run.evaluation_parameters.batch_size,
        dataset=run.dataset,
    )

    def save_metrics():
        inference.metrics.set(run.hyper_parameters.metrics.all())
        inference.callbacks.set(run.evaluation_parameters.callbacks.all())

    lock_safe_db_operation(save_metrics)
    # now start the inference run
    run_inference.apply_async(args=(inference.id,), queue=queue)
