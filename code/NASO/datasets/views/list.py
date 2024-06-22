from django.views.generic.base import TemplateView

from naso.models.page import PageSetup
from neural_architecture.models.dataset import Dataset, DatasetLoader


class ListDatasets(TemplateView):
    template_name = "datasets/list.html"
    page = PageSetup(title="Datasets", description="List")
    context = {"page": page.get_context()}

    def get(self, request, *args, **kwargs):
        datasets = []
        dataset_loaders = DatasetLoader.objects.all()
        for loader in dataset_loaders:
            # get all the datasets this loader has to offer
            for loader_set in loader.get_datasets():
                # check if this dataset has alrady been used once
                # because if it has we can get the informatiom on it.
                dataset = {
                    "id": (str(loader.id) + loader_set)
                    .replace(" ", "_")
                    .replace(")", "")
                    .replace("(", ""),
                    "name": loader_set,
                    "module": loader.module_name,
                    "description": loader.description,
                    "size": "Unknown",
                    "info": "",
                }
                if Dataset.objects.filter(
                    name=loader_set, dataset_loader=loader
                ).exists():
                    dataset["info"] = loader.dataset_loader.get_info(loader_set)
                    try:
                        dataset["img"] = loader.dataset_loader.get_sample_images(
                            loader_set
                        )
                    except Exception as exc:
                        dataset["img"] = str(exc)
                        print("no img")
                        print(exc)
                    if "size" in dataset["info"]:
                        dataset["size"] = dataset["info"]["size"]
                    datasets.insert(0, dataset)
                else:
                    datasets.append(dataset)

        self.context["datasets"] = datasets
        return self.render_to_response(self.context)
