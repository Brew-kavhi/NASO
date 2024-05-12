import os

import matplotlib.pyplot as plt
import numpy as np
from django.http import HttpResponse
from django.shortcuts import redirect
from django.views.generic.base import TemplateView
from mpl_toolkits.mplot3d import Axes3D

from api.views.comparisons import clear_session
from comparisons.forms.create import SaveSession
from comparisons.forms.edit import AddRunForm
from comparisons.models.comparison import Comparison, get_comparison_details
from naso import settings
from naso.models.page import PageSetup
from runs.forms.base import get_gpu_choices


def plot_points(request):
    # Example 3D points
    loss = np.array([0.2381, 0.0673, 0.1961, 0.1835, 0.1521, 2.3013, 0.2225])
    sparsity = np.array([0.5, 0, 0.25, 0.75, 0.90, 0.95, 0.33])
    energy = np.array([0.208, 0.222, 0.227, 0.224, 0.2338, 0.2227, 0.2225])

    memory = np.array([27574, 40207, 27574, 27574, 27574, 27574, 27574])

    checkpoint = np.array([3640, 16064, 4954, 2218, 1250, 870, 3713])
    labels = ["50%", "Baseline", "25%", "75%", "90%", "95%", "75% Block"]

    # PlotSIZEoints
    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.scatter(sparsity, memory)
    for i, txt in enumerate(labels):
        ax.text(sparsity[i], memory[i], txt, rotation=40, rotation_mode="anchor")
    ax.set_xlabel("SPARSITY")
    ax.set_ylabel("MEMORY")

    ax = fig.add_subplot(224)
    ax.scatter(sparsity, energy)
    for i, txt in enumerate(labels):
        ax.text(sparsity[i], energy[i], txt, rotation=40, rotation_mode="anchor")
    ax.set_xlabel("SPARSITY")
    ax.set_ylabel("ENERGY")

    ax = fig.add_subplot(223)
    ax.scatter(sparsity, checkpoint)
    for i, txt in enumerate(labels):
        ax.text(sparsity[i], checkpoint[i], txt, rotation=40, rotation_mode="anchor")
    ax.set_xlabel("SPARSITY")
    ax.set_ylabel("CHECKPOINT SIZE")

    ax = fig.add_subplot(222)
    ax.scatter(sparsity, loss)
    for i, txt in enumerate(labels):
        ax.text(sparsity[i], loss[i], txt, rotation=40, rotation_mode="anchor")
    ax.set_xlabel("SPARSITY")
    ax.set_ylabel("VAL LOSS")

    # Save the plot to a temporary file
    filename = "MNIST_ENERGY_CHECKPOINT.png"
    fig.tight_layout()
    plt.savefig(filename)
    filepath = filename  # os.path.join(settings.MEDIA_ROOT, filename)

    # Open the image file for reading
    with open(filepath, "rb") as f:
        response = HttpResponse(f.read(), content_type="image/png")
        response["Content-Disposition"] = 'attachment; filename="' + filename + '"'
        return response
    return filename


class ComparisonDetailView(TemplateView):
    template_name = "comparison/comparison.html"
    page = PageSetup(
        title="Vergleich", description="Metriken zwischen Laufen vergleichen"
    )
    context = {"page": page.get_context()}

    def get(self, request, pk):
        comparison = Comparison.objects.get(id=pk)
        # get the ComparisonDetailsView
        # then the the etails and append to the runs
        self.context["runs"] = comparison.get_details()
        self.context["comparison"] = comparison
        self.context["add_form"] = AddRunForm()
        self.context["gpus"] = get_gpu_choices()
        return self.render_to_response(self.context)

    def post(self, request, pk):
        comparison = Comparison.objects.get(id=pk)
        add_form = AddRunForm(request.POST)
        if add_form.is_valid():
            # ad the run here
            if "run" in add_form.cleaned_data and add_form.cleaned_data["run"]:
                for run in add_form.cleaned_data["run"]:
                    comparison.runs["tensorflow:" + str(run.id)] = "tensorflow"
            if (
                "inference" in add_form.cleaned_data
                and add_form.cleaned_data["inference"]
            ):
                comparison.runs[
                    "inference:" + str(add_form.cleaned_data["inference"].id)
                ] = "inference"
            comparison.save()
            self.context["runs"] = comparison.get_details()
            self.context["add_form"] = AddRunForm()
            self.context["comparison_id"] = comparison.id
            self.context["gpus"] = get_gpu_choices()
            return self.render_to_response(self.context)


class SessionComparisonView(TemplateView):
    template_name = "comparison/comparison.html"
    page = PageSetup(
        title="Vergleich", description="Metriken zwischen Laufen vergleichen"
    )
    context = {"page": page.get_context()}

    def get(self, request):
        details = []
        if "comparison" in request.session:
            details = get_comparison_details(request.session["comparison"])
        # get the ComparisonDetailsView
        # then the the etails and append to the runs
        save_form = SaveSession()
        self.context["session"] = save_form
        self.context["runs"] = details
        self.context["gpus"] = get_gpu_choices()
        return self.render_to_response(self.context)

    def post(self, request):
        save_form = SaveSession(request.POST)
        if save_form.is_valid():
            comparison = Comparison(
                name=save_form.cleaned_data["name"],
                description=save_form.cleaned_data["description"],
            )
            comparison.runs = request.session["comparison"]
            comparison.save()
            clear_session(request)
            return redirect("comparisons:details", pk=comparison.id)
        self.context["session"] = save_form
        self.context["runs"] = get_comparison_details(request.session["comparison"])
        return self.render_to_response(self.context)
