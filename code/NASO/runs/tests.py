from unittest.mock import MagicMock, patch

from django.test import RequestFactory, TestCase

from neural_architecture.models.autokeras import AutoKerasRun
from runs.forms.trial import RerunTrialForm
from runs.models.training import CallbackFunction, Metric
from runs.views.trial import TrialView


# TODO check these tests
class TrialViewTestCase(TestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.run = AutoKerasRun.objects.create()
        self.metric = Metric.objects.create(instance_type="metric_type")
        self.callback = CallbackFunction.objects.create(instance_type="callback_type")
        self.run.model.metrics.add(self.metric)
        self.run.model.callbacks.add(self.callback)
        self.view = TrialView.as_view()
        self.url = f"/runs/{self.run.id}/trials/1/"

    def test_get_success(self):
        request = self.factory.get(self.url)
        with patch.object(self.run.model, "load_model") as mock_load_model:
            mock_trial = MagicMock()
            mock_trial.hyperparameters.values = {"param1": 1, "param2": "value2"}
            mock_oracle = MagicMock()
            mock_oracle.trials = {"1": mock_trial}
            mock_tuner = MagicMock()
            mock_tuner.oracle = mock_oracle
            mock_model = MagicMock()
            mock_model.tuner = mock_tuner
            mock_load_model.return_value = mock_model

            response = self.view(request, self.run.id, 1)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.template_name[0], "trial.html")
        self.assertEqual(response.context_data["hp"], {"param1": 1, "param2": "value2"})
        self.assertIsInstance(response.context_data["form"], RerunTrialForm)
        self.assertEqual(
            response.context_data["metric_configs"],
            [{"id": self.metric.instance_type, "arguments": None}],
        )
        self.assertEqual(
            response.context_data["callback_configs"],
            [{"id": self.callback.instance_type, "arguments": None}],
        )

    def test_get_exception(self):
        request = self.factory.get(self.url)
        with patch.object(self.run.model, "load_model") as mock_load_model:
            mock_load_model.side_effect = Exception("Model loading failed")

            response = self.view(request, self.run.id, 1)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.template_name[0], "trial.html")
        self.assertEqual(response.context_data["hp"], {})
        self.assertIsInstance(response.context_data["form"], RerunTrialForm)
        self.assertEqual(response.context_data["metric_configs"], [])
        self.assertEqual(response.context_data["callback_configs"], [])
