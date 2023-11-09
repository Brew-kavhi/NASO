from django.test import RequestFactory, TestCase
from django.urls import reverse_lazy

from dashboard.views.dashboard import Dashboard


class DashboardViewTestCase(TestCase):
    def setUp(self):
        self.factory = RequestFactory()

    def test_dashboard_view(self):
        url = reverse_lazy("dashboard")
        request = self.factory.get(url)
        response = Dashboard.as_view()(request)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "dashboard/dashboard.html")
