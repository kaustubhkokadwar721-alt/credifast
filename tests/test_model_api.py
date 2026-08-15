import unittest

from fastapi.testclient import TestClient

from credifast.api import app, model_runtime_dependency


class _RuntimeStub:
    def status(self):
        return {"ready": True, "production_approved": False}

    def profiles(self):
        return [{"key": "strong_full", "application_id": 178290}]

    def input_schema(self):
        return {"schema_version": "1.0.0", "selected_factor_count": 257}

    def score(self, application_id, **kwargs):
        if application_id == 999:
            raise ValueError("application ID 999 was not found")
        return {
            "application_id": application_id,
            "review_route": "STANDARD_REVIEW",
            "overrides_received": kwargs["overrides"],
        }


class ModelApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app.dependency_overrides[model_runtime_dependency] = lambda: _RuntimeStub()
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        app.dependency_overrides.clear()

    def test_status_and_profiles(self) -> None:
        self.assertTrue(self.client.get("/v1/model/status").json()["ready"])
        profiles = self.client.get("/v1/model/applicants").json()
        self.assertEqual(profiles["count"], 1)

        schema = self.client.get("/v1/model/input-schema").json()
        self.assertEqual(schema["selected_factor_count"], 257)

    def test_score_accepts_controlled_overrides(self) -> None:
        response = self.client.post(
            "/v1/model/score",
            json={"application_id": 178290, "annual_income": 150000.0, "explain": False},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["overrides_received"]["annual_income"], 150000.0)

    def test_unknown_applicant_maps_to_422(self) -> None:
        response = self.client.post("/v1/model/score", json={"application_id": 999})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
