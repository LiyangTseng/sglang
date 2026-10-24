"""Rust server weight-update controls: /pause_generation, /continue_generation,
/update_weights_from_disk and /update_weight_version, against a real scheduler.

Their responses and status codes are Python's; the pause gate, generation
leases and weight writer live in the Rust frontend instead of the
TokenizerManager.
"""

import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_rust_server_built,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

PROMPT = {
    "text": "The capital of France is",
    "sampling_params": {"temperature": 0, "max_new_tokens": 16},
}


@unittest.skipUnless(
    is_rust_server_built(),
    "embedded rust server extension not built (e.g. AMD suite)",
)
class TestRustWeightUpdateControls(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={"SGLANG_RUST_SERVER": "1"},
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)

    def tearDown(self):
        # Leave the server generating for the next case, even after a failure.
        requests.post(self.base_url + "/continue_generation", json={}, timeout=60)

    def _post(self, path, body):
        return requests.post(self.base_url + path, json=body, timeout=120)

    def _generate(self, body=PROMPT):
        response = self._post("/generate", body)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_generation_waits_while_paused_and_resumes_on_continue(self):
        for mode in ("in_place", "retract"):
            with self.subTest(mode=mode):
                response = self._post("/pause_generation", {"mode": mode})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(
                    response.json(),
                    {"message": "Generation paused successfully.", "status": "ok"},
                )
                with ThreadPoolExecutor(1) as pool:
                    waiting = pool.submit(self._generate)
                    time.sleep(2)
                    self.assertFalse(waiting.done(), "generated while paused")
                    response = self._post("/continue_generation", {})
                    self.assertEqual(
                        response.json(),
                        {
                            "message": "Generation continued successfully.",
                            "status": "ok",
                        },
                    )
                    self.assertTrue(waiting.result(timeout=60)["text"])

    def test_abort_pause_aborts_in_flight_generation(self):
        long_request = {
            "text": "Count from one to one thousand:",
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 4096,
                "ignore_eos": True,
            },
        }
        with ThreadPoolExecutor(1) as pool:
            running = pool.submit(self._post, "/generate", long_request)
            time.sleep(1)
            response = self._post("/pause_generation", {"mode": "abort"})
            self.assertEqual(response.status_code, 200, response.text)
            # Aborted, not run to max_new_tokens: the frontend's abort-all
            # renders a cancelled generation as 499.
            aborted = running.result(timeout=30)
            if aborted.status_code == 200:
                finish_reason = aborted.json()["meta_info"]["finish_reason"]
                self.assertEqual(finish_reason["type"], "abort")
            else:
                self.assertEqual(aborted.status_code, 499, aborted.text)
        self._post("/continue_generation", {})
        self.assertTrue(self._generate()["text"])

    def test_update_weights_from_disk_reloads_and_relabels(self):
        before = self._generate()["text"]
        for paused in (False, True):
            with self.subTest(paused=paused):
                if paused:
                    self._post("/pause_generation", {"mode": "retract"})
                version = f"reload-{paused}"
                response = self._post(
                    "/update_weights_from_disk",
                    {"model_path": self.model, "weight_version": version},
                )
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                self.assertTrue(body["success"])
                self.assertTrue(
                    body["message"].endswith(f" Weight version updated to {version}.")
                )
                info = requests.get(self.base_url + "/model_info").json()
                self.assertEqual(info["weight_version"], version)
                self.assertEqual(info["model_path"], self.model)
                self._post("/continue_generation", {})
                # Same checkpoint, greedy decoding: the same text.
                self.assertEqual(self._generate()["text"], before)

    def test_failed_update_reports_400_and_keeps_serving(self):
        response = self._post(
            "/update_weights_from_disk",
            {"model_path": "/nonexistent/checkpoint", "weight_version": "never"},
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertFalse(response.json()["success"])
        info = requests.get(self.base_url + "/model_info").json()
        self.assertNotEqual(info["weight_version"], "never")
        self.assertEqual(info["model_path"], self.model)
        self.assertTrue(self._generate()["text"])

    def test_update_weight_version(self):
        response = self._post("/update_weight_version", {"new_version": "v-e2e"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json(),
            {
                "success": True,
                "message": "Weight version updated to v-e2e",
                "new_version": "v-e2e",
            },
        )
        info = requests.get(self.base_url + "/model_info").json()
        self.assertEqual(info["weight_version"], "v-e2e")


if __name__ == "__main__":
    unittest.main(verbosity=3)
