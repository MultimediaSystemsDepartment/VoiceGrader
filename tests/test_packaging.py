import json
import pathlib
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class PackagingTests(unittest.TestCase):
    def test_compose_isolates_gpu_work_and_shares_rlhf_state(self):
        result = subprocess.run(
            [
                "docker", "compose", "--project-directory", str(ROOT),
                "--profile", "training", "config", "--format", "json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        services = json.loads(result.stdout)["services"]
        self.assertEqual(set(services), {"rlhf", "training", "asr", "tts-api", "tts-piper"})
        self.assertNotIn("deploy", services["rlhf"])
        for name in ("tts-api", "tts-piper"):
            self.assertNotIn("deploy", services[name])
            self.assertNotIn("ports", services[name])
        self.assertEqual(services["training"]["profiles"], ["training"])
        self.assertEqual(
            services["training"]["command"],
            ["python3.12", "scripts/coordinator.py"],
        )
        self.assertEqual(services["rlhf"]["volumes"], services["training"]["volumes"])
        for service in services.values():
            build = service["build"]
            context = pathlib.Path(build["context"])
            self.assertTrue((context / build["dockerfile"]).is_file())
            self.assertFalse(
                any("docker.sock" in volume["target"] for volume in service["volumes"])
            )
        for name in ("training", "asr"):
            devices = services[name]["deploy"]["resources"]["reservations"]["devices"]
            self.assertEqual(devices[0]["capabilities"], ["gpu"])

    def test_runtime_state_is_ignored(self):
        paths = [
            "_data/rlhf/archive.zip",
            "rlhf/config/admin_password.json",
            "rlhf/_outputs/feedback.sqlite",
            "asr/data/models/model.bin",
            ".env",
        ]
        result = subprocess.run(
            ["git", "-C", str(ROOT), "check-ignore", "-z", "--stdin"],
            input=("\0".join(paths) + "\0").encode("utf-8"),
            capture_output=True,
            check=True,
        )
        self.assertEqual(
            set(result.stdout.decode("utf-8").strip("\0").split("\0")),
            set(paths),
        )


if __name__ == "__main__":
    unittest.main()
