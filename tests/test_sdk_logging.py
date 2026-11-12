"""The SDK process bootstrap preserves frozen-script semantics and job logging."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from csl_validation.sdk import sdk_environment, sdk_step_argv


class SDKLoggingTests(unittest.TestCase):
    def test_appliance_info_and_script_semantics_with_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/"request source"
            root.mkdir()
            (root/"sibling.py").write_text("VALUE = 'sibling imported'\n")
            script = root/"action.py"
            script.write_text("""import json,logging,sys
from sibling import VALUE
logging.getLogger('cerebras.cluster.client').info('Job id: wsjob-info')
print(json.dumps(dict(argv=sys.argv, file=__file__, name=__name__, value=VALUE)))
raise SystemExit(int(sys.argv[1]))
""")
            for kind, code in (("appliance", 0), ("appliance", 7), ("host", 0)):
                with self.subTest(kind=kind, code=code):
                    command = sdk_step_argv({"python":sys.executable}, script, [str(code), "two words"], kind)
                    result = subprocess.run(command, cwd=temporary, env=sdk_environment(),
                                            text=True, capture_output=True, timeout=15)
                    self.assertEqual(result.returncode, code, result.stderr)
                    self.assertEqual(json.loads(result.stdout), dict(argv=[str(script), str(code), "two words"],
                                     file=str(script), name="__main__", value="sibling imported"))
                    self.assertEqual("Job id: wsjob-info" in result.stderr, kind == "appliance")
                    self.assertIn("-B", command)


if __name__ == '__main__':
    unittest.main()
