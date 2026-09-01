import base64
import json

from mjlab_microduck.hf_jobs import BOOTSTRAP, _encode_train_args


def test_train_argv_transport_preserves_shell_sensitive_arguments():
    args = [
        "Mjlab-BeakThrow-Flat-MicroDuck",
        "--agent.load-run",
        "^bootstrap$",
        "--agent.run-name",
        "name with spaces and 'quotes' $(ignored)",
    ]

    encoded = _encode_train_args(args)
    decoded = json.loads(base64.urlsafe_b64decode(encoded))

    assert decoded == args


def test_bootstrap_executes_decoded_argv_without_shell_reparsing():
    assert 'os.environ["TRAIN_ARGS_B64"]' in BOOTSTRAP
    assert "os.execvp(argv[0], argv)" in BOOTSTRAP
    assert "uv run train $TRAIN_ARGS" not in BOOTSTRAP
