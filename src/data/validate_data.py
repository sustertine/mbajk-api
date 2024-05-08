import great_expectations as ge
from great_expectations.checkpoint.types.checkpoint_result import CheckpointResult


def main():
    context = ge.get_context()
    result: CheckpointResult = context.run_checkpoint(checkpoint_name="mbajk_checkpoint")
    if not result["success"]:
        raise ValueError("[Validate]: Checkpoint validation failed!")
    else:
        print("[Validate]: Checkpoint validation passed!")


if __name__ == "__main__":
    main()