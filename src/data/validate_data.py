import os

import great_expectations as ge
from great_expectations.checkpoint.types.checkpoint_result import CheckpointResult


def main():
    context = ge.get_context()
    result: CheckpointResult = context.run_checkpoint(checkpoint_name="mbajk_checkpoint")
    if not result["success"]:
        print(result)
        raise ValueError("Data validation failed")

    print("Data validation passed!")

if __name__ == "__main__":
    main()