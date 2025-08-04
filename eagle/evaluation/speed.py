import argparse
import json


def compute_speed(path: str) -> float:
    """Compute tokens per second from a benchmark jsonl file."""
    total_tokens = 0
    total_time = 0.0
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            tokens = sum(data["choices"][0]["new_tokens"])
            times = sum(data["choices"][0]["wall_time"])
            total_tokens += tokens
            total_time += times
    return total_tokens / total_time if total_time > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Compute generation speed from answer files")
    parser.add_argument("--file", required=True, help="Answer file produced by evaluation script")
    parser.add_argument("--baseline-file", help="Optional baseline file for comparison")
    args = parser.parse_args()

    speed = compute_speed(args.file)
    print(f"speed: {speed}")

    if args.baseline_file:
        baseline_speed = compute_speed(args.baseline_file)
        print(f"baseline speed: {baseline_speed}")
        if baseline_speed > 0:
            print(f"ratio: {speed / baseline_speed}")


if __name__ == "__main__":
    main()
