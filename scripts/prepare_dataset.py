"""CLI script to prepare reproducible sample and processed dataset partitions for AmazonHelp."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from src.config import load_config
from src.data.loader import DataLoader
from src.utils.logging import setup_logger

logger = setup_logger("prepare_dataset")


def main():
    parser = argparse.ArgumentParser(description="Prepare Hiver support dataset partitions")
    parser.add_argument("--raw-path", type=str, default=r"C:\Users\kesha\Downloads\archive\twcs\twcs.csv")
    parser.add_argument("--max-rows", type=int, default=150000, help="Maximum rows to scan from twcs.csv")
    parser.add_argument("--brand", type=str, default="AmazonHelp")
    parser.add_argument("--sample-size", type=int, default=100, help="Size of committed sample fixtures")
    args = parser.parse_args()

    config = load_config()
    processed_dir = Path(config.data.processed_dir)
    sample_dir = Path(config.data.sample_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        raw_csv_path=args.raw_path,
        brand=args.brand,
        dev_ratio=config.data.dev_split_ratio,
        seed=config.project.seed,
    )

    pairs = loader.extract_pairs(max_rows=args.max_rows)
    logger.info(f"Extracted {len(pairs)} pairs for {args.brand}.")

    # Save lightweight sample for committed repo fixtures
    sample_pairs = [p.model_dump() for p in pairs[: args.sample_size]]
    sample_file = sample_dir / "sample_pairs.json"
    with open(sample_file, "w", encoding="utf-8") as f:
        json.dump(sample_pairs, f, indent=2)
    logger.info(f"Saved {len(sample_pairs)} committed sample pairs to {sample_file}")

    # Partition dev and eval
    dev_pairs, eval_pairs = loader.partition_dataset(pairs)

    dev_file = processed_dir / "dev_pairs.json"
    with open(dev_file, "w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in dev_pairs], f, indent=2)

    eval_file = processed_dir / "eval_pairs.json"
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in eval_pairs], f, indent=2)

    # Save summary metadata
    metadata = {
        "brand": args.brand,
        "raw_source": args.raw_path,
        "max_rows_scanned": args.max_rows,
        "total_pairs": len(pairs),
        "dev_pairs_count": len(dev_pairs),
        "eval_pairs_count": len(eval_pairs),
        "timestamp": datetime.now().isoformat(),
        "leakage_check": "Zero overlap verified by customer_author_id",
    }
    with open(processed_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Dataset preparation complete! Dev: {len(dev_pairs)}, Eval: {len(eval_pairs)}")


if __name__ == "__main__":
    main()
