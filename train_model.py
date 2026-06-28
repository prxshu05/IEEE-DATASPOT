"""
train_model.py
Entry point to train the VeriLens fake news detection model.

Usage:
    python train_model.py --train data/train.csv --val data/val.csv \
                          --text_col text --label_col label \
                          --epochs 3 --batch_size 16

Dataset CSV format expected:
    text,label
    "Breaking news: ...",1
    "The government today ...",0

Label encoding:
    0 = REAL
    1 = FAKE
    (Strings like "fake"/"real" are also accepted and auto-converted.)
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from modules.model import train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VeriLens AI fake news model")

    parser.add_argument(
        "--train", required=True,
        help="Path to training CSV file"
    )
    parser.add_argument(
        "--val", default=None,
        help="Path to validation CSV file (optional; auto-splits from train if not provided)"
    )
    parser.add_argument(
        "--text_col", default="text",
        help="Column name for news text (default: 'text')"
    )
    parser.add_argument(
        "--label_col", default="label",
        help="Column name for labels (default: 'label')"
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Training batch size (default: 16)"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5,
        help="Learning rate (default: 2e-5)"
    )

    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════╗
║       VeriLens AI — Model Training       ║
╠══════════════════════════════════════════╣
║  Dataset  : {args.train:<28} ║
║  Val Set  : {(args.val or 'auto-split'):<28} ║
║  Text col : {args.text_col:<28} ║
║  Label col: {args.label_col:<28} ║
║  Epochs   : {args.epochs:<28} ║
║  Batch sz : {args.batch_size:<28} ║
╚══════════════════════════════════════════╝
    """)

    train_model(
        train_path=args.train,
        val_path=args.val,
        text_col=args.text_col,
        label_col=args.label_col,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    print("\n✅ Training complete. Model saved to models/fake_news_model/")
    print("▶️  Run the app with:  streamlit run app.py")
