import argparse


def main():
    parser = argparse.ArgumentParser(prog="gcsrm_gesture")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_c = sub.add_parser("collect", help="Collect gesture samples")
    p_c.add_argument("--session", choices=["train", "test"], default="train")
    p_c.add_argument("--camera", type=int, default=0)

    p_t = sub.add_parser("train", help="Train and print 4-cell table")
    p_t.add_argument("--train", default="data/train/data.csv")
    p_t.add_argument("--test", default="data/test/data.csv")
    p_t.add_argument("--out", default="models")

    p_d = sub.add_parser("demo", help="Real-time inference")
    p_d.add_argument("--model", default="models/best_model.pkl")
    p_d.add_argument("--config", default="models/model_config.json")
    p_d.add_argument("--camera", type=int, default=0)

    args = parser.parse_args()

    if args.cmd == "collect":
        from src.data_collector import main as collect_main
        collect_main(session=args.session, camera=args.camera)
    elif args.cmd == "train":
        from src.train_model import main as train_main
        train_main(args.train, args.test, args.out)
    elif args.cmd == "demo":
        from src.live_demo import main as demo_main
        demo_main(args.model, args.config, args.camera)


if __name__ == "__main__":
    main()
