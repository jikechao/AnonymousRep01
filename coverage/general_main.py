# -*-coding:UTF-8-*-
import os
import datetime
import argparse

if __name__ == '__main__':
    start = datetime.datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str)
    parser.add_argument("--target", "-target", help="adv", type=str, default="cw")
    args = parser.parse_args()

    for i in range(500):
        if args.d == "mnist":
            coverage_metrics = "python coverage_metrics.py -d mnist -lsa_lower -35 -lsa_upper 1000 -dsa_lower 0 -dsa_upper 6 -count {} -target {} -num_classes 10".format(i, args.target)
        elif args.d == "cifar10":
            coverage_metrics = "python coverage_metrics.py -d cifar -lsa_lower -40 -lsa_upper 300 -dsa_lower 0 -dsa_upper 4 -count {} -target {} -num_classes 10".format(i, args.target)
        elif args.d == "vgg16":
            coverage_metrics = "python coverage_metrics.py -d cifar -lsa_lower -300 -lsa_upper 1000 -dsa_lower 0 -dsa_upper 7 -count {} -target {} -num_classes 10".format(i, args.target)
        elif args.d == "cifar100":
            coverage_metrics = "python coverage_metrics.py -d cifar100 -lsa_lower -350 -lsa_upper 120000 -dsa_lower 0 -dsa_upper 5 -count {} -target {} -num_classes 100".format(i, args.target)
        elif args.d == "driving_ori":
            if args.target == "black":
                coverage_metrics = "python coverage_metrics.py -d driving_ori -lsa_lower 20 -lsa_upper 673 -count {} -target {}".format(
                    i, args.target)
            elif args.target == "light":
                coverage_metrics = "python coverage_metrics.py -d driving_ori -lsa_lower 20 -lsa_upper 5800 -count {} -target {}".format(
                    i, args.target)

        elif args.d == "driving_drop":
            if args.target == "black":
                coverage_metrics = "python coverage_metrics.py -d driving_drop -lsa_lower -11 -lsa_upper 543 -count {} -target {}".format(
                    i, args.target)
            elif args.target == "light":
                coverage_metrics = "python coverage_metrics.py -d driving_drop -lsa_lower -11 -lsa_upper 25200 -count {} -target {}".format(
                    i, args.target)

        elif args.d == "imagenet":
            coverage_metrics = "python coverage_metrics.py -d imagenet -lsa_lower 0 -lsa_upper 5000 -dsa_lower 0 -dsa_upper 4 -count {} -target {} -num_classes 1000".format(i, args.target)
        elif args.d == "speech":
            coverage_metrics = "python coverage_metrics_speech.py -d speech -lsa_lower 0 -lsa_upper 5000 -dsa_lower 0 -dsa_upper 4 -count {} -target {} -num_classes 29".format(
                i, args.target)


        os.system(coverage_metrics)

    elapsed = (datetime.datetime.now() - start)
    print("Time used: ", elapsed)