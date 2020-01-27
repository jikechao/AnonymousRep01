# -*-coding:UTF-8-*-
import os
import datetime
import argparse

if __name__ == '__main__':
    start = datetime.datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str)
    # parser.add_argument("--target", "-target", help="adv", type=str)
    args = parser.parse_args()

    for i in range(500):
        if args.d == "mnist":
            coverage_metrics = "python coverage_metrics.py -d mnist -lsa_lower -34 -lsa_upper 310 -dsa_lower 0 -dsa_upper 3 -count {} -num_classes 10".format(i)
        elif args.d == "cifar10":
            coverage_metrics = "python coverage_metrics.py -d cifar -lsa_lower -40 -lsa_upper 128 -dsa_lower 0 -dsa_upper 2 -count {} -num_classes 10".format(i)
        elif args.d == "vgg16":
            coverage_metrics = "python coverage_metrics.py -d cifar -lsa_lower -300 -lsa_upper 1000 -dsa_lower 0 -dsa_upper 4 -count {} -num_classes 10".format(i)
        elif args.d == "cifar100":
            coverage_metrics = "python coverage_metrics.py -d cifar100 -lsa_lower -350 -lsa_upper 62000 -dsa_lower 0 -dsa_upper 4 -count {} -num_classes 100".format(i)
        elif args.d == "driving_ori":
            coverage_metrics = "python coverage_metrics.py -d driving_ori -lsa_lower 20 -lsa_upper 65 -count {}".format(i)

        elif args.d == "driving_drop":
            coverage_metrics = "python coverage_metrics.py -d driving_drop -lsa_lower -11 -lsa_upper 37 -count {}".format(i)
        elif args.d == "imagenet":
            coverage_metrics = "python coverage_metrics.py -d imagenet -lsa_lower 0 -lsa_upper 5000 -dsa_lower 0 -dsa_upper 4 -count {} -num_classes 1000".format(i)


        os.system(coverage_metrics)

    elapsed = (datetime.datetime.now() - start)
    print("Time used: ", elapsed)