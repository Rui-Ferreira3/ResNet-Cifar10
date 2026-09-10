#!/bin/bash
# ./run.sh -t -s -m modelA.th

MODE="test"
PROF_FILE="output.prof"
MODEL="model.th"
ARCH="resnet20"
RUNS=1
EPOCHS=200
LR=0.1

ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--train)
            MODE="train"
            ARGS="$ARGS --train"
            shift 1
            ;;
        -p|--prof)
            if [ -z "$2" ]; then
                echo "Error: --prof requires an argument"
                return 2>/dev/null || exit 1
            fi
            MODE="prof"
            PROF_FILE=$2
            shift 2
            ;;
        -v|--viz)
            if [ -z "$2" ]; then
                echo "Error: --viz requires an argument"
                return 2>/dev/null || exit 1
            fi
            PROF_FILE=$2
            mkdir -p ./prof
            snakeviz ./prof/$PROF_FILE
            ;;
        -m|--model)
            if [ -z "$2" ]; then
                echo "Error: --model requires an argument"
                return 2>/dev/null || exit 1
            fi
            MODEL=$2
            shift 2
            ;;
        -a|--arch)
            if [ -z "$2" ]; then
                echo "Error: --arch requires an argument"
                return 2>/dev/null || exit 1
            fi
            ARCH=$2
            shift 2
            ;;
        -s|--sim)
            ARGS="$ARGS --sim --hist"
            shift 1
            ;;
        -r|--runs)
            if [ -z "$2" ]; then
                echo "Error: --runs requires an argument"
                return 2>/dev/null || exit 1
            fi
            ARGS="$ARGS --runs $2"
            shift 2
            ;;
        -e|--epochs)
            if [ -z "$2" ]; then
                echo "Error: --epochs requires an argument"
                return 2>/dev/null || exit 1
            fi
            EPOCHS=$2
            shift 2
            ;;
        -lr| --learning_rate)
            if [ -z "$2" ]; then
                echo "Error: --epochs requires an argument"
                return 2>/dev/null || exit 1
            fi
            LR=$2
            shift 2
            ;;
        -h|--help)
            echo "Usage: source run.sh [options]"
            echo "Options:"
            echo "  -t, --test           Run the test without profiling (default is train)"
            echo "  -p, --prof           Profile the test run and save to a file (default: output.prof)"
            echo "  -v, --viz            Visualize the profile data from a file (default: output.prof)"
            echo "  -m, --model MODEL    Specify the trained model file (default: model.th)"
            echo "  -a, --arch ARCH      Specify the architecture to test (default: resnet20)"
            echo "  -s, --sim            Use accelerator simulator for inference"
            echo "  -r, --runs           Number of runs to perform (default: 1)"
            echo "  -e, --epochs         Number of epochs for training (default: 200)"
            echo "  -lr, --learning_rate Learning rate (default: 0.1)"
            echo "  -d, --dataset        Dataset to use: cifar10, mnist or imagenet (default: cifar10)"
            return 2>/dev/null || exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            return 2>/dev/null || exit 1
            ;;
    esac
done

ARGS="$ARGS --arch $ARCH"

if [ "$MODE" = "train" ]; then
    python run.py $ARGS  --model ./save_$ARCH/$MODEL  --lr $LR --save-dir ./save_$ARCH --epochs $EPOCHS 
elif [ "$MODE" = "test" ]; then
    python run.py $ARGS
elif [ "$MODE" = "profile" ]; then
    mkdir -p ./prof
    python -m cProfile -o ./prof/$PROF_FILE trainer.py $ARGS
else
    echo "Invalid mode: $MODE"
    return 2>/dev/null || exit 1
fi

