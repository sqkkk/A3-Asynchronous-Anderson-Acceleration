## test AdaSAM
python main.py --train-part-size 12000 --batch-size 6000 --test-part-size 1000 --test-batch-size 1000 --epochs 100 --lr 0.1 --log-interval 10

## test AdaSAM-VR
python tryvr_avggd.py --train-part-size 12000 --batch-size 6000 --test-part-size 1000 --test-batch-size 1000 --epochs 100 --lr 0.1 --log-interval 10

## test distributed SGD / distributed SAM
python distributed_main.py --alg dsgd --partition iid --num-workers 10 --train-part-size 12000 --batch-size 600 --test-part-size 1000 --test-batch-size 1000 --epochs 5 --lr 0.1 --log-interval 1
python distributed_main.py --alg dsam --partition iid --num-workers 10 --train-part-size 12000 --batch-size 600 --test-part-size 1000 --test-batch-size 1000 --epochs 5 --lr 0.1 --log-interval 1

## test async distributed SGD / async distributed SAM
python async_distributed_main.py --alg asyncsgd --partition iid --num-workers 10 --train-part-size 12000 --batch-size 600 --test-part-size 1000 --test-batch-size 1000 --epochs 5 --lr 0.1 --log-interval 10
python async_distributed_main.py --alg asyncsam --partition iid --num-workers 10 --train-part-size 12000 --batch-size 600 --test-part-size 1000 --test-batch-size 1000 --epochs 5 --lr 0.1 --log-interval 10

## test async distributed SAM with SCAFFOLD-style control variates
python async_distributed_main.py --alg asyncsam_cv --partition iid --num-workers 10 --train-part-size 12000 --batch-size 600 --test-part-size 1000 --test-batch-size 1000 --epochs 10 --lr 0.1 --log-interval 20 --cv-server-lr 0.5
