@echo off

python .\train_segmentation.py --mode=e2e --data_size=2000 --epochs=100 --batch_size=50

python .\train_segmentation.py --mode=frozen --data_size=2000 --epochs=100 --batch_size=50

python .\train_segmentation.py --mode=e2e --data_size=160 --epochs=200 --batch_size=10

python .\train_segmentation.py --mode=frozen --data_size=160 --epochs=200 --batch_size=10

python .\train_segmentation.py --mode=e2e --data_size=80 --epochs=300 --batch_size=10

python .\train_segmentation.py --mode=frozen --data_size=80 --epochs=300 --batch_size=10

python .\train_segmentation.py --mode=e2e --data_size=40 --epochs=500 --batch_size=10

python .\train_segmentation.py --mode=frozen --data_size=40 --epochs=500 --batch_size=10

python .\train_segmentation.py --mode=e2e --data_size=20 --epochs=800 --batch_size=10

python .\train_segmentation.py --mode=frozen --data_size=20 --epochs=800 --batch_size=10

python .\train_segmentation.py --mode=e2e --data_size=10 --epochs=1200 --batch_size=5

python .\train_segmentation.py --mode=frozen --data_size=10 --epochs=1200 --batch_size=5

pause