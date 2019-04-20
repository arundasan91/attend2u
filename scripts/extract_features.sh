echo "=== STARTING DATA PREPROCESS ==="
echo "=== Loading TF Module ==="
module load tensorflow/1.7_py2_gpu
echo "===%%%### Generate dataset being ###%%%==="
python generate_dataset.py
echo "===%%%### Generate dataset complete ###%%%==="
echo "=== Begin train feat extraction #1==="
python cnn_feature_extractor.py --gpu_id "0,1,2,3,4,5,6,7" --batch_size 512 --input_fname ../data/caption_dataset/train.txt
echo "=== End train feat extraction #1==="
echo "=== Begin test1 feat extraction #1==="
python cnn_feature_extractor.py --gpu_id "0,1,2,3,4,5,6,7" --batch_size 512 --input_fname ../data/caption_dataset/test1.txt
echo "=== End test1 feat extraction #1==="
echo "=== Begin test2 feat extraction #1"
python cnn_feature_extractor.py --gpu_id "0,1,2,3,4,5,6,7" --batch_size 512 --input_fname ../data/caption_dataset/test2.txt
echo "=== End test2 feat extraction #1"
echo "=== Begin train feat extraction #2==="
python cnn_feature_extractor.py --gpu_id "0,1,2,3,4,5,6,7" --batch_size 512 --input_fname ../data/hashtag_dataset/train.txt
echo "=== End train feat extraction #2==="
echo "=== Begin test1 feat extraction #2==="
python cnn_feature_extractor.py --gpu_id "0,1,2,3,4,5,6,7" --batch_size 512 --input_fname ../data/hashtag_dataset/test1.txt
echo "=== End test1 feat extraction #2==="
echo "=== Begin test2 feat extraction #2==="
python cnn_feature_extractor.py --gpu_id "0,1,2,3,4,5,6,7" --batch_size 512 --input_fname ../data/hashtag_dataset/test2.txt
echo "=== End test2 feat extraction #2==="
echo "%%% ALL DONE %%%"
